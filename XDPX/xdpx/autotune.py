import os
import sys
import math
import multiprocessing
import matplotlib
matplotlib.use('Agg')  # prevent using the default interactive XWindows backend on the server
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from typing import List
from functools import lru_cache
from xdpx.options import Options, Argument, undefined, Arg
from xdpx.utils import io, run_script, log_to_file, get_total_steps, parse_relative_config
from xdpx.bootstrap import bootstrap
from xdpx.train import build_train_options, cli_main as train_cli
from xdpx.automl.space import load_space
from xdpx.automl.tuners import tuners, SpaceExhausted


def cli_main(argv=sys.argv):
    options = Options()

    options.register(
        Argument('space_config', required=True, type=str, post_process=parse_relative_config),
        Argument('base_config', type=str, post_process=parse_relative_config),
        Argument('save_dir', required=True, unique=True, validate=[
            lambda value: (not io.exists(value), f'Automl save_dir exists. Please change save_dir or remove it manually'),
            lambda value: (io.is_writable(value), 'save_dir is not writable'),
        ]),
        # TODO: implement resumed training (by update previous runs to Archive)
        Argument('resume', default=False),
        Argument('max_trials', type=int, required=True),
        Argument('seed', type=int, default=1),
        Argument('ref_runs', default=[], type=List[str],
                 validate=lambda value: (all(io.exists(os.path.join(path, 'valid.log.tsv')) for path in value),
                                         'invalid train paths in ref_runs'),
                 ),
    )
    # TODO support resume tuning
    options.register(
        # support dynamic resource allocation in the future
        Argument('available_rss', default=1),
        Argument('min_rss', default=1),
        domain='resources',
    )
    options.register(
        Argument('tuner', doc='name of the HPO algorithm', validate=lambda value: value in tuners, required=True,
                 register=lambda value: tuners[value].register),
        domain='tuner'
    )
    Commander.register(options)
    
    def validate_configs(args: Arg):
        space = load_space(args.space_config)
        options = build_train_options()
        configs = options.load_configs_from_file(args.base_config)
        if len(configs) > 1:
            return False, 'base_config cannot have multiple configs.'
        config = configs[0]
        # save_dir does not matter
        config['save_dir'] = os.path.join(args.save_dir, 'baseline')
        config.update(required_config)
        base_args = options.parse_dict(config)
        for param in space:
            if param.name not in options:
                return False, f'Param {param.name} not defined in base_config.'
            if param.name == 'max_update' or param.name == 'max_epoch':
                return False, 'tuning max steps is not supported. Exclude max_update and max_epoch from search space.'
            argument = options[param.name]
            for value in param.values:
                value = argument.finalize(value)
                if value is undefined:
                    argument.post_process(base_args)
        return True

    options.add_global_constraint(validate_configs)
    bootstrap(options, main, __file__, argv)


required_config = dict(
    save=True,
    save_full_checkpoint=True,
    save_best_only=True,
    save_last_only=True,
    save_above_score=None,
    overwrite=False,
    resume=False,
)


def main(cli_args):
    # TODO support multi-worker tuning
    io.makedirs(cli_args.save_dir)
    config_dir = os.path.join(cli_args.save_dir, 'configs')
    plot_path = os.path.join(cli_args.save_dir, 'overview.png')
    io.makedirs(config_dir)
    Options.save_hjson(cli_args, os.path.join(cli_args.save_dir, 'args.hjson'))
    log_to_file(os.path.join(cli_args.save_dir, 'log.txt'))

    # load space config
    space = load_space(cli_args.space_config)
    # build tuner and commander
    archive = Archive(cli_args)
    tuner = tuners[cli_args.tuner](cli_args, archive, space)
    commander = Commander(cli_args, archive, tuner)
    # load ref_runs if available
    for ref_run in cli_args.ref_runs:
        # TODO: simulate previous runs
        # plot_runs(archive, plot_path)
        ...
    # start the major loop
    while True:
        orders = commander.give_orders()
        commands = []
        terminated = False
        for order in orders:
            if isinstance(order, Terminate):
                terminated = True
                break
            elif isinstance(order, NewRun):
                config_path = os.path.join(config_dir, f'{order.name}.hjson')
                Options.save_hjson(order.args, config_path)
                commands.append((order.name, config_path))
            elif isinstance(order, Continue):
                config_path = os.path.join(config_dir, f'{order.name}.hjson')
                config = Options.load_hjson(config_path)
                config.update(dict(
                    resume=True,
                    train_steps=order.steps,
                ))
                Options.save_hjson(config, config_path)
                commands.append((order.name, config_path))
            else:
                raise RuntimeError(f'Unexpected order type {order.__class__.__name__}')
        if not commands:
            break
        # run a trial with (maybe) limited steps
        mp = multiprocessing.get_context('spawn')
        processes = []
        for _, command in commands:
            p = mp.Process(
                target=spawned_main,
                args=(train_cli, ['x-train', command]),
                daemon=False,
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f'\nTraining exits with non-zero exit code {p.exitcode}.')
        # update results
        archive.update(commands)
        tuner.update()
        plot_runs(archive, plot_path)
        if terminated:
            break
    results = aggregate_results(cli_args.save_dir)
    if len(results) > 0:
        with io.open(os.path.join(cli_args.save_dir, 'results.tsv'), 'w') as f:
            results.to_csv(f, sep='\t', index=False)


class Commander:
    """A HyperBand commander"""
    @staticmethod
    def register(options):
        options.register(
            Argument('assess_start_step', default=0.0),
            Argument('assess_interval', required=True, type=float),
            Argument('parallel_runs', required=True, type=int, validate=lambda value: value > 0),
            Argument('min_ref', default=1),
            Argument('cull_keep_ratio', default=0.5),
            domain='commander'
        )
        options.add_global_constraint(lambda args: (
            math.ceil(args.cull_keep_ratio * args.parallel_runs) < args.parallel_runs,
            'cull_keep_ratio is too large to cull any runs'
        ))
        options.add_global_constraint(lambda args: args.parallel_runs > args.min_ref)

    def __init__(self, args, archive, tuner):
        self.args = args
        self.archive = archive
        self.tuner = tuner
        self.next_trial_id = 0
        self.task_queue = []

        # load base config
        base_config = Options.load_configs_from_file(args.base_config)[0]
        base_options = build_train_options()
        base_config['save_dir'] = os.path.join(args.save_dir, 'baseline')
        base_config.update(required_config)
        self.base_options = base_options
        self.base_config = base_config
        self.base_args = deepcopy(base_options).parse_dict(base_config)

    def give_orders(self):
        if not self.task_queue:
            self.plan_ahead()
            assert self.task_queue
        return [self.task_queue.pop(0) for _ in range(self.args.available_rss // self.args.min_rss) if self.task_queue]

    def plan_ahead(self):
        active_runs = self.archive.active_runs
        if not active_runs:
            if self.next_trial_id >= self.args.max_trials:
                print('| Max trials reached. Automl terminates.')
                return self.task_queue.append(Terminate())
            for _ in range(self.args.parallel_runs):
                initial_assess = self.args.assess_start_step if self.args.assess_start_step \
                    else self.args.assess_interval
                name = f'trial_{self.next_trial_id}'
                options = deepcopy(self.base_options)
                config = deepcopy(self.base_config)
                config['save_dir'] = os.path.join(self.args.save_dir, name)
                if self.archive.best_run:
                    config['viz_ref_dir'] = os.path.join(self.args.save_dir, self.archive.best_run)
                # generate a new config from tuner
                try:
                    config = self.tuner.suggest(config)
                except SpaceExhausted:
                    print('| Exhausted space reported by tuner.')
                    if not self.task_queue:
                        print('| AutoML terminates due to exhausted space.')
                        self.task_queue.append(Terminate())
                    return
                if self.base_args.max_update is not None and ('batch_size' in config or 'update_freq' in config):
                    base_bs = self.base_args.batch_size * self.base_args.update_freq
                    current_bs = config['batch_size'] * config.get('update_freq', 1)
                    config['max_update'] = round(config['max_update'] / current_bs * base_bs)
                config['train_steps'] = initial_assess
                print(f'| create a new run "{name}" with {initial_assess*100:.0f}% steps')
                args = options.parse_dict(config)
                order = NewRun(name, args)

                self.task_queue.append(order)
                self.next_trial_id += 1
                if self.next_trial_id >= self.args.max_trials:
                    break
            return
        for run in active_runs:
            progress = self.archive.progress(run)
            peer_scores = self.archive.scores_at(progress)
            num_peers = len(peer_scores)
            peer_scores.sort(reverse=self.archive.ascending_metric)
            score = self.archive.get_score(run)
            index = peer_scores.index(score)
            passing_rank = max(math.ceil(num_peers * self.args.cull_keep_ratio), self.args.min_ref)
            if index >= passing_rank:
                print(f'| deactivate "{run}" due to low rank ({index}/{num_peers}) with passing rank {passing_rank}')
                self.archive.deactivate(run)
        active_runs = self.archive.active_runs
        num_peers = len(active_runs)
        if not num_peers:
            # all runs are deactivated due to low scores compared to peers
            return self.plan_ahead()
        steps = []
        inactive_scores = [self.archive.get_score(name) for name, status in self.archive.status.items() if not status]
        best_inactive_score = max(inactive_scores, default=-math.inf) \
            if self.archive.ascending_metric else min(inactive_scores, default=math.inf)
        for run in active_runs:
            future_scores = self.archive.scores_at(self.archive.progress(run) +
                                                   self.archive.assess_interval_samples)
            safe = num_peers <= self.args.min_ref or math.ceil(num_peers * self.args.cull_keep_ratio) == num_peers
            initial = not future_scores
            promising = self.archive.get_score(run) > best_inactive_score == self.archive.ascending_metric
            if safe and (initial or promising):
                steps.append(None)  # run till the end
            else:
                steps.append(self.args.assess_interval)
        for run, step in zip(active_runs, steps):
            msg = f'| continue "{run}" '
            if steps is not None:
                msg += f'for {self.args.assess_interval * 100:.0f}% steps.'
            else:
                msg += 'till the end.'
            print(msg)
            self.task_queue.append(Continue(run, step))


class Order:
    ...


class NewRun(Order):
    def __init__(self, name, args):
        self.name = name
        self.args = args


class Continue(Order):
    def __init__(self, name, steps):
        self.name = name
        self.steps = steps


class Terminate(Order):
    ...


class Archive:
    def __init__(self, args):
        self.args = args
        self.configs = {}
        self.logs = {}
        self.status = {}
        self._progress = {}
        self._eval_deviation = None
        self._ascending_metric = None

    def update(self, commands):
        self.get_score.cache_clear()
        self.scores_at.cache_clear()
        Archive.best_run.fget.cache_clear()
        for name, _ in commands:
            save_dir = os.path.join(self.args.save_dir, name)
            if name not in self.configs:
                args = Arg(**Options.load_hjson(os.path.join(save_dir, 'args.hjson')))
                self.configs[name] = args
                if self._ascending_metric is None:
                    self._ascending_metric = args.ascending_metric
                else:
                    assert self._ascending_metric == args.ascending_metric
            args = self.configs[name]
            with io.open(os.path.join(save_dir, 'valid.log.tsv')) as f:
                log = pd.read_csv(f, sep='\t', header=0)
            log['read'] = (log['step'] * (args.batch_size * args.update_freq))
            self._progress[name] = log['read'].iloc[-1]
            self.logs[name] = log
            # tell whether a training is done
            with io.open(os.path.join(save_dir, 'log.txt')) as f:
                is_done = 'done training' in f.read()
                self.status[name] = not is_done

    @property
    def active_runs(self):
        if not self.status:
            return []
        return sorted(
            [name for name, status in self.status.items() if status],
            key=lambda name: self.get_score(name), reverse=self._ascending_metric
        )

    @property
    @lru_cache(1)
    def best_run(self):
        if not self.logs:
            return
        agg_func = max if self._ascending_metric else min
        return agg_func(self.logs.keys(), key=lambda name: self.get_score(name))

    @property
    def ascending_metric(self):
        if self._ascending_metric is None:
            raise RuntimeError('ascending_metric not set')
        return self._ascending_metric

    @property
    def assess_interval_samples(self):
        """approximate steps for assess_interval"""
        return self._eval_deviation * 2

    def deactivate(self, name):
        self.status[name] = False

    def progress(self, name):
        return self._progress[name]

    @lru_cache()
    def get_score(self, name):
        return round(self.logs[name].iloc[-1]['best_score'], 5)

    @lru_cache()
    def scores_at(self, progress):
        scores = []
        if self._eval_deviation is None:
            args = next(iter(self.configs.values()))
            total_samples = get_total_steps(args, runtime=False) * args.batch_size * args.update_freq
            self._eval_deviation = round(total_samples * self.args.assess_interval * 0.5)
            print(f'| eval deviation is set to {self._eval_deviation}')
        for log in self.logs.values():
            # find the closest record
            index = (log['read'] - progress).abs().argsort()[0]
            row = log.iloc[index]
            if abs(row['read'] - progress) < self._eval_deviation:
                scores.append(round(row['best_score'], 5))
        return scores


def plot_runs(archive, path):
    fig = plt.figure(figsize=(12.8, 9.6))
    ax = plt.gca()
    start_score = []
    for name, log in archive.logs.items():
        trial_id = name[6:]
        line = '--' if archive.status[name] else '-'
        plt.plot(log['read'], log['best_score'], line, alpha=0.4, label=name)
        start_score.append(log.iloc[0]['best_score'].item())
        best_x = log.iloc[-1]['read']
        best_y = log.iloc[-1]['best_score']
        plt.scatter([best_x], [best_y], s=[50], color='gray', alpha=0.3)
        ax.annotate(trial_id, xy=(best_x, best_y), color='darkgray')
        ax.annotate(trial_id, xy=(log.iloc[0]['read'], log.iloc[0]['best_score']), color='darkgray')
    plt.xlabel('#samples')
    # adjust ylim to avoid outliers skewing the plot
    med_start_score = pd.Series(start_score).median()
    if archive.ascending_metric:
        plt.ylim(bottom=med_start_score)
    else:
        plt.ylim(top=med_start_score)

    with io.open(path, 'wb') as f:
        plt.savefig(f, format='png')
    plt.close(fig)


def aggregate_results(path):
    return run_script('aggregate_results', [path, '--full'])


def spawned_main(entry, argv):
    try:
        entry(argv)
    except KeyboardInterrupt:
        pass  # SIGINT; Killed by parent, do nothing
    except Exception:
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
