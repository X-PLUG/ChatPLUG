import os
import sys
import time
import math
import torch
from typing import Union, List, Dict, Any, Optional
from tqdm import tqdm
from xdpx.options import Options, Argument, Arg
from xdpx.loaders import loaders
from xdpx.tasks import tasks
from xdpx.utils import io, move_to_cuda, parse_model_path, load_script, parse_relative_config
from xdpx.bootstrap import bootstrap
from xdpx.train import build_train_options
from xdpx.logger import SummaryLogger


def cli_main(argv=sys.argv):
    options = Options()
    Evaluator.register(options)
    options.register(
        Argument('valid_subset', required=True, type=Union[List[str], str],
                 doc='full path or name without file ext, like "dev", or a list of names',
                 post_process=lambda val: val if isinstance(val, list) else [val]),
        Argument('save_to_file', default=None, children=[
            Argument('save_mode', default='a', validate=lambda val: val in ('w', 'a')),
        ]),
        Argument('max_eval_steps', type=int, validate=lambda val: val is None or val > 0),
        Argument('compare_origin', default=False),
        Argument('ignore_novel_target', default=False),
    )
    options.add_global_constraint(lambda args: not args.compare_origin or args.save_dir is not None)
    bootstrap(options, main, __file__, argv)


class Evaluator:
    @staticmethod
    def register(options):
        options.register(
            Argument('save_dir', validate=lambda val: not val or io.exists(os.path.join(val, 'args.py')),
                     doc='evaluate from existed save_dir'),
            Argument('config', doc='evaluate from training config',
                     validate=lambda val: not val or (
                                 (val.endswith('.hjson') or val.endswith('.json')) and io.exists(val)),
                     post_process=parse_relative_config, type=str),
            Argument('extra_config', type=Union[str, Dict[str, Any]], default={},
                     doc='extra config that will overwrite the loaded config'),
            Argument('checkpoint', doc='Full path is needed if not provided',
                     post_process=lambda val, args: parse_model_path('<best>', args) if args.save_dir and val is None
                     else parse_model_path(val, args),
                     type=Optional[str],
                     validate=lambda val: (not val or io.exists(val) or io.exists(val + '.meta'),
                                           f'{val} does not exist. Check the path again and '
                                           f'don\'t forget to set "save: True" in training')),
            Argument('from_tf', default=False),
            Argument('batch_size', type=int),
            Argument('cuda', default=torch.cuda.is_available(),
                     validate=lambda value: not value or torch.cuda.is_available()),
            Argument('seed', default=1),
            # arguments below only makes sense when input format is raw text
            Argument('workers', type=int, doc='num of workers for raw text data loading.'),
            Argument('skip_bad_lines', default=False),
        )
        options.add_global_constraint(
            lambda args: (args.save_dir or args.config, 'either "save_dir" or "config" must be set.')
        )
        options.add_global_constraint(
            lambda args: (not (args.save_dir and args.config), 'only one of "save_dir" and "config" can be set.')
        )

    @staticmethod
    def build_evaluation(cli_args):
        if cli_args.save_dir:
            with io.open(os.path.join(cli_args.save_dir, 'args.py')) as f:
                args = Options.parse_tree(eval(f.read()))
            if isinstance(cli_args.extra_config, dict):
                args.update(Arg(**cli_args.extra_config))
            else:
                args.update(Arg(**Options.load_hjson(cli_args.extra_config)))
            args.__cmd__ = cli_args.__cmd__
            args.distributed_world_size = 1
            args.distributed_rank = 0
            args.device_id = 0
            args.distributed_init_method = None
            args.strict_size = True
            args.batch_size //= args.distributed_world_size
            args.orig_batch_size = args.batch_size
            args.cuda = cli_args.cuda
            args.fp16 = False
            if cli_args.batch_size:
                args.batch_size = cli_args.batch_size
            else:
                cli_args.batch_size = args.batch_size
        else:
            options = build_train_options()
            configs = Options.load_configs_from_file(cli_args.config)
            assert len(configs) == 1
            config = configs[0]
            if isinstance(cli_args.extra_config, dict):
                config.update(cli_args.extra_config)
            else:
                config.update(Options.load_hjson(cli_args.extra_config))
            config.update(cli_args.extra_config)
            config['save_dir'] = f'.tmp{time.time()}'
            if cli_args.batch_size:
                config['batch_size'] = cli_args.batch_size
            else:
                cli_args.batch_size = config['batch_size']
            args = options.parse_dict(config)
            args.distributed_world_size = 1
            args.distributed_rank = 0
            args.device_id = 0
            args.distributed_init_method = None
            args.fp16 = False
            args.__cmd__ = 'train'
        # load preprocess args
        try:
            with io.open(os.path.join(args.data_dir, 'args.py')) as f:
                data_args = Options.parse_tree(eval(f.read()))
            # update args
            args = Arg().update(data_args, args)
        except FileNotFoundError:
            print('Warning: preprocess args not found.')
        if cli_args.checkpoint:
            args.pretrained_model = None  # prevent loading the original pretrained model
        args.strict_size = True
        args.cuda = cli_args.cuda
        args.update_freq = 1
        # specify loader args
        if cli_args.workers is not None:
            args.workers = cli_args.workers
        args.skip_bad_lines = cli_args.skip_bad_lines
        args.remove_duplicate = False

        if cli_args.seed:
            torch.manual_seed(args.seed)
            if args.cuda:
                torch.cuda.manual_seed(args.seed)
        # build the loader
        if hasattr(args, 'loader'):
            loader = loaders[args.loader](args)
        else:
            loader = None

        # build the task
        task = tasks[args.task](args)
        model = task.build_model(args)
        loss = task.build_loss(args)

        # load checkpoint if provided
        if cli_args.checkpoint:
            model.load(cli_args.checkpoint, from_tf=cli_args.from_tf)
        else:
            print('No checkpoint is provided. Using the initial value of the model.')

        if cli_args.cuda:
            model = model.cuda()
            loss = loss.cuda()
        model.eval()
        loss.eval()
        return args, task, model, loss, loader


def load_input_data(path, loader, processor, with_target=True):
    """try to parse data in different formats, namely raw text files, binary data files and ODPS tables."""
    origin_data = None
    try:
        if path.startswith('odps://'):
            import common_io
            with common_io.table.TableReader(path) as f:
                data = []
                n = f.get_row_count()
                while n > 0:
                    read_batch = min(n, 1024)
                    data.extend(f.read(read_batch))
                    n -= read_batch
                data = origin_data = \
                    [[segment.decode() if isinstance(segment, bytes) else segment for segment in segments]
                     for segments in data if segments]
                data = [loader.parse(segments) for segments in data]
        else:
            f = loader.parser.open_file(path)[0]
            origin_data = [loader.parser.parse_line(line) for line in f]
            loader.parser.close_file(f)
            data = loader.load_data(path, ordered=True)
        data = processor.numerize_samples(data, with_target=with_target)
    except UnicodeDecodeError:
        import pickle
        try:
            with io.open(path, 'rb') as f:
                data = torch.load(f)
        except pickle.UnpicklingError:
            raise ValueError(f'Unable to parse file {path}')
    return data, origin_data


def main(cli_args):
    args, task, model, loss, loader = Evaluator.build_evaluation(cli_args)
    if cli_args.ignore_novel_target:
        task.processor.target_map._unk_index = 0
    logger = SummaryLogger()

    if cli_args.compare_origin:
        if args.valid_subset not in cli_args.valid_subset:
            raise ValueError(f'Error when compare_origin: original valid subset {args.valid_subset} '
                             f'not in x-eval valid_subset (subset aliases must be used when "compare_origin" is true)')
        if args.max_eval_steps is not None:
            if cli_args.max_eval_steps is not None and \
                    args.max_eval_steps * args.orig_batch_size != cli_args.max_eval_steps * args.batch_size:
                raise ValueError(f'Error when compare_origin: original max eval samples '
                                 f'{args.max_eval_steps * args.orig_batch_size} '
                                 f'is not equal to x-eval max eval samples '
                                 f'{cli_args.max_eval_steps * args.batch_size}')
            max_eval_steps = args.max_eval_steps * args.orig_batch_size / args.batch_size
            if max_eval_steps != int(max_eval_steps):
                raise ValueError(f'Error when compare_origin: cannot auto set max_eval_steps '
                                 f'given batch_size {cli_args.batch_size}')
            cli_args.max_eval_steps = int(max_eval_steps)
    valid_files = [os.path.join(args.data_dir, f'{file}.pt') if file in args.data_size else file
                   for file in cli_args.valid_subset]

    for valid_file, valid_subset in zip(valid_files, cli_args.valid_subset):
        # load data
        data, _ = load_input_data(valid_file, loader, task.processor)
        data = task.build_dataset(data, is_train=False)

        logging_outputs, sample_size = [], 0
        total_cnt = min(len(data), cli_args.max_eval_steps or math.inf)
        try:
            with tqdm(data, desc=f'evaluate {valid_subset}', total=total_cnt) as progress:
                for i, sample in enumerate(progress):
                    if cli_args.max_eval_steps is not None and i >= cli_args.max_eval_steps:
                        break
                    if cli_args.cuda:
                        sample = move_to_cuda(sample)

                    _loss, sample_size_i, logging_output = task.valid_step(
                        sample, model, loss
                    )
                    logging_outputs.append(logging_output)
                    sample_size += sample_size_i
        except KeyboardInterrupt:
            pass
        logging_output = task.aggregate_logging_outputs(logging_outputs, int(sample_size), loss, 
                                                        max_count=args.data_size.get(valid_subset, None))

        keys = sorted(logging_output.keys())
        vals = [str(logging_output[key]) for key in keys]
        try:
            index = keys.index('sample_size')
            del keys[index]
            del vals[index]
        except ValueError:
            pass
        keys = ['split'] + keys
        vals = [valid_subset] + vals
        if cli_args.checkpoint:
            keys = ['checkpoint'] + keys
            vals = [cli_args.checkpoint] + vals
        if cli_args.config:
            keys = ['config'] + keys
            vals = [cli_args.config] + vals
        sum_keys, sum_vals = logger.summarize()
        keys += sum_keys
        vals += sum_vals
        print('\t'.join(keys))
        print('\t'.join(vals))
        if cli_args.save_to_file:
            header = cli_args.save_mode == 'w' or not io.exists(cli_args.save_to_file)
            with io.open(cli_args.save_to_file, cli_args.save_mode) as f:
                if header:
                    f.write('\t'.join(keys) + '\n')
                f.write('\t'.join(vals) + '\n')
        if cli_args.compare_origin and valid_subset == args.valid_subset:
            valid_log = os.path.join(cli_args.save_dir, 'valid.log.tsv')
            headers, scores = load_script('aggregate_results').get_best_result(valid_log)
            deviated_results = []
            for name, orig_score in zip(headers, scores):
                if name.startswith('valid_'):
                    name = name[6:]
                    try:
                        index = keys.index(name)
                        score = float(vals[index])
                        if abs(score - orig_score) > 1e-6:
                            deviated_results.append((name, score, orig_score))
                    except ValueError:
                        ...
            if deviated_results:
                errmsg = '\n'.join(f'{name}: {score:.6} (originally: {orig_score:.6})'
                                   for name, score, orig_score in deviated_results)
                raise RuntimeError('Evaluated results are different from the original reported ones:\n' + errmsg)


if __name__ == "__main__":
    cli_main()
