import os
import sys
sys.path = sys.path[1:] + sys.path[:1]
import time
import random
import hjson
from functools import partial
from xdpx.options import Options
from xdpx.utils import run_script_cli, io, current_time
from xdpx.utils.env_utils import dependencies
from xdpx.preprocess import cli_main as prepro_cli
from xdpx.train import cli_main as train_cli
from xdpx.visualize import cli_main as viz_cli
from xdpx.evaluate import cli_main as eval_cli
from xdpx.predict import cli_main as pred_cli
from xdpx.autotune import cli_main as tune_cli


command_map = {
    'x-prepro': prepro_cli,
    'x-train': train_cli,
    'x-viz': viz_cli,
    'x-eval': eval_cli,
    'x-pred': pred_cli,
    'x-tune': tune_cli,
    'x-script': run_script_cli,
}


def main(argv=sys.argv, dry=False):
    """
    Cannot install cli commands on the command line with order version of pip (on PAI)
    so we have to parse manually
    """
    flags = list(set(name for name in argv if name.startswith('--')))
    argv = [name for name in argv if not name.startswith('--')]
    try:
        flags.remove('--dry')
        dry = True
    except ValueError:
        ...
    with_denpendencies = False
    if not dry:
        for name, hook in dependencies.items():
            try:
                flags.remove(f'--{name}')
                hook()
                with_denpendencies = True
            except ValueError:
                ...
    i = 1
    first_run = True
    while i < len(argv):
        prog, *config = argv[i: i + 2]
        i += 2
        if prog == 'x-script':
            while i < len(argv):
                if argv[i].startswith('x-'):
                    break
                config.append(argv[i])
                i += 1
        elif prog not in command_map:
            raise ValueError(f'Unknown command {prog}, should be one of ' + ' '.join(f'"{key}"' for key in command_map))
        print(prog, *config)
        cleanup = lambda: ...
        if not dry and 'RANK' in os.environ and int(os.environ['RANK']) > 0:
            # when running on multi-worker, workers other than master should only join in training
            if prog != 'x-train':
                continue
            assert len(config) == 1
            rank = int(os.environ['RANK'])
            # this should not be in the while loop otherwise the filename length will keep growing
            tmp_config = f'{os.path.splitext(config[0])[0]}.{rank}.hjson'

            retry = 0
            while True:
                try:
                    # skip the check for save_dir exists. This should be in the try-except block because
                    # usually "load_configs_from_file" will load `${data_dir}/meta` which may not have been created.
                    configs = Options.load_configs_from_file(config[0])
                    for cfg in configs:
                        if 'resume' not in cfg or not cfg['resume']:
                            tmp_dir = './tmp' + str(random.Random().randint(0, 1000000))
                            cfg['save_dir'] = tmp_dir

                    with io.open(tmp_config, 'w') as f:
                        hjson.dump(configs, f)
                    config = [tmp_config]
                    # check whether prepro (by the master node) is ready
                    command_map[prog]([prog, *config] + ['--dry'])
                    break
                except Exception as e:
                    sys.stderr.write(f'\n{current_time()} Cannot start training yet..Keep waiting\n' + str(e))
                    retry += 1
                    if retry >= 20:
                        break
                    time.sleep(30)

            def custom_cleanup(tmp_path):
                if io.exists(tmp_path):
                    io.remove(tmp_path)

            cleanup = partial(custom_cleanup, tmp_path=tmp_config)

        if dry or (first_run and not with_denpendencies):
            try:
                command_map[prog]([prog, *config] + (['--dry'] if dry else []))
            except KeyboardInterrupt:
                pass
            except Exception:
                if not dry:
                    safe_exit()
                raise
        else:
            # use a subprocess to launch sub-commands to ensure isolation
            # but leave the first run in the main process to enable local debugging
            import multiprocessing
            mp = multiprocessing.get_context('spawn')
            run_state = mp.Queue()
            process = mp.Process(
                target=spawned_main,
                args=(run_state, prog, *config),
                daemon=False,
            )
            process.start()
            process.join()
            if run_state.get():
                sys.stderr.write('\n')
                raise RuntimeError('Command failed: ' + ' '.join([prog, *config]))
        first_run = False
        cleanup()


def safe_exit():
    import traceback
    from xdpx.utils import delayed_flush
    # ensure the error is logged to file
    print('| ' + traceback.format_exc())
    delayed_flush(0)


def spawned_main(run_state, prog, *config):
    try:
        command_map[prog]([prog, *config])
        run_state.put(0)
    except KeyboardInterrupt:
        pass
    except Exception:
        safe_exit()
        run_state.put(1)


def io_entry():
    ret = getattr(io, sys.argv[1])(*sys.argv[2:])
    if ret is not None:
        print(ret)


def cli_entry(cmd_name):
    argv = sys.argv.copy()
    argv.insert(1, cmd_name)
    main(argv)


prepro_entry = partial(cli_entry, cmd_name='x-prepro')
train_entry = partial(cli_entry, cmd_name='x-train')
viz_entry = partial(cli_entry, cmd_name='x-viz')
pred_entry = partial(cli_entry, cmd_name='x-pred')
eval_entry = partial(cli_entry, cmd_name='x-eval')
tune_entry = partial(cli_entry, cmd_name='x-tune')
script_entry = partial(cli_entry, cmd_name='x-script')


if __name__ == "__main__":
    main()
