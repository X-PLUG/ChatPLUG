import os
import sys
from typing import Callable
from xdpx.options import Options, Arg


def bootstrap(options: Options, entry_func: Callable[[Arg], None], entry_file: str, argv=sys.argv):
    # according to https://github.com/pytorch/examples/issues/538#issuecomment-523681239
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    try:
        argv.remove('--dry')
        dry_run = True
    except ValueError:
        dry_run = False
    try:
        argv.remove('--verbose')
    except ValueError:
        import logging; logging.disable(logging.WARNING)
        import warnings
        from sklearn.exceptions import UndefinedMetricWarning
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

    if len(argv) == 1:
        print('Usage: <config_path>')
        exit()
    config_file = argv[1]
    arg_group = options.parse(config_file)

    if len(set(arg_group)) != len(arg_group):
        raise ValueError('Duplicated args found. Please check if values in a loop '
                         'are overwritten by higher-level configs.')

    for args in arg_group:
        print(options.tree(args))
        if dry_run:
            if getattr(args, 'resume', False):
                import re
                from xdpx.train import compare_args
                from xdpx.utils import io
                latest_args = 0
                for file in io.listdir(args.save_dir, contains='args.resume'):
                    step = int(re.search(r'(?<=args\.resume-)\d+', file).group())
                    if step > latest_args:
                        latest_args = step
                if latest_args:
                    latest_args = f'args.resume-{latest_args}.py'
                else:
                    latest_args = f'args.py'
                with io.open(os.path.join(args.save_dir, latest_args)) as f:
                    prev_args = Options.parse_tree(eval(f.read()))

                compare_args(prev_args,
                             args.change(batch_size=max(1, args.batch_size // args.distributed_world_size)))
            print()
            continue
        args.__cmd__ = os.path.basename(entry_file)[:-3]
        entry_func(args)
    if dry_run and len(arg_group) > 1:
        print(f'{len(arg_group)} runs detected.')
