import os
import re
import sys
from typing import List, Union, Dict, Any
from pprint import pprint, pformat
from xdpx.utils import io, cache_file, load_args
from xdpx.bootstrap import bootstrap
from xdpx.options import Arg, Options, Argument


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('save_root', required=True),
        Argument('out_dir', required=True),
        Argument('tasks', type=List[str],
                 default='timi-qq gov-qq youku-qq hotline-qq timi-cls hotline-cls supermi-cls cainiao-cls'.split()),
        Argument('extra_config', type=Union[Dict[str, Any], str]),
    )
    bootstrap(options, main, __file__, argv)


def main(cli_args: Arg):
    from scripts.aggregate_results import get_best_result

    tasks = {}
    re_task = re.compile('|'.join(cli_args.tasks))
    for path in io.listdir(cli_args.save_root, recursive=True, full_path=True):
        if path.endswith('valid.log.tsv'):
            task = re_task.search(path)
            if not task:
                continue
            task = task.group()
            if task not in tasks:
                tasks[task] = []
            result = get_best_result(path)
            if not result:
                continue
            result = {name: score for name, score in zip(*result)}
            model_path = os.path.join(os.path.dirname(path), 'checkpoint-{}.pt'.format(result['step']))
            if not io.exists(model_path):
                continue
            tasks[task].append((model_path, result['best_score']))
    tasks = {name: max(scores, key=lambda x: x[1]) for name, scores in tasks.items() if scores}
    pprint(tasks, width=200)
    tasks = {name: scores[0] for name, scores in tasks.items()}

    script_dir = os.path.join(cli_args.out_dir, 'scripts')
    model_dir = os.path.join(cli_args.out_dir, 'models')
    
    io.makedirs(script_dir, exist_ok=True)
    script_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils')
    for script_file in ('pred_cls.py', 'pred_match.py'):
        io.copy(os.path.join(script_source, script_file), os.path.join(script_dir, script_file))

    for name, checkpoint in tasks.items():
        save_dir = os.path.dirname(checkpoint)
        export_dir = os.path.join(model_dir, name)
        os.makedirs(export_dir, exist_ok=True)
        train_args = load_args(save_dir)
        data_dir = train_args['data_dir']
        data_args = load_args(data_dir)
        args = Arg(**{**data_args, **train_args})
        io.copy(cache_file(checkpoint), os.path.join(export_dir, 'checkpoint.pt'))
        io.copy(os.path.join(save_dir, 'snapshot.zip'), os.path.join(export_dir, 'snapshot.zip'))
        for resource_file in io.listdir(data_dir, contains='.txt'):
            io.copy(os.path.join(data_dir, resource_file),
                    os.path.join(export_dir, resource_file))
        for config_file in io.listdir(save_dir, contains='.json'):
            io.copy(os.path.join(save_dir, config_file),
                    os.path.join(export_dir, config_file))
        # update args
        if cli_args.extra_config is not None:
            config_files = []
            if isinstance(cli_args.extra_config, dict):
                parent = cli_args.extra_config.get('__parent__', None)
                if parent is not None:
                    if isinstance(parent, str):
                        config_files.append(parent)
                    elif isinstance(parent, list):
                        config_files.extend(parent)
                    else:
                        raise ValueError
            else:
                config_files.append(cli_args.extra_config)
            for extra_config in config_files:
                if io.islocal(extra_config):
                    extra_config = os.path.join(save_dir, extra_config)
                args = args.change(**Options.load_hjson(extra_config))
            if isinstance(cli_args.extra_config, dict):
                args = args.change(**cli_args.extra_config)
        args.__cmd__ = 'serve'
        args.distributed_world_size = 1
        args.distributed_rank = 0
        args.device_id = 0
        args.distributed_init_method = None
        args.remove_duplicate = False
        args.skip_bad_lines = False
        args.strict_size = True
        args.checkpoint = 'checkpoint.pt'
        args.config_file = 'config.json'
        args.data_dir = '.'
        args.vocab_file = 'vocab.txt'
        args.target_map_file = 'target_map.txt'
        with io.open(os.path.join(export_dir, 'args.py'), 'w') as f:
            f.write(pformat(vars(args), width=200))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cli_main()
