import os
import sys
import pandas as pd
from io import StringIO
from collections import OrderedDict
from xdpx.options import Options
from xdpx.utils import io


def cli_main(argv=sys.argv):
    keywords = []
    root_dirs = []
    full_stats = False
    try:
        argv.remove('--full')
        full_stats = True
    except ValueError:
        ...
    for name in argv[1:]:
        if name.startswith('-'):
            keywords.append(name[1:])
        else:
            root_dirs.append(name)
    if not root_dirs:
        print('Usage: <root_dir1> <root_dir2> ... -<keyword1> -<keyword2>')
        exit()
    common_root = os.path.commonpath(root_dirs)
    if 'oss' in common_root:
        common_root = common_root[:4] + '/' + common_root[4:]
    prefix_len = len(common_root) + 1  # tailing slash
    stats = []
    configs = []
    for root_dir in root_dirs:
        root_dir = root_dir.rstrip('/') + '/'
        for file in io.listdir(root_dir, recursive=True, full_path=True, contains=keywords):
            if file.endswith('meta.hjson'):
                current_root = os.path.dirname(file)
                if full_stats:
                    with io.open(os.path.join(current_root, 'args.py')) as f:
                        configs.append(vars(Options.parse_tree(eval(f.read()))))

                meta = Options.load_hjson(os.path.join(current_root, 'meta'))
                meta.update(meta.pop('data_size'))
                del meta['__version__']

                path = root_dir[prefix_len:].rstrip('/') + '/'
                path = path + '/'.join(file.split('/')[:-1])[len(root_dir):]
                stat = OrderedDict()
                stat['path'] = path
                for key, val in meta.items():
                    stat[key] = val
                stats.append(stat)
    if not stats:
        return
    outputs = []
    if full_stats and len(configs) > 1:
        header = list(dict.fromkeys([key for stat in stats for key in stat.keys()]))
        values = [[stat.get(key, '') for key in header] for stat in stats]

        union_keys = set().union(*(config.keys() for config in configs))
        diff_keys = [key for key in union_keys
                     if any(key not in config or configs[0][key] != config[key] for config in configs)]

        header = header[:1] + diff_keys + header[1:]
        for i in range(len(values)):
            config_values = [configs[i].get(key, '') for key in diff_keys]
            values[i] = values[i][:2] + config_values + values[i][2:]
        outputs.append(header)
        outputs.extend(values)
    else:
        headers = ['\t'.join(stat.keys()) for stat in stats]
        for header in set(headers):
            header_added = False
            for i, match_header in enumerate(headers):
                if match_header != header:
                    continue
                if not header_added:
                    outputs.append(stats[i].keys())
                    header_added = True
                outputs.append(stats[i].values())
    outputs = '\n'.join(['\t'.join(map(str, line)) for line in outputs])
    print(outputs)
    print()
    print(common_root)
    if full_stats:
        return pd.read_csv(StringIO(outputs), sep='\t', header=0)


if __name__ == "__main__":
    cli_main()
