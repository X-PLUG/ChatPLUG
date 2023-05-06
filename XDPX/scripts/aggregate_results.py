import os
import sys
import pandas as pd
from io import StringIO
from datetime import datetime
from collections import OrderedDict
from xdpx.options import Arg
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
        print('Usage: <root_dir1> <root_dir2> ... -<keyword1> -<keyword2> (--full)')
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
            if file.endswith('valid.log.tsv'):
                current_root = os.path.dirname(file)
                if full_stats:
                    from xdpx.options import Options
                    with io.open(os.path.join(current_root, 'args.py')) as f:
                        configs.append(vars(Options.parse_tree(eval(f.read()))))
                mtime = io.last_modified(os.path.join(current_root, 'train.log.tsv'))
                active = (datetime.now() - mtime).total_seconds() < 60
                results = get_best_result(file, full=full_stats)
                if results is None:
                    continue
                targets, scores = results

                if not full_stats:
                    targets = [col[6:] if col.startswith('valid_') else col for col in targets]
                path = root_dir[prefix_len:].rstrip('/') + '/'
                path = path + '/'.join(file.split('/')[:-1])[len(root_dir):] + ('*' if active else '')
                stat = OrderedDict()
                stat['path'] = path
                for target, score in zip(targets, scores):
                    stat[target] = score
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
        ignored_keys = Arg.__exclude__
        ignored_keys.add('save_above_score')
        for ignore_key in ignored_keys:
            try:
                diff_keys.remove(ignore_key)
            except ValueError:
                ...

        header = header[:2] + diff_keys + header[2:]
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
    if full_stats:
        # sort results according to best_score
        assert all(config['ascending_metric'] == configs[0]['ascending_metric'] for config in configs)
        outputs[1:] = sorted(outputs[1:], key=lambda x: x[1], reverse=configs[0]['ascending_metric'])
    outputs = '\n'.join(['\t'.join(map(str, line)) for line in outputs])
    print(outputs)
    print()
    print(common_root)
    if full_stats:
        return pd.read_csv(StringIO(outputs), sep='\t', header=0)


def get_best_result(path, full=False):
    with io.open(path) as f:
        try:
            df = pd.read_csv(f, sep='\t', header=0)
        except pd.errors.EmptyDataError:
            return
        if full:
            headers = [col for col in df.columns if col not in ['ups', 'train_wall', 'clip', 'train_lr']]
        else:
            headers = [col for col in df.columns if col.startswith('valid_') or col == 'best_score' or col == 'step']
        df = df[headers]
        if df['best_score'].iloc[-1] >= df['best_score'].iloc[0]:
            best_index = df['best_score'].idxmax()
        else:
            best_index = df['best_score'].idxmin()
        scores = df.iloc[best_index].tolist()
        step_idx = headers.index('step')
        scores[step_idx] = int(scores[step_idx])
        # use "best_score" as the first column
        headers = headers[-1:] + headers[:-1]
        scores = scores[-1:] + scores[:-1]
        return headers, scores
    

if __name__ == "__main__":
    cli_main()
