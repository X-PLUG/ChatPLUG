import re
import os
import sys
import pandas as pd
from io import StringIO
from collections import defaultdict
from xdpx.utils import io, pformat_dataframe


def cli_main(argv=sys.argv):
    if len(argv) != 3:
        print('Usage: x-script diff_meta $data_dir1 $data_dir2 > stats.txt')
        exit()
    path1, path2 = argv[1:3]
    meta_group = []
    for path in (path1, path2):
        group = {
            'td': defaultdict(lambda: {})
        }
        with io.open(os.path.join(path, 'log.txt')) as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                if 'target distribution' in line:
                    name = re.search(r'target distribution of (.+):', line).group(1)
                    i += 1
                    line = lines[i][2:-1].replace(' | ', '\n')
                    line = re.sub(r' \((\d+)\)', r'\t\1', line)
                    td = pd.read_csv(StringIO(line), sep='\t', header=None, dtype=str)
                    td.columns = ['target_name', 'count']
                    group['td'][name] = td
                i += 1
        meta_group.append(group)
    for name in meta_group[0]['td'].keys():
        print(f'target distribution difference of {name}')
        td = meta_group[0]['td'][name].merge(meta_group[1]['td'][name], on='target_name', how='outer', suffixes=('_l', '_r'))
        td = td[td.count_l != td.count_r]
        if td:
            print(pformat_dataframe(td))
        else:
            print('\tNo difference.')


if __name__ == "__main__":
    cli_main()
