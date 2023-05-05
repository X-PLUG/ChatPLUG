import sys
from collections import defaultdict
from xdpx.utils import io


"""
Show the difference between data1 and data2
"""


def cli_main(argv=sys.argv):
    if len(argv) != 3:
        print('Usage: x-script diff_data $data_file1 $data_file2 > $diff_file.txt')
        exit()
    path1, path2 = argv[1:3]
    group = []
    for path in (path1, path2):
        stats = defaultdict(lambda: 0)
        with io.open(path) as f:
            for line in f:
                segments = line.split('\t')
                line = '\t'.join(seg.strip() for seg in segments)
                stats[line] += 1
        group.append(stats)
    if not group[0].keys() & group[1].keys():
        print('two data files are completely different')
        exit()
    stats = group[1]
    results = []
    for key, val in group[0].items():
        diff = stats[key] - val
        if diff < 0:
            results.append((f'[{diff}]', key))
        elif diff > 0:
            results.append((f'[~{diff + 1}]', key))
    for key in stats.keys() - group[0].keys():
        results.append((f'[+{stats[key]}]', key))
    results.sort(key=lambda x: x[1])
    for diff, line in results:
        print(diff, line)


if __name__ == "__main__":
    cli_main()
