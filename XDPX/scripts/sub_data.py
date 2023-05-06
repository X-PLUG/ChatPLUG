import sys
from collections import defaultdict
from xdpx.utils import io


"""
remove lines in data2 from data1
"""


def cli_main(argv=sys.argv):
    if len(argv) != 3:
        print('Usage: x-script sub_data $data_file1 $data_file2 > $data_diff')
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
    for key in group[0].keys() - group[1].keys():
        for _ in range(group[0][key]):
            print(key)


if __name__ == "__main__":
    cli_main()
