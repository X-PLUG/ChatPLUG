import os
import sys
import json
from xdpx.utils import io


def main(argv=sys.argv):
    if len(argv) != 2:
        print('Usage: <profile_dir>')
        exit()
    root = argv[1].rstrip('/')
    name = os.path.basename(root)
    combine = []
    for path in io.listdir(root, full_path=True):
        if not path.endswith('.json'):
            continue
        with io.open(path) as f:
            prof = json.load(f)
        combine.extend(prof)
    with io.open(os.path.join(os.path.dirname(root), name+'.json'), 'w') as f:
        json.dump(combine, f, indent=2)


if __name__ == '__main__':
    main()
