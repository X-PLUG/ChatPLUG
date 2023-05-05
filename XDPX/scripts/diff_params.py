import os
import sys
import subprocess
from xdpx.options import Options
from xdpx.utils import io, default_cache_root, diff_params, load_args


def cli_main(argv=sys.argv):
    path1, path2 = argv[1:3]
    args_group = []
    for path in (path1, path2):
        if path.endswith('.hjson'):
            os.makedirs(default_cache_root, exist_ok=True)
            config_file = f'{default_cache_root}/args.py'
            with io.open(path) as f:
                content = f.read()
            if 'data_source' in content:
                command = 'x-prepro'
            else:
                command = 'x-train'
            p = subprocess.Popen(f'{command} {path} --dry | tail +2 > {config_file}', shell=True)
            _, status = os.waitpid(p.pid, 0)
            if status != 0:
                exit()
            with io.open(config_file) as f:
                args = vars(Options.parse_tree(eval(f.read())))
        else:
            args = load_args(path)
        args_group.append(args)
    args1, args2 = args_group
    diff = diff_params(args1, args2)
    if not diff:
        print('The 2 runs have no diffirent params.')
        return
    for item in diff:
        print(item)


if __name__ == "__main__":
    cli_main()
