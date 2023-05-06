import re
import os
import sys
import json
import torch
from xdpx.utils import move_to_cuda
from xdpx.utils.profiling import profile_module
from profiling import configs


def main(argv=sys.argv):
    argv = argv.copy()
    kwargs = []
    for arg in argv[1:]:
        if arg.startswith('-'):
            m = re.match(r'-([^=]+)=(.+)', arg)
            if not m:
                raise ValueError(f'Unknown keyword format: "{arg}". Should be like "-key=val"')
            key, val = m.group(1), m.group(2)
            val = eval(val)
            kwargs.append((key, val))
            argv.remove(arg)

    if len(argv) != 3:
        print('Usage: python scripts/run_profiling <profile_name> <trace_save_path> -key1=val1 -key2=val2 ...')
        exit()
    profile, save_path = argv[1:]
    profile = configs[profile]()
    pid = os.path.splitext(os.path.basename(save_path))[0]

    seed = 1
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    module = profile.build_module(**dict(kwargs))
    data = profile.create_fake_data()
    if torch.cuda.is_available():
        module = module.cuda()
        data = move_to_cuda(data)

    with torch.no_grad():
        with profile_module(module) as prof:
            # Warmup CUDA memory allocator and profiler
            if torch.cuda.is_available():
                for _ in range(10):
                    module(**data)
            prof.clear()
            module(**data)
    
    chrome_events = prof.summarize(pid=pid)
    with open(save_path, 'w') as f:
        json.dump(chrome_events, f)


if __name__ == "__main__":
    main()

