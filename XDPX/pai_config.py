import os


def loop_until_valid(msg, default, min_val, max_val):
    msg += f' ({default}): '
    while True:
        try:
            n = int(input(msg).strip() or default)
            if not min_val <= n <= max_val:
                raise ValueError(f'should be in range [{min_val}, {max_val}]')
            return n
        except ValueError as e:
            print(e)


def loop_until_valid_choice(msg, default, options):
    msg += f' ({default}): '
    while True:
        try:
            n = input(msg).strip().lower() or default
            if n not in options:
                raise ValueError(f'should be one of {options}')
            return n
        except ValueError as e:
            print(e)


config = {}
if os.path.exists('.pai_config'):
    with open('.pai_config') as f:
        for line in f:
            key, val = line.rstrip().split('=')
            if key not in config:
                config[key] = val

pt_version = int(config.get('pt_version', 0)) or loop_until_valid_choice('Which PyTorch version/pytorch版本',
                                                                         '180', ('100', '131', '151','180'))
n_workers = int(config.get('n_workers', 0)) or loop_until_valid('How many workers/使用几台机器', 1, 1, 8)
n_gpus = int(config.get('n_gpus', 0)) or loop_until_valid('How many GPUs per worker/每台机器几张卡', 1, 1, 8)
n_cpus = int(config.get('n_cpus', 0)) or loop_until_valid('How many CPUs per worker/每台机器CPU几核', 4, 1, 64)
memory = int(config.get('memory', 0)) or loop_until_valid('How much machine memory/每台机器几G内存', 10, 2, 128)
v100 = config.get('v100', 0) or loop_until_valid_choice('Force V100/是否指定V100', 'n', ('y', 'n'))

config = f"""
pt_version={pt_version}
n_workers={n_workers}
n_gpus={n_gpus}
n_cpus={n_cpus}
memory={memory*1024}
v100={v100}
"""

with open('.pai_config', 'w') as f:
    f.write(config)
