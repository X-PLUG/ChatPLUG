import os
import sys
import pandas as pd
from tqdm import tqdm
from xdpx.trainer import Trainer
from xdpx.utils import io


"""
Remove checkpoints except the last one and the best one. 
"""


def cli_main(argv=sys.argv):
    try:
        keyword = next(name for name in argv if name.startswith('-'))[1:]
    except StopIteration:
        keyword = None
    root_dir = argv[1]
    action = False
    freed_storage = 0
    try:
        for file in io.listdir(root_dir, recursive=True, full_path=True, contains=keyword):
            if file.endswith('valid.log.tsv'):
                root = os.path.dirname(file)
                files = [file for file in io.listdir(root, recursive=False, full_path=False, contains='.pt')]
                if len(files) <= 2:
                    continue
                with io.open(file) as f:
                    try:
                        df = pd.read_csv(f, sep='\t', header=0)
                    except pd.errors.EmptyDataError:
                        continue
                    best_step = int(df.iloc[df['best_score'].idxmax()]['step'])
                    last_step = int(df.iloc[-1]['step'])
                    best_path = Trainer.save_pattern(best_step)
                    last_path = Trainer.save_pattern(last_step)
                    progress = tqdm(files, desc=f'cleaning {root}')
                    for file in progress:
                        if file != best_path and file != last_path:
                            path = os.path.join(root, file)
                            freed_storage += io.size(path) / 1024 ** 3
                            io.remove(path)
                            progress.write(path)
                            action = True
    except KeyboardInterrupt:
        pass
    if not action:
        print('Nothing to be cleaned.')
    else:
        print(f'{freed_storage:.2f} GB storage has been freed.')


if __name__ == "__main__":
    cli_main()
