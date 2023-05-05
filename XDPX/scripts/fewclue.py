import sys
import csv
import pandas as pd
from xdpx.utils import io
import json
from xdpx.bootstrap import bootstrap
from xdpx.options import Options, Argument, Arg
import os
import numpy as np
from collections import Counter
import random


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('predict_file', required=True, type=str),
        Argument('out_dir', required=True, type=str),
        Argument('domain', required=True, type=str),
        Argument('gold_file', required=True, type=str),
        Argument('splits', default=5),

    )
    bootstrap(options, main, __file__, argv)


def main(args: Arg):
    with io.open(args.predict_file) as f:
        data = pd.read_csv(f, sep='\t', header=0, quoting=csv.QUOTE_NONE)

    print(f'total data size: {len(data)}')

    selected_data = {}
    for _, sample in data.iterrows():
        text = sample['tokens']
        prob = float(sample['target_prob'])
        label = sample['pred']
        if label not in selected_data:
            selected_data[label] = []
        selected_data[label].append((text, prob))

    selected_data2 = {}
    for label, samples in selected_data.items():
        threshold = np.average([p for _, p in samples])
        print(f'{label} {threshold}')
        samples = [sample for sample in samples if sample[1] > threshold]
        selected_data2[label] = samples

    results = []
    for label, samples in selected_data2.items():
        for text, prob in samples:
            results.append({'mode': 'episode', 'domain': args.domain, 'text': text, 'label': label})

    print(f'augmented data size: {len(results)}')
    print(Counter([t['label'] for t in results]))

    gold_data = json.loads(io.open(args.gold_file).read())
    results.extend(gold_data)

    print(f'gold data size: {len(gold_data)}')
    print(f'all data size: {len(results)}')

    random.shuffle(results)
    split_size = len(results) // args.splits
    print(f'split size: {split_size}')

    for idx in range(args.splits):
        with io.open(args.out_dir + f'/split{idx}.json', 'w') as save_file:
            split = results[split_size * idx: split_size * (idx + 1)]
            save_file.write(json.dumps(split, indent=3, ensure_ascii=False))

    with io.open(args.out_dir + f'/train_noisy.json', 'w') as save_file:
        save_file.write(json.dumps(results, indent=3, ensure_ascii=False))


if __name__ == "__main__":
    cli_main(sys.argv)
