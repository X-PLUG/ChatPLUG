import sys
import csv
import hjson
import math
import torch
import pandas as pd
from collections import defaultdict
from xdpx.options import Arg, Options, Argument
from xdpx.bootstrap import bootstrap
from xdpx.utils import io, pformat_dataframe
from sklearn.metrics import precision_recall_fscore_support

"""
Compute thresholds for BCE prediction
"""


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('predict_file', required=True, validate=lambda val: io.exists(val)),
        Argument('target_map_file', required=True, validate=lambda val: io.exists(val)),
        Argument('eval_file', doc='default is predict_file', validate=lambda val: not val or io.exists(val)),
        Argument('out_file', type=str),
        Argument('mode', default='f1', validate=lambda val: val in ['precision', 'f1']),
        Argument('min_threshold', default=0.4),
        Argument('max_threshold', default=1.0),
        Argument('beta', default=1.0, doc='beta value in F-beta, default F1'),
        Argument('resolution', default=4, doc='decimal points for threshold results'),
        Argument('gap', default=0.5, validate=lambda val: 0 <= val < 1,
                 doc='smaller value means the threshold is closer to the positive'),
    )
    options.add_global_constraint(lambda args: args.min_threshold < args.max_threshold)
    bootstrap(options, main, __file__, argv)


def main(cli_args: Arg):
    with io.open(cli_args.target_map_file) as f:
        targets = [line.strip().lower() for line in f]
        target_map = {target: i for i, target in enumerate(targets)}
    num_classes = len(target_map)

    with io.open(cli_args.predict_file) as f:
        data = pd.read_csv(f, sep='\t', header=0, quoting=csv.QUOTE_NONE)

    records = defaultdict(lambda: [])
    n_pos = defaultdict(lambda: 0)
    probs = []
    for _, sample in data.iterrows():
        prob = eval(sample['prob'])
        probs.append(prob)
        target = sample['target'].lower()
        t_index = target_map[target]
        for i in range(1, num_classes):
            records[i].append([prob[i], t_index == i])
        n_pos[t_index] += 1

    beta = cli_args.beta
    gap = cli_args.gap
    thresholds = []
    for i in range(1, num_classes):
        record = records[i]
        if not any(x[1] for x in record):
            thresholds.append(cli_args.min_threshold + 10 ** - cli_args.resolution)
            continue
        thresh = None

        if cli_args.mode == 'precision':
            record.sort(key=lambda x: x[0], reverse=True)
            best_value = -math.inf
            value = 0
            for prob, tag in record:
                value += 1 if tag else -1
                if value > best_value:
                    best_value = value
                    thresh = prob
        elif cli_args.mode == 'f1':
            tp = 0
            fp = 0
            fn = n_pos[i]
            best_value = -1
            record.sort(key=lambda x: x[0], reverse=True)
            for j in range(len(record)):
                prob, tag = record[j]
                if tag:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
                f1 = ((1 + beta ** 2) * tp) / ((1 + beta ** 2) * tp + fp + beta ** 2 * fn)
                if f1 > best_value:
                    best_value = f1
                    if j + 1 < len(record):
                        thresh = gap * record[j+1][0] + (1 - gap) * prob
                    else:
                        thresh = prob
        else:
            raise NotImplementedError

        assert thresh is not None
        thresholds.append(thresh)
    assert len(thresholds) == num_classes - 1, len(thresholds)
    print()
    print('original thresholds:', thresholds)
    r = cli_args.resolution
    thresholds = [
        min(
            max(
                math.floor(t * 10 ** r) / 10 ** r,
                cli_args.min_threshold
            ),
            cli_args.max_threshold
        ) for t in thresholds
    ]
    print('clamped thresholds: ', thresholds)
    if cli_args.out_file:
        with io.open(cli_args.out_file, 'w') as f:
            hjson.dump(dict(predict_threshold=thresholds), f)

    if cli_args.eval_file and cli_args.eval_file != cli_args.predict_file:
        with io.open(cli_args.eval_file) as f:
            data = pd.read_csv(f, sep='\t', header=0, quoting=csv.QUOTE_NONE)
        probs = []
        for _, sample in data.iterrows():
            prob = eval(sample['prob'])
            probs.append(prob)
    thresholds = torch.tensor([0.5] + thresholds)
    stats = []
    for name, threshold in zip(
        ['origin', 'tuned'],
        [torch.ones(num_classes) * 0.5, thresholds],
    ):
        predictions = []
        inverse_target_map = {v: k for k, v in target_map.items()}
        for prob in probs:
            prob = torch.tensor(prob)
            prob = prob.masked_fill(prob < threshold, -1.0)
            prob[0] = 0.0
            index = prob.max(0)[1].item()
            predictions.append(inverse_target_map[index])
        p, r, f1, _ = precision_recall_fscore_support(data.target, predictions, average='micro', labels=targets[1:])
        stats.append([name, p, r, f1])
    title = ' '
    stats = pd.DataFrame(stats, columns=[title, 'precision', 'recall', f'f1'])
    print(pformat_dataframe(stats))


if __name__ == "__main__":
    cli_main()
