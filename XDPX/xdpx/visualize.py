import re
import os
import sys
import math
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # prevent using the default interactive Xwindows backend on the server
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from packaging import version
from collections import defaultdict
from typing import List, Union, Dict
from xdpx.options import Options, Argument
from xdpx.utils import io
from xdpx.bootstrap import bootstrap
plt.style.use('ggplot')  # see `plt.style.available`


def cli_main(argv=sys.argv):
    options = Options()

    options.register(
        Argument('save_dir', required=True),
        Argument('label', doc='label for the original save_dir'),
        ref_dir_arg('ref_dir'),
        figext_arg('figext'),
        Argument('walltime', default=False, doc='use (walltime x world_size) as the x-axis in comparing'),
    )
    bootstrap(options, main, __file__, argv)


def ref_dir_arg(name):
    return Argument(name, type=Union[str, List[str], Dict[str, str]], post_process=parse_ref_dir,
        doc='a ref path, a list of ref paths, or a dict of <label: path>',
        validate=lambda value: value is None or (len(value) < 5 and all(io.exists(path) for path in value.values())))


def figext_arg(name):
    return Argument(name, default='png', validate=validate_ext)


def parse_ref_dir(value):
    if isinstance(value, str):
        value = [value]
    if isinstance(value, list):
        value = {
            os.path.basename(val.rstrip('/')): val for val in value
        }
    return value


def find_group_names(names):
    groups = defaultdict(lambda: [])
    for name in names:
        m = re.search(r'\d+$', name)
        if m:
            groups[name[:m.start()]].append(name)
    groups = {key: value for key, value in groups.items() if len(value) > 1}
    for key, value in groups.items():
        value.sort()
    return groups


def validate_ext(value):
    return value.lower() in plt.gcf().canvas.get_supported_filetypes()


def plot_grad_flow(named_parameters, path):
    """
    Reference: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    """
    plot_dir = os.path.dirname(path)
    io.makedirs(plot_dir, exist_ok=True)

    plt.clf()
    named_parameters = list(named_parameters)
    max_len = max(len(name) for name, _ in named_parameters)
    fig = plt.figure(figsize=(max(round(6.4 / 35 * len(named_parameters), 1), 6.4), 4.8 + round(max_len // 20, 1)))
    plt.gca().tick_params(axis='x', which='major', labelsize=7)
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            if p.grad is not None and p.grad.numel():
                if not p.grad.is_sparse:
                    abs_grad = p.grad.abs()
                    if 'keys' in n or 'values' in n:
                        if version.parse(torch.__version__) > version.parse('1.2'):
                            abs_grad = abs_grad[torch.where(abs_grad)]
                        else:
                            abs_grad = abs_grad[abs_grad.nonzero().split(1, 1)]
                        if not abs_grad.numel():
                            abs_grad = torch.zeros(1)
                else:
                    abs_grad = p.grad.coalesce().values().abs()
                ave_grads.append(abs_grad.mean().item())
                max_grads.append(abs_grad.max().item())
            else:
                ave_grads.append(0)
                max_grads.append(0)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    for i, g in enumerate(max_grads):
        if g < 1e-6:
            plt.hlines(1e-6, i - 0.5, i + 0.5, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.yscale('log')
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    fig.tight_layout()
    with io.open(path, 'wb') as f:
        plt.savefig(f)
    plt.close(fig)


ref_colors = [
    dict(alpha=0.5, color='darkgray'),
    dict(alpha=0.3, color='darkred'),
    dict(alpha=0.3, color='darkgreen'),
    dict(alpha=0.3, color='purple'),
]


def main(args):
    if args.save_dir.endswith('/*'):
        recursive = args.save_dir.endswith('**/*')
        save_dirs = [os.path.dirname(path) for path in io.listdir(args.save_dir[:-2], recursive=recursive, full_path=True) if path.endswith('valid.log.tsv')]
    else:
        save_dirs = [args.save_dir]
    for save_dir in save_dirs:
        if len(save_dirs) > 1:
            print('>>', save_dir)
        out_dir = os.path.join(save_dir, 'plots')
        io.makedirs(out_dir, exist_ok=True)
        train_log = os.path.join(save_dir, 'train.log.tsv')
        valid_log = os.path.join(save_dir, 'valid.log.tsv')

        def load_tsv(path):
            with io.open(path) as f:
                data = pd.read_csv(f, sep='\t', header=0)
                data.replace([np.inf, -np.inf], np.nan, inplace=True)
                data.fillna(0., inplace=True)
                return data

        def savefig(path):
            with io.open(path + '.' + args.figext, 'wb') as f:
                plt.savefig(f, format=args.figext)

        train = load_tsv(train_log)
        valid = load_tsv(valid_log)

        with io.open(os.path.join(save_dir, 'args.py')) as f:
            tree = eval(f.read())
            train_args = Options.parse_tree(tree)

        if args.ref_dir:
            train_ref = []
            valid_ref = []
            ref_names = []
            for ref_name, ref_dir in args.ref_dir.items():
                with io.open(os.path.join(ref_dir, 'args.py')) as f:
                    tree = eval(f.read())
                    ref_args = Options.parse_tree(tree)
                train_log2 = os.path.join(ref_dir, 'train.log.tsv')
                valid_log2 = os.path.join(ref_dir, 'valid.log.tsv')
                train2 = load_tsv(train_log2)
                valid2 = load_tsv(valid_log2)
                # synchronize steps to samples processed for a fair comparison
                train2['step'] *= (ref_args.batch_size * ref_args.update_freq) / (train_args.batch_size * train_args.update_freq)
                valid2['step'] *= (ref_args.batch_size * ref_args.update_freq) / (train_args.batch_size * train_args.update_freq)
                train_ref.append(train2)
                valid_ref.append(valid2)
                ref_names.append(ref_name)
        
        def add_footnote(text):
            plt.annotate(text, (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top', 
                color='lightgray', fontsize='small', annotation_clip=False)

        plt.clf()
        plt.plot(train['step'], train['gnorm'])
        plt.plot(train['step'], [train_args.clip_norm] * len(train), color='gray')
        plt.title('gradient norm')
        min_gnorm = min(train['gnorm'])
        if min_gnorm < train_args.clip_norm:
            plt.ylim([min_gnorm * 0.9, max(train['gnorm']) * 1.1])
        savefig(os.path.join(out_dir, 'gnorm'))

        plt.clf()
        if args.ref_dir:
            for train2, color, ref_name in zip(train_ref, ref_colors, ref_names):
                plt.plot(train2['step'], train2['lr'], **color, label=ref_name)
        plt.plot(train['step'], train['lr'], label='this')
        plt.title('learning rate')
        # original curves are drawed at last but legends should be on top
        handles, labels = plt.gca().get_legend_handles_labels()
        order = list(range(len(labels)))
        order = order[-1:] + order[:-1]
        if order:
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        add_footnote(save_dir)
        savefig(os.path.join(out_dir, 'lr'))

        if valid['clip'].max() > 0:
            plt.clf()
            plt.plot(valid['step'], valid['clip'])
            plt.title('gradient clipping rate')
            savefig(os.path.join(out_dir, 'clip'))

        train_sample_int = max(int(train_args.eval_interval / train_args.log_interval / 10), 1)

        valid_names = {name[6:] for name in valid.columns if name.startswith('valid_')}
        groups = find_group_names(valid_names)
        for prefix, group in groups.items():
            fig = plt.figure(figsize=(6.4 * max(1, int(math.ceil(len(valid) / 250))), 4.8 + max((len(group)-4), 0) * 0.4))
            for name in group:
                col = 'valid_' + name
                valid_names.discard(name)
                if name in train.columns and not args.walltime:
                    plt.plot(train['step'][::train_sample_int], train[name][::train_sample_int],
                             '--', alpha=0.5, label='train' + name[len(prefix):])
                else:
                    train_name = f'train_{name}'
                    if train_name in valid.columns:
                        plt.plot(valid['step'], valid[train_name], '--', alpha=0.5,
                                 label='train' + name[len(prefix):])

                plt.plot(valid['step'], valid[col], '-', alpha=0.5,
                         label='valid' + name[len(prefix):])

            if valid['best_score'].iloc[-1] >= valid['best_score'].iloc[0] and 'loss' not in group[0]:
                ymin = min(valid['valid_' + name].loc[int(len(valid)*0.1):].min() for name in group) * 0.9
                plt.ylim(bottom=ymin)
            else:
                ymin = min(valid['valid_' + name].min() for name in group) * 0.9
                ymax = max(max(valid['valid_' + name].loc[int(len(valid)*0.1):].max() for name in group) * 1.1,
                           ymin + 1e-6)
                if group[0] in train.columns and not args.walltime:
                    ymin = min(ymin, min(train[name].min() for name in group))
                plt.ylim(top=ymax, bottom=ymin)

            group_name = prefix.rstrip('_')
            plt.title(group_name)
            plt.legend()
            for step in train[~train.duplicated('epoch', keep='first')]['step'][1:]:
                plt.axvline(x=step, lw=0.5, color='red', alpha=0.4)
            add_footnote(save_dir)
            savefig(os.path.join(out_dir, group_name))
            plt.close(fig)
        for name in valid_names:
            col = 'valid_' + name
            fig = plt.figure(figsize=(6.4 * max(1, int(math.ceil(len(valid) / 250))), 4.8))
            x_name = 'train_wall' if args.walltime else 'step'
            if name in train.columns and not args.walltime:
                if args.ref_dir:
                    for train2, valid2, color, ref_name in zip(train_ref, valid_ref, ref_colors, ref_names):
                        if name in train2:
                            plt.plot(train2['step'][::train_sample_int], train2[name][::train_sample_int],
                                     '--', **color, label='train' + f'({ref_name})')
                        if col in valid2:
                            plt.plot(valid2['step'], valid2[col], '-', **color, label='valid' + f'({ref_name})')
                plt.plot(train['step'][::train_sample_int], train[name][::train_sample_int],
                         '--', alpha=0.5, color='navy', label='train' + (f'({args.label})' if args.label else ''))
            else:
                train_name = f'train_{name}'
                if train_name in valid.columns or col in valid.columns:
                    if args.ref_dir:
                        for valid2, color, ref_name in zip(valid_ref, ref_colors, ref_names):
                            if train_name in valid2:
                                plt.plot(valid2[x_name], valid2[train_name], '--', **color, label='train' + f'({ref_name})')
                            if col in valid2:
                                plt.plot(valid2[x_name], valid2[col], '-', **color, label='valid' + f'({ref_name})')
                    if train_name in valid.columns:
                        plt.plot(valid[x_name], valid[train_name], '--', alpha=0.5, color='navy', label='train' + (f'({args.label})' if args.label else ''))
                    else:
                        plt.plot(valid[x_name], valid[col], '--', alpha=0.5, color='navy',
                                 label='train' + (f'({args.label})' if args.label else ''))

            plt.plot(valid[x_name], valid[col], '-', alpha=0.5, color='navy', label='valid' + (f'({args.label})' if args.label else ''))
            best_idx = valid[col].idxmax() if 'loss' not in col else valid[col].idxmin()
            plt.scatter([valid.iloc[best_idx][x_name]], [valid.iloc[best_idx][col]], s=[50], color='green', alpha=0.3)

            plt.title(name + ('/walltime' if args.walltime else ''))
            # original curves are drawed at last but legends should be on top
            handles, labels = plt.gca().get_legend_handles_labels()
            order = list(range(len(labels)))
            order = order[-2:] + order[:-2]
            if order:
                plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
            if len(valid) > 0:
                if valid['best_score'].iloc[-1] >= valid['best_score'].iloc[0] and 'loss' not in col:
                    ymin = valid[col].loc[int(len(valid)*0.1):].min() * 0.9
                    plt.ylim(bottom=ymin)
                else:
                    ymin = valid[col].min() * 0.9
                    ymax = max(valid[col].loc[int(len(valid)*0.1):].max() * 1.1, ymin + 1e-6)
                    if name in train.columns and not args.walltime:
                        ymin = min(ymin, train[name].min())
                    plt.ylim(top=ymax, bottom=ymin)
            for step in train[~train.duplicated('epoch', keep='first')]['step'][1:]:
                if args.walltime:
                    row = valid[valid['step'] >= step].iloc[0]
                    step = row['train_wall'] / row['step'] * step
                plt.axvline(x=step, lw=0.5, color='red', alpha=0.4)
            add_footnote(save_dir)
            savefig(os.path.join(out_dir, name))
            plt.close(fig)
        # plot summaries
        valid_names = {name[6:] for name in valid.columns if name.startswith('valid_')}
        valid_names.update(valid.columns)
        valid_names.add('lr')
        valid_names -= {'wps', 'ups'}
        summary_names = set(train.columns) - valid_names
        groups = find_group_names(summary_names)
        for prefix, group in groups.items():
            fig = plt.figure(figsize=(6.4 * max(1, int(math.ceil(len(valid) / 250))), 4.8))
            for name in group:
                plt.plot(train['step'][::train_sample_int], train[name][::train_sample_int],
                         '-', alpha=0.5, label=name[len(prefix):])
                summary_names.discard(name)
            group_name = prefix.rstrip('_')
            plt.title(group_name)
            plt.legend()
            add_footnote(save_dir)
            savefig(os.path.join(out_dir, group_name))
            plt.close(fig)
        for name in summary_names:
            fig = plt.figure(figsize=(6.4 * max(1, int(math.ceil(len(valid) / 250))), 4.8))
            if args.ref_dir:
                for train2, color, ref_name in zip(train_ref, ref_colors, ref_names):
                    if name in train2:
                        plt.plot(train2['step'][::train_sample_int], train2[name][::train_sample_int],
                                 '-', **color, label=f'({ref_name})')
            plt.plot(train['step'][::train_sample_int], train[name][::train_sample_int],
                     '-', alpha=0.5, color='navy', label=(f'({args.label})' if args.label else ''))
            plt.title(name)
            # original curves are drawed at last but legends should be on top
            handles, labels = plt.gca().get_legend_handles_labels()
            order = list(range(len(labels)))
            order = order[-2:] + order[:-2]
            if order:
                plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
            add_footnote(save_dir)
            savefig(os.path.join(out_dir, name))
            plt.close(fig)
        
        # plot accumulative summaries
        summary_names = {
            name for name in valid.columns if not (
                name.startswith('train_') or name.startswith('valid_') or
                name in 'step epoch best_score memory gnorm clip wps ups'.split()
            )
        }
        groups = find_group_names(summary_names)
        for prefix, group in groups.items():
            fig = plt.figure(figsize=(6.4 * max(1, int(math.ceil(len(valid) / 250))), 4.8))
            for name in group:
                plt.plot(valid['step'], valid[name], alpha=0.5, label=name[len(prefix):])
                summary_names.discard(name)
            group_name = prefix.rstrip('_')
            plt.title(group_name)
            plt.legend()
            for step in train[~train.duplicated('epoch', keep='first')]['step'][1:]:
                plt.axvline(x=step, lw=0.5, color='red', alpha=0.4)
            add_footnote(save_dir)
            savefig(os.path.join(out_dir, group_name))
            plt.close(fig)
        for name in summary_names:
            fig = plt.figure(figsize=(6.4 * max(1, int(math.ceil(len(valid) / 250))), 4.8))
            if args.ref_dir:
                for valid2, color, ref_name in zip(valid_ref, ref_colors, ref_names):
                    if name in valid2:
                        plt.plot(valid2['step'], valid2[name], **color, label=f'({ref_name})')
            plt.plot(valid['step'], valid[name], alpha=0.5, color='navy', label=(f'({args.label})' if args.label else ''))
            plt.title(name)
            # original curves are drawed at last but legends should be on top
            handles, labels = plt.gca().get_legend_handles_labels()
            order = list(range(len(labels)))
            order = order[-2:] + order[:-2]
            if order:
                plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
            for step in train[~train.duplicated('epoch', keep='first')]['step'][1:]:
                plt.axvline(x=step, lw=0.5, color='red', alpha=0.4)
            add_footnote(save_dir)
            savefig(os.path.join(out_dir, name))
            plt.close(fig)


if __name__ == "__main__":
    cli_main()
