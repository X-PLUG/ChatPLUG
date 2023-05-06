import os
import sys
import math
import json
import torch
import pandas as pd
from tqdm import tqdm
from typing import Dict
from xdpx.bootstrap import bootstrap
from xdpx.options import Options, Argument, Arg
from xdpx.evaluate import Evaluator
from xdpx.utils import io, move_to_cuda, pformat_dataframe


def cli_main(argv=sys.argv):
    options = Options()
    Evaluator.register(options)
    options.register(
        Argument('train_subset', required=True),
        Argument('train_steps', type=int, required=True),
        Argument('valid_subset', default='dev'),
        Argument('prune_config', required=True, type=Dict[int, int], doc='{layer: #pruned}',
                 post_process=lambda x: {int(key):int(val) for key, val in x.items()}),
        Argument('save_path'),
    )
    bootstrap(options, main, __file__, argv)


def compute_heads_importance(
    model, loss_module, data, head_mask, max_steps=math.inf,
):
    """compute head importance scores according to http://arxiv.org/abs/1905.10650"""
    # Prepare our tensors
    cuda = next(model.parameters()).is_cuda
    head_importance = torch.zeros_like(head_mask)
    if cuda:
        head_importance = head_importance.cuda()
        head_mask = head_mask.cuda()
    head_mask.requires_grad_()
    tot_tokens = 0.0

    with tqdm(data, total=min(max_steps, len(data)), desc="compute importance") as progress:
        try:
            for step, sample in enumerate(progress):
                if cuda:
                    sample = move_to_cuda(sample)
                model.zero_grad()
                if head_mask.grad is not None:
                    head_mask.grad.zero_()
                sample['net_input']['head_mask'] = head_mask
                loss, sample_size, _ = loss_module(model, sample)
                loss.backward()
                head_importance += head_mask.grad.abs().detach()
                tot_tokens += sample_size
                if step >= max_steps:
                    break
        except KeyboardInterrupt:
            pass

    # Normalize
    head_importance /= tot_tokens
    # Layerwise importance normalization
    exponent = 2
    norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
    head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    # Head ranked by importance scores
    head_ranks = torch.zeros(head_importance.numel()).to(device=head_mask.device, dtype=torch.long)
    head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(head_importance.numel()).to(head_ranks)
    head_ranks = head_ranks.view_as(head_importance)

    return head_importance, head_ranks


def eval_pruning(model, loss, task, data, head_mask, max_eval_steps=None):
    cuda = next(model.parameters()).is_cuda
    logging_outputs, sample_size = [], 0
    model.eval()
    loss.eval()

    with tqdm(data, desc='evaluate') as progress:
        for i, sample in enumerate(progress):
            if max_eval_steps is not None and i >= max_eval_steps:
                break
            sample['net_input']['head_mask'] = head_mask
            if cuda:
                sample = move_to_cuda(sample)
            _loss, sample_size_i, logging_output = task.valid_step(
                sample, model, loss
            )
            logging_outputs.append(logging_output)
            sample_size += sample_size_i

    logging_output = task.aggregate_logging_outputs(logging_outputs, int(sample_size), loss)
    logging_output.pop('sample_size', None)
    return logging_output


def main(cli_args: Arg):
    if not cli_args.save_path:
        print('WARNING: "save_dir" not provided. Will evaluate head pruning only.')
    args, task, model, loss, loader = Evaluator.build_evaluation(cli_args)
    # load data
    train = task.load_dataset(cli_args.train_subset, is_train=True)
    dev = task.load_dataset(cli_args.valid_subset, is_train=False)

    head_mask = torch.ones(args.num_hidden_layers, args.num_attention_heads)
    init_eval = eval_pruning(model, loss, task, dev, head_mask)

    head_importance, head_ranks = compute_heads_importance(
        model, loss, train, head_mask,
        max_steps=cli_args.train_steps,
    )
    print('Head importance')
    print(pformat_dataframe(pd.DataFrame(head_importance), showindex=True, floatfmt='.3f'))
    print('Head ranks')
    print(pformat_dataframe(pd.DataFrame(head_ranks), showindex=True))

    for layer, count in cli_args.prune_config.items():
        head_mask[layer, head_ranks[layer].argsort(descending=True)[:count]] = 0
    pruned_eval = eval_pruning(model, loss, task, dev, head_mask)
    stats = pd.concat([pd.Series(init_eval, name='before'),
                       pd.Series(pruned_eval, name='after')], axis=1).transpose()
    print(pformat_dataframe(stats))

    heads_to_prune = dict(
        (layer, (1 - head_mask[layer].long()).nonzero().squeeze().tolist()) for layer in range(len(head_mask))
    )
    heads_to_prune = {key: val for key, val in heads_to_prune.items() if val}

    print('Heads to prune:')
    print(heads_to_prune)
    orig_params = sum(p.numel() for p in model.parameters())
    model.bert.prune_heads(heads_to_prune)
    new_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {orig_params} -> {new_params} ({new_params / orig_params * 100:.1f}%)')
    if cli_args.save_path:
        print('Save to', cli_args.save_path)
        io.makedirs(os.path.dirname(cli_args.save_path), exist_ok=True)
        with io.open(cli_args.save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        config = model.get_bert_config()
        config['pruned_heads'] = heads_to_prune
        config_path = os.path.join(os.path.dirname(cli_args.save_path), 'config.json')
        with io.open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


if __name__ == '__main__':
    cli_main()
