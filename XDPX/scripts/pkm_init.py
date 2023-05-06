import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from functools import partial
from collections import defaultdict
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from xdpx.options import Options, Argument, Arg
from xdpx.utils import io, log_to_file, move_to_cpu, pformat_dataframe, get_train_subsets
from xdpx.evaluate import Evaluator
from xdpx.bootstrap import bootstrap
from xdpx.optimizers.adam import AdamW
from xdpx.modules.projections.lopkm import eval_memory_and_head_usage, gini_score, kl_score


def cli_main(argv=sys.argv):
    options = build_options()
    bootstrap(options, main, __file__, argv)


def build_options():
    options = Options()
    Evaluator.register(options)
    options.register(
        Argument('save_path', required=True),
        Argument('input_size', type=int, required=True),
        Argument('output_size', type=int, required=True),
        Argument('log_file', type=str, post_process=lambda val, args: val if val is not None else
        os.path.splitext(args.save_path)[0] + '.log.txt'),
    )
    options.register(
        Argument('k_dim', type=int, required=True),
        Argument('nkeys', type=int, required=True),
        Argument('nheads', type=int, required=True),
        Argument('knn', type=int, required=True),
        domain='pkm'
    )
    with options.with_prefix('coarse_kmeans_'):
        options.register(
            Argument('iter', default=500),
            Argument('tolerance', default=math.inf),
            Argument('seed', default=1),
            Argument('n_init', default=50, validate=lambda value: value > 1),
            Argument('reassignment_ratio', default=0.01),
            domain='coarse_kmeans'
        )
    with options.with_prefix('pca_'):
        options.register(
            Argument('iter', default=200)
        )
    with options.with_prefix('fine_kmeans_'):
        options.register(
            Argument('iter', default=2000),
            Argument('seed', default=1),
            Argument('n_init', default=4, validate=lambda value: value > 1),
            Argument('reassignment_ratio', default=0.01),
            domain='fine_kmeans'
        )
    options.register(
        Argument('value_init_iter', default=1000),
        Argument('value_finetune_iter', default=8000),
        Argument('value_finetune_lr', default=0.001),
        domain='value_init'
    )
    return options


def main(cli_args: Arg):
    if cli_args.log_file:
        log_to_file(cli_args.log_file, prefix='')
    options = build_options()
    print(options.tree(cli_args))
    args, task, model, loss, loader = Evaluator.build_evaluation(cli_args)
    cli_args.max_eval_steps = args.max_eval_steps
    target_layers = args.teacher_target_layers
    share_memory = args.student_share_memory and len(target_layers) > 1
    modules = []
    usages = []
    for i in target_layers:
        task.build_model(args.change(teacher_target_layers=[i]))
        # only use the first split to initialize
        train = task.load_dataset(get_train_subsets(args)[:1], is_train=True, reload=True)
        dev = task.load_dataset(args.valid_subset, is_train=False, reload=True)

        print(f'Initializing memory for layer {i}')
        memory = model.memories[model.target_layers[i]]
        if cli_args.cuda:
            memory = memory.cuda()
        memory.eval()
        initial_usage = eval_usage(memory, dev)

        # init parameters
        init_head_center(cli_args, memory, train, dev)
        init_query_net(cli_args, memory, train, dev)
        init_mem_keys(cli_args, memory, train, dev)
        if not share_memory:
            init_mem_values(cli_args, memory, train, dev)

        # compare usage
        improved_usage = eval_usage(memory, dev)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print('\n', pd.concat([pd.Series(initial_usage, name='before'),
                                   pd.Series(improved_usage, name='after')], axis=1).transpose())
        modules.append(memory)
        usages.append(improved_usage)
    if share_memory:
        task.build_model(args)
        train = task.load_dataset(get_train_subsets(args)[:1], is_train=True, reload=True)
        dev = task.load_dataset(args.valid_subset, is_train=False, reload=True)
        shared_usage = eval_usage(modules, dev)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print('\n', pd.concat([pd.Series({key: np.mean([usage[key] for usage in usages]).item()
                                              for key in usages[0].keys()}, name='single_avg'),
                                   pd.Series(shared_usage, name='shared')], axis=1).transpose())
        # initialize shared memory
        init_mem_values(cli_args, modules, train, dev)

    # save state dict
    with io.open(cli_args.save_path, 'wb') as f:
        torch.save({'memories.'+key: val for key, val in nn.ModuleList(modules).state_dict().items()}, f)


def eval_usage(memory, data):
    mem_att = []
    if not isinstance(memory, list):
        memory = [memory]
    assert len(memory) == len(data.target_layers)
    memory_map = {layer_id: memory[i] for i, layer_id in enumerate(data.target_layers)}
    for batch in tqdm(data, desc=f'eval usage'):
        inputs = batch['net_input']['inputs']
        memory = memory_map[batch['net_input']['layer_id']]
        if next(memory.parameters()).is_cuda:
            inputs = inputs.cuda()
        with torch.no_grad():
            mem_att.append(move_to_cpu(memory.get_indices_from_input(inputs)))
    return eval_memory_and_head_usage(mem_att, memory.size, memory.heads)


def init_head_center(args, memory, data, valid_data):
    args = args.strip_prefix('coarse_kmeans_')

    # train kmeans
    init_size = max(args.n_init, 10 * memory.heads // args.batch_size)
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
    kmeans = MiniBatchKMeans(
        n_clusters=memory.heads, compute_labels=True,
        random_state=args.seed, reassignment_ratio=args.reassignment_ratio,
    )
    # max_iter/max_no_improvements/batch_size/init_size are not in effect in "fit_partial", so they're implemented here.
    ewa_inertia = None
    ewa_inertia_min = None
    no_improvement = 0
    epoch = 0
    step = 0
    init_cache = []
    with tqdm(total=args.iter, desc='init mem heads') as progress:
        while no_improvement < args.tolerance and step <= args.iter:
            epoch += 1
            for sample in data:
                step += 1
                if step > args.iter:
                    break
                progress.update()
                inputs = sample['net_input']['inputs']
                if ewa_inertia is None:
                    # accumulate init_size at first
                    if len(init_cache) < init_size:
                        init_cache.append(inputs)
                        continue
                    else:
                        inputs = torch.cat(init_cache, 0)
                        init_cache.clear()
                batch_size = len(inputs)
                kmeans.partial_fit(inputs.numpy())
                batch_inertia = kmeans.inertia_
                # Normalize inertia to be able to compare values when
                # batch_size changes
                batch_inertia /= batch_size
                if ewa_inertia is None:
                    ewa_inertia = batch_inertia
                else:
                    alpha = float(batch_size) * 2.0 / (len(data) + 1)
                    alpha = 1.0 if alpha > 1.0 else alpha
                    ewa_inertia = ewa_inertia * (1 - alpha) + batch_inertia * alpha
                    if step % 10 == 0:
                        progress.set_postfix(dict(
                            loss=f'{batch_inertia:.1f}',
                            ewa=f'{ewa_inertia:.1f}'
                        ))
                if ewa_inertia_min is None or ewa_inertia < ewa_inertia_min:
                    no_improvement = 0
                    ewa_inertia_min = ewa_inertia
                else:
                    no_improvement += 1
                if no_improvement >= args.tolerance:
                    break

    print('head center KMEANS loss: ', ewa_inertia_min)
    head_center = kmeans.cluster_centers_

    # load into model
    memory.load_state_dict({
        'head_center': torch.from_numpy(head_center),
    }, strict=False)

    head_indices = defaultdict(lambda: 0)
    with torch.no_grad():
        for sample in tqdm(valid_data, desc='evaluating heads'):
            inputs = sample['net_input']['inputs']
            if next(memory.parameters()).is_cuda:
                inputs = inputs.cuda()
            bs, i_dim = inputs.size()
            head_dist = ((inputs.view(bs, 1, i_dim) -
                          memory.head_center.view(1, memory.heads, i_dim)
                          ) ** 2).sum(2)
            head_index = head_dist.argmin(dim=1)
            for index in head_index.cpu().tolist():
                head_indices[index] += 1
    counts = pd.Series(head_indices).sort_index()
    print('Head assignment on dev:')
    print('\n', pformat_dataframe(counts.to_frame('counts'), showindex=True))
    counts /= counts.sum()
    print(f'gini {gini_score(counts):.5f} KL {kl_score(counts):.5f}')


def init_query_net(args, memory, train, dev):
    args = args.strip_prefix('pca_')
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA
    modules = [IncrementalPCA(n_components=memory.k_dim) for _ in range(memory.heads)]
    step = 0
    cache = defaultdict(lambda: [])
    batch_size = max(args.batch_size, memory.k_dim * 4)
    with torch.no_grad():
        with tqdm(total=args.iter, desc='init_query_net') as progress:
            while step <= args.iter:
                for sample in train:
                    step += 1
                    progress.update()
                    if step > args.iter:
                        break
                    inputs = sample['net_input']['inputs']
                    if next(memory.parameters()).is_cuda:
                        inputs = inputs.cuda()
                    bs, i_dim = inputs.size()
                    head_dist = ((inputs.view(bs, 1, i_dim) -
                                  memory.head_center.view(1, memory.heads, i_dim)
                                  ) ** 2).sum(2)
                    head_index = head_dist.argmin(dim=1)
                    for i in range(memory.heads):
                        mask = head_index == i
                        if mask.sum().item() == 0:
                            continue
                        input_i = inputs[mask]
                        query = input_i - memory.head_center[i].view(1, i_dim)
                        query = query.cpu().numpy()
                        cache[i].append(query)
                        if sum(map(len, cache[i])) >= batch_size:
                            pca = modules[i]
                            batch = np.concatenate(cache[i])
                            pca.partial_fit(batch)
                            cache[i].clear()
    stats = []
    for i in range(memory.heads):
        pca = modules[i]
        try:
            var = pca.explained_variance_
        except AttributeError:
            print(f'WARNING: query net in head {i} does not have enough samples to fit')
            continue
        stats.append([i, pca.n_samples_seen_, f'{var[0]:.2f}', f'{var[-1]/var[0]*100:.2f}%'])
        W, b = pca.components_, pca.mean_
        b = W @ b  # in scikit-learn the bias are estimated in the original space

        # evenly distribute
        W2 = np.concatenate([W[np.arange(0, args.k_dim, 2)], W[np.arange(1, args.k_dim, 2)]])
        b2 = np.concatenate([b[np.arange(0, args.k_dim, 2)], b[np.arange(1, args.k_dim, 2)]])
        memory.query_proj[i].load_state_dict({
            'weight': torch.from_numpy(W2),
            'bias': torch.from_numpy(b2),
        })
    print('\n', pformat_dataframe(pd.DataFrame(stats, columns='head_id sample var[0] var[-1]/var[0]'.split())))


def init_mem_keys(args, memory, train, dev):
    args = args.strip_prefix('fine_kmeans_')
    half = args.k_dim // 2
    modules = [[MiniBatchKMeans(
        n_clusters=memory.n_keys, compute_labels=True,
        random_state=args.seed, reassignment_ratio=args.reassignment_ratio,
    )for _ in range(2)] for _ in range(memory.heads)]

    epoch = 0
    step = 0
    cache = defaultdict(lambda: [])
    batch_size = max(args.batch_size, memory.n_keys)
    with torch.no_grad():
        with tqdm(total=args.iter, desc='init_mem_keys') as progress:
            while step <= args.iter:
                epoch += 1
                for sample in train:
                    step += 1
                    progress.update()
                    if step > args.iter:
                        break
                    inputs = sample['net_input']['inputs']
                    if next(memory.parameters()).is_cuda:
                        inputs = inputs.cuda()
                    bs, i_dim = inputs.size()
                    head_dist = ((inputs.view(bs, 1, i_dim) -
                                  memory.head_center.view(1, memory.heads, i_dim)
                                  ) ** 2).sum(2)
                    head_index = head_dist.argmin(dim=1)
                    for i in range(memory.heads):
                        mask = head_index == i
                        if mask.sum().item() == 0:
                            continue
                        input_i = inputs[mask]
                        query = input_i - memory.head_center[i].view(1, i_dim)
                        query = memory.query_proj[i](query)
                        query = query.cpu().numpy()
                        cache[i].append(query)
                        if not hasattr(modules[i][0], 'inertia_') and \
                                sum(map(len, cache[i])) < batch_size * args.n_init:
                            continue
                        if sum(map(len, cache[i])) >= batch_size:
                            query = np.concatenate(cache[i])
                            cache[i].clear()
                            q1 = np.ascontiguousarray(query[:, :half])
                            q2 = np.ascontiguousarray(query[:, half:])
                            kmeans1, kmeans2 = modules[i][0], modules[i][1]
                            kmeans1.partial_fit(q1)
                            kmeans2.partial_fit(q2)
                            if step % 10 == 0:
                                progress.set_postfix(dict(
                                    loss1=f'{kmeans1.inertia_ / len(query):.1f}',
                                    loss2=f'{kmeans2.inertia_ / len(query):.1f}'
                                ))

    keys = []
    for i in range(memory.heads):
        centroids = []
        for j in range(2):
            try:
                centroids.append(torch.from_numpy(modules[i][j].cluster_centers_))
            except AttributeError:
                print(f'WARNING: Head {i} subkey {j} does not have enough samples to initialize.')
                centroids.append(torch.empty(memory.n_keys, half).uniform_(1 / math.sqrt(memory.k_dim // 2)))
        keys.append(torch.stack(centroids))
    keys = torch.stack(keys)
    memory.load_state_dict({
        'keys': keys,
    }, strict=False)

    records = defaultdict(lambda: defaultdict(lambda: 0))
    with torch.no_grad():
        for sample in tqdm(dev, desc='evaluating keys'):
            inputs = sample['net_input']['inputs']
            if next(memory.parameters()).is_cuda:
                inputs = inputs.cuda()
            bs, i_dim = inputs.size()
            head_dist = ((inputs.view(bs, 1, i_dim) -
                          memory.head_center.view(1, memory.heads, i_dim)
                          ) ** 2).sum(2)
            head_index = head_dist.argmin(dim=1)
            for i in range(memory.heads):
                mask = head_index == i
                if mask.sum().item() == 0:
                    continue
                input_i = inputs[mask]
                bs_i = len(input_i)
                query = input_i - memory.head_center[i].view(1, i_dim)
                query = memory.query_proj[i](query)
                subkeys = memory.keys[i]
                q1 = query[:, :half]
                q2 = query[:, half:]
                d1 = ((q1.view(bs_i, 1, half) - subkeys[0].view(1, memory.n_keys, half))**2).sum(2)
                d2 = ((q2.view(bs_i, 1, half) - subkeys[1].view(1, memory.n_keys, half))**2).sum(2)
                index1 = d1.argmin(dim=1)
                index2 = d2.argmin(dim=1)
                indices = (index1, index2)
                for j in range(2):
                    for index in indices[j].cpu().tolist():
                        records[2 * i + j][index] += 1
    stats = []
    for i in range(memory.heads):
        for j in range(2):
            counts = records[2 * i + j].values()
            stats.append([i, j, f'{len(counts)/memory.n_keys*100:.0f}%', ' '.join(map(str, counts))])
    print('\n', pformat_dataframe(pd.DataFrame(stats, columns='head subkey perc value_counts'.split())))


def eval_loss(memory_map, data, max_steps=None):
    mse = 0
    counts = 0
    with torch.no_grad():
        with tqdm(total=min(len(data), max_steps or math.inf), desc='eval loss') as progress:
            for i, sample in enumerate(data):
                if max_steps is not None and i >= max_steps:
                    break
                progress.update()
                inputs = sample['net_input']['inputs']
                target = sample['target']
                memory = memory_map[sample['net_input']['layer_id']]
                if next(memory.parameters()).is_cuda:
                    inputs = inputs.cuda()
                    target = target.cuda()
                logits = memory(inputs)
                mse += ((logits - target) ** 2).sum().item()
                counts += inputs.numel()
    mse /= max(counts, 1)
    return mse


def init_mem_values(args: Arg, memories, train, dev):
    if not isinstance(memories, list):
        memories = [memories]
    if len(memories) > 1:
        # only when values are shared
        memory_map = {layer_id: memories[i] for i, layer_id in enumerate(train.target_layers)}
        assert len(memories) == len(train.target_layers) == len(dev.target_layers)
    else:
        memory_map = defaultdict(lambda: memories[0])
        assert 1 == len(memories) == len(train.target_layers) == len(dev.target_layers)

    init_loss = eval_loss(memory_map, dev, max_steps=args.max_eval_steps)
    print(f'Initial MSE:', init_loss)

    # initialize values by even distribution to all KNNs
    values = torch.zeros(memories[0].size * memories[0].heads, memories[0].v_dim)
    counts = torch.zeros(memories[0].size * memories[0].heads)
    if next(memories[0].parameters()).is_cuda:
        values = values.cuda()
        counts = counts.cuda()
    n = min(args.value_init_iter, len(train) * 60) * len(memories)
    step = 0
    with tqdm(total=n, desc=f'init_values') as progress:
        while step < n:
            for sample in train:
                step += 1
                progress.update()
                if step > n:
                    break
                inputs = sample['net_input']['inputs']
                target = sample['target']
                memory = memory_map[sample['net_input']['layer_id']]
                if next(memory.parameters()).is_cuda:
                    inputs = inputs.cuda()
                    target = target.cuda()
                with torch.no_grad():
                    _, indices = memory.get_indices_from_input(inputs)
                    # evenly distribute to all KNNs initially
                    scores = torch.ones_like(indices, dtype=torch.float) / indices.size(1)
                    indices = indices.view(-1)
                    weighted_outputs = scores.unsqueeze(2) * target.unsqueeze(1)
                    values[indices, :] += weighted_outputs.view(-1, memory.v_dim)
                    counts[indices] += scores.view(-1)
    values /= counts.unsqueeze_(1).clamp(min=1)
    weights = counts.clamp(max=1)
    values = memories[0].values.weight * (-weights + 1.) + values * weights
    with torch.no_grad():
        memories[0].values.weight.copy_(values.cpu())
        memories[0].dist_w.fill_(math.sqrt(2 * memory.k_dim))
    del values, counts
    dev_loss = eval_loss(memory_map, dev, max_steps=args.max_eval_steps)
    print(f'DEV mse after init: {dev_loss}')

    # finetune values by gradient descend
    optim_cls = torch.optim.SparseAdam if memories[0].sparse else partial(AdamW, weight_decay=0.0)
    optim = optim_cls([memories[0].values.weight], lr=args.value_finetune_lr)
    n = args.value_finetune_iter
    step = 0
    with tqdm(total=n, desc=f'init_values') as progress:
        accum_step = 0
        while step < n:
            for sample in train:
                inputs = sample['net_input']['inputs']
                target = sample['target']
                memory = memory_map[sample['net_input']['layer_id']]
                if next(memory.parameters()).is_cuda:
                    inputs = inputs.cuda()
                    target = target.cuda()

                logits = memory(inputs)
                loss = ((target - logits) ** 2).mean()
                loss.backward()
                accum_step += 1
                if accum_step < len(memories):
                    continue
                accum_step = 0
                step += 1
                progress.update()

                gnorm = torch.sqrt(sum((
                    p.grad.data.norm() ** 2 for p in (memory.values.weight,)
                )))
                optim.step()
                optim.zero_grad()
                progress.set_postfix(dict(loss=loss.item(), gnorm=gnorm.item()))
                if step % (n // 5) == 0 and step < n:
                    dev_loss = eval_loss(memory_map, dev, max_steps=args.max_eval_steps)
                    print(f'DEV mse @{step}: {dev_loss}')
                if step >= n:
                    break
    print('final DEV mse:', eval_loss(memory_map, dev, max_steps=args.max_eval_steps))


if __name__ == '__main__':
    cli_main()
