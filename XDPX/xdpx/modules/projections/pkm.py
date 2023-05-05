import math
import numpy as np
import torch
from functools import partial
from torch import nn
from torch.nn import functional as F
from packaging import version
from xdpx.utils.profiling import measure_time
from xdpx.logger import log


def get_uniform_keys(n_keys, dim, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    keys = rng.uniform(-bound, bound, (n_keys, dim))
    return keys.astype(np.float32)


def get_codebook(size, v_dim, sparse=False):
    if version.parse(torch.__version__) < version.parse('1.1'):
        values = nn.Embedding(size, v_dim, sparse=sparse)
    else:
        values = nn.EmbeddingBag(size, v_dim, mode='sum', sparse=sparse)
    return values


class HashingMemory(nn.Module):
    MEM_VALUES_PARAMS = '.values.weight'
    VALUES = None
    EVAL_MEMORY = True

    def __init__(
        self, input_dim, output_dim, k_dim, n_keys, heads, knn, query_net='linear',
        input_dropout=0., query_dropout=0., value_dropout=0., distance_fn='dot',
        sparse=False, query_norm=None, share_values=False, init_kernel_alpha=0.5,
    ):
        super().__init__()

        # global parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_dim = k_dim
        self.v_dim = output_dim
        self.n_keys = n_keys
        self.size = self.n_keys ** 2
        self.heads = heads
        self.knn = knn
        self.distance_fn = distance_fn
        self.kernel_alpha = nn.Parameter(torch.tensor(init_kernel_alpha))
        assert self.k_dim >= 2 and self.k_dim % 2 == 0

        # dropout
        self.input_dropout = input_dropout
        self.query_dropout = query_dropout
        self.value_dropout = value_dropout

        # initialize keys / values
        self.initialize_keys()  # (heads,2,n_keys,half)
        if distance_fn == 'mah':
            self.kernel_cov = nn.Parameter(torch.eye(k_dim // 2).expand(heads, 2, n_keys, -1, -1).clone())
        elif distance_fn == 'mah_fast':
            self.kernel_cov = nn.Parameter(torch.ones(heads, 2, n_keys, k_dim // 2))
        elif distance_fn == 'euc':
            self.kernel_cov = nn.Parameter(torch.ones(heads, 2, n_keys, 1))
        else:
            self.kernel_cov = [None] * heads

        self.values = get_codebook(self.size, self.v_dim, sparse)
        if share_values:
            if HashingMemory.VALUES is None:
                HashingMemory.VALUES = self.values.weight
            else:
                self.values.weight = HashingMemory.VALUES

        nn.init.normal_(self.values.weight, mean=0, std=self.v_dim ** -0.5)

        # query network
        if query_net == 'linear':
            query_net = [nn.Linear(self.input_dim, self.heads * self.k_dim, bias=True)]
        elif query_net == 'mlp':
            query_net = [
                nn.Linear(self.input_dim, self.input_dim),
                nn.Tanh(),
                nn.Linear(self.input_dim, self.heads * self.k_dim),
            ]
        else:
            raise NotImplementedError
        if query_norm == 'batchnorm':
            query_norm = nn.BatchNorm1d(self.heads * self.k_dim)
        elif query_norm == 'layernorm':
            query_norm = nn.LayerNorm(self.heads * self.k_dim)
        elif query_norm == 'groupnorm':
            query_norm = nn.GroupNorm(8, self.heads * self.k_dim)
        elif query_norm is not None:
            raise NotImplementedError
        self.query_proj = nn.Sequential(*filter(None, [
            *query_net,
            query_norm,
        ]))

    def initialize_keys(self):
        """
        Create two subkey sets per head.
        `self.keys` is of shape (heads, 2, n_keys, k_dim // 2)
        """
        half = self.k_dim // 2
        keys = torch.from_numpy(np.array([
            get_uniform_keys(self.n_keys , half, seed=(2 * i + j))
            for i in range(self.heads)
            for j in range(2)
        ])).view(self.heads, 2, self.n_keys, half)
        self.keys = nn.Parameter(keys)

    def _get_indices(self, query, subkeys, cov):
        """
        Generate scores and indices for a specific head.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        bs = query.size(0)
        knn = self.knn
        half = self.k_dim // 2
        n_keys = len(subkeys[0])

        # split query for product quantization
        q1 = query[:, :half]                                                                # (bs,half)
        q2 = query[:, half:]                                                                # (bs,half)

        # compute indices with associated scores
        if self.distance_fn == 'dot':
            scores1 = F.linear(q1, subkeys[0], bias=None)                                   # (bs,n_keys)
            scores2 = F.linear(q2, subkeys[1], bias=None)                                   # (bs,n_keys)
        elif self.distance_fn == 'euc' or self.distance_fn.startswith('mah'):
            d1 = q1.view(bs, 1, half) - subkeys[0].view(1, n_keys, half)                    # (bs,n_keys,half)
            d2 = q2.view(bs, 1, half) - subkeys[1].view(1, n_keys, half)                    # (bs,n_keys,half)

            log.add_summary('kernel_alpha', self.kernel_alpha)
            log.add_summary('kernel_cov_avg', self.kernel_cov.mean())
            log.add_summary('kernel_cov_median', self.kernel_cov.median())
            log.add_summary('kernel_cov_min', self.kernel_cov.min())
            log.add_summary('kernel_cov_max', self.kernel_cov.max())

            if self.distance_fn == 'mah':
                #    (bs,n_keys,1,half)    (1,n_keys,half,half)        (bs,n_keys,half,1)
                d1 = d1.unsqueeze(2).matmul(cov[0].unsqueeze(0)).matmul(d1.unsqueeze(3))    # (bs,n_keys,1,1)
                d2 = d2.unsqueeze(2).matmul(cov[1].unsqueeze(0)).matmul(d2.unsqueeze(3))    # (bs,n_keys,1,1)
                d1 = d1.squeeze(2).squeeze(2)
                d2 = d2.squeeze(2).squeeze(2)
            elif self.distance_fn == 'mah_fast' or self.distance_fn == 'euc':
                d1 = (d1 ** 2 / cov[0].unsqueeze(0)).sum(2)                                 # (bs,n_keys)
                d2 = (d2 ** 2 / cov[1].unsqueeze(0)).sum(2)                                 # (bs,n_keys)
            else:
                raise NotImplementedError
            scores1 = -self.kernel_alpha * d1                                               # (bs,n_keys)
            scores2 = -self.kernel_alpha * d2                                               # (bs,n_keys)
        else:
            raise NotImplementedError
        scores1, indices1 = scores1.topk(knn, dim=1)                  # (bs,knn)
        scores2, indices2 = scores2.topk(knn, dim=1)                  # (bs,knn)

        # cartesian product on best candidate keys
        all_scores = (
            scores1.view(bs, knn, 1).expand(bs, knn, knn) +
            scores2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)                                                # (bs,knn**2)
        all_indices = (
            indices1.view(bs, knn, 1).expand(bs, knn, knn) * n_keys +
            indices2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)                                                # (bs,knn**2)

        # select best scores with associated indices
        scores, best_indices = torch.topk(all_scores, k=knn, dim=1)   # (bs,knn)
        indices = all_indices.gather(1, best_indices)                 # (bs,knn)

        assert scores.shape == indices.shape == (bs, knn)
        return scores, indices

    def get_indices(self, query):
        """
        Generate scores and indices.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        query = query.view(-1, self.heads, self.k_dim)
        bs = len(query)
        with measure_time(self, 'get_indices'):
            outputs = [self._get_indices(query[:, i], self.keys[i], self.kernel_cov[i]) for i in range(self.heads)]
        s = torch.cat([s.view(bs, 1, self.knn) for s, _ in outputs], 1)  # (bs,heads,knn)
        i = torch.cat([i.view(bs, 1, self.knn) for _, i in outputs], 1)  # (bs,heads,knn)
        return s.view(-1, self.knn), i.view(-1, self.knn)

    def forward(self, input, mask=None, return_index=False):
        """
        Read from the memory.
        """
        input = F.dropout(input, p=self.input_dropout, training=self.training)  # (...,i_dim)
        # input dimensions
        assert input.shape[-1] == self.input_dim
        prefix_shape = input.shape[:-1]
        if mask is None:
            input = input.reshape(-1, self.input_dim)
        else:
            input = input.masked_select(mask).reshape(-1, self.input_dim)
        bs = input.size(0)

        # compute query
        query = self.query_proj(input)                                          # (bs,heads*k_dim)
        query = query.view(bs * self.heads, self.k_dim)                         # (bs*heads,k_dim)
        query = F.dropout(query, p=self.query_dropout, training=self.training)  # (bs*heads,k_dim)

        # retrieve indices and scores
        scores, indices = self.get_indices(query)                               # (bs*heads,knn)
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)              # (bs*heads,knn)

        # merge heads / knn (since we sum heads)
        indices = indices.view(bs, self.heads * self.knn)                       # (bs,heads*knn)
        scores = scores.view(bs, self.heads * self.knn)                         # (bs,heads*knn)

        # weighted sum of values
        if version.parse(torch.__version__) < version.parse('1.1'):
            output = self.values(indices)                                       # (bs, heads*knn, v_dim)
            with measure_time(self, 'weighted'):
                output *= scores.unsqueeze(-1)
            with measure_time(self, 'output_sum'):
                output = output.sum(1)                                          # (bs, v_dim)
        else:
            output = self.values(indices, per_sample_weights=scores)            # (bs,v_dim)
        output = F.dropout(output, p=self.value_dropout, training=self.training)# (bs,v_dim)

        # reshape output
        if len(prefix_shape) >= 2:
            if mask is None:
                output = output.view(prefix_shape + (self.v_dim,))              # (...,v_dim)
            else:
                output = input.new_zeros(prefix_shape + (self.v_dim,))\
                    .masked_scatter_(mask, output)                              # (...,v_dim)

        # store indices / scores (eval mode only - for usage statistics)
        if not self.training and self.EVAL_MEMORY:
            log.add_accumulative_summary(
                name='mem_stats',
                values=(
                    scores.detach().cpu().float(),
                    indices.detach().cpu(),
                ),
                reduce_fn=partial(eval_memory_usage, mem_size=self.size),
            )
        if return_index:
            return output, indices
        return output


def eval_memory_usage(mem_att, mem_size):
    """
    Evaluate memory usage (HashingMemory / FFN).
    """
    # memory slot scores
    assert mem_size > 0
    mem_scores_w = np.zeros(mem_size, dtype=np.float32)  # weighted scores
    mem_scores_u = np.zeros(mem_size, dtype=np.float32)  # unweighted scores

    # sum each slot usage
    for weights, indices in mem_att:
        np.add.at(mem_scores_w, indices, weights)
        np.add.at(mem_scores_u, indices, 1)

    # compute the KL distance to the uniform distribution
    mem_scores_w = mem_scores_w / mem_scores_w.sum()
    mem_scores_u = mem_scores_u / mem_scores_u.sum()

    top50_w, top90_w, top99_w = tops(mem_scores_w)
    top50_u, top90_u, top99_u = tops(mem_scores_u)

    return {
        'mem_used': float(100 * (mem_scores_w != 0).sum() / len(mem_scores_w)),
        'mem_w_kl': float(kl_score(mem_scores_w)),
        'mem_u_kl': float(kl_score(mem_scores_u)),
        'mem_w_gini': float(gini_score(mem_scores_w)),
        'mem_u_gini': float(gini_score(mem_scores_u)),
        'mem_w_top50': int(top50_w),
        'mem_w_top90': int(top90_w),
        'mem_w_top99': int(top99_w),
        'mem_u_top50': int(top50_u),
        'mem_u_top90': int(top90_u),
        'mem_u_top99': int(top99_u),
    }


def tops(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    y = np.cumsum(np.sort(x))
    top50, top90, top99 = y.shape[0] - np.searchsorted(y, [0.5, 0.1, 0.01])
    return top50, top90, top99


def kl_score(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    _x = x.copy()
    _x[x == 0] = 1
    return np.log(len(x)) + (x * np.log(_x)).sum()


def gini_score(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    B = np.cumsum(np.sort(x)).mean()
    return 1 - 2 * B
