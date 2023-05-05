import math
import torch
from functools import partial, update_wrapper
from torch import nn
from torch.nn import functional as F
from packaging import version
from xdpx.utils.profiling import measure_time
from xdpx.logger import log
from .pkm import eval_memory_usage, get_codebook, kl_score, gini_score


class LocallyOptimizedHashingMemory(nn.Module):
    MEM_VALUES_PARAMS = '.values.weight'
    VALUES = None
    EVAL_MEMORY = True
    instance_id = 0

    def __init__(
            self, input_dim, output_dim, k_dim, n_keys, heads, knn,
            input_dropout=0., value_dropout=0., sparse=False, share_values=False
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
        self.sparse = sparse
        self.instance_id = self.__class__.instance_id
        self.__class__.instance_id += 1
        assert self.k_dim >= 2 and self.k_dim % 2 == 0

        # dropout
        self.input_dropout = input_dropout
        self.value_dropout = value_dropout

        # initialize keys / values
        self.keys = nn.Parameter(torch.empty(heads, 2, n_keys, k_dim // 2).uniform_(1 / math.sqrt(k_dim // 2)))
        self.values = get_codebook(self.size * heads, self.v_dim, sparse)
        nn.init.normal_(self.values.weight, mean=0, std=self.v_dim ** -0.5)
        # initialize dist_w to the coefficient in Wilson–Hilferty transformation
        self.dist_w = nn.Parameter(torch.ones(heads) * math.sqrt(2 * self.k_dim))
        # TODO: figure out why dist_w is NaN when fp16 is on
        self.dist_w.register_hook(lambda grad: grad.masked_fill(torch.isnan(grad), 0))
        self.first_memory = True
        if share_values:
            if self.__class__.VALUES is None:
                self.__class__.VALUES = self.values.weight
            else:
                self.values.weight = self.__class__.VALUES
                self.first_memory = False

        # query net
        self.head_center = nn.Parameter(torch.randn(heads, input_dim))
        self.query_proj = nn.ModuleList([
            nn.Linear(self.input_dim, self.k_dim) for _ in range(self.heads)
        ])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        if not self.first_memory:
            del state_dict[prefix + self.MEM_VALUES_PARAMS.lstrip('.')]
        return state_dict

    def forward(self, input, mask=None, return_index=False):
        """
        Read from the memory.
        """
        prefix = '' if self.__class__.instance_id == 1 else f'{self.instance_id}_'
        if self.dist_w.requires_grad:
            for i in range(self.heads):
                log.add_summary(prefix + f'dist_w_{i}', self.dist_w[i])
            with torch.no_grad():
                self.dist_w.clamp_(min=0.1)
        input = F.dropout(input, p=self.input_dropout, training=self.training)             # (...,i_dim)
        # input dimensions
        assert input.shape[-1] == self.input_dim
        prefix_shape = input.shape[:-1]
        if mask is None:
            input = input.reshape(-1, self.input_dim)
        else:
            input = input.masked_select(mask).reshape(-1, self.input_dim)

        scores, indices = self.get_indices_from_input(input)

        # weighted sum of values
        if version.parse(torch.__version__) < version.parse('1.1'):
            output = self.values(indices)                                                  # (bs,-,v_dim)
            with measure_time(self, 'weighted'):
                output *= scores.unsqueeze(-1)
            with measure_time(self, 'output_sum'):
                output = output.sum(1)                                                     # (bs,v_dim)
        else:
            output = self.values(indices, per_sample_weights=scores)                       # (bs,v_dim)
        output = F.dropout(output, p=self.value_dropout, training=self.training)           # (bs,v_dim)

        # reshape output
        if len(prefix_shape) >= 2:
            if mask is None:
                output = output.view(prefix_shape + (self.v_dim,))                         # (...,v_dim)
            else:
                output = input.new_zeros(prefix_shape + (self.v_dim,)) \
                    .masked_scatter_(mask, output)                                         # (...,v_dim)

        # store indices / scores (eval mode only - for usage statistics)
        if not self.training and self.EVAL_MEMORY:
            log.add_accumulative_summary(
                name=f'mem_stats_{self.instance_id}',
                prefix=prefix,
                values=(
                    scores.detach().cpu().float(),
                    indices.detach().cpu(),
                ),
                reduce_fn=partial(eval_memory_and_head_usage, mem_size=self.size,
                                  head_size=self.heads),
            )
        if return_index:
            return output, indices
        return output

    def get_indices_from_input(self, input):
        bs, i_dim = input.size()

        with measure_time(self, 'head_score'):
            head_dist = ((input.view(bs, 1, i_dim) -
                          self.head_center.view(1, self.heads, i_dim)
                          ) ** 2).sum(2)                                              # (bs,heads)
            head_score = -head_dist                                                   # (bs,heads)

        head_index = head_score.argmax(dim=1)                                         # (bs,)
        scores = input.new_zeros((bs, self.knn))
        indices = input.new_zeros((bs, self.knn), dtype=head_index.dtype)

        def rescale_grad(grad, multiplier):
            return grad * multiplier

        for i in range(self.heads):
            mask = head_index == i
            bs_i = mask.sum().item()
            if bs_i == 0:
                continue
            with measure_time(self, f'mask_get_{i}'):
                input_i = input[mask]                                                 # (bs_h, i_dim)
            with measure_time(self, f'residual_{i}'):
                query = input_i - self.head_center[i].view(1, i_dim)                  # (bs_h, i_dim)
            query = self.query_proj[i](query)                                         # (bs_h, k_dim)

            with measure_time(self, f'get_indices_{i}'):
                (scores_i, indices_i) = \
                    self._get_indices(query, self.keys[i])                            # (bs_h, knn)
            scores_i = scores_i * self.dist_w[i]

            if scores_i.requires_grad:
                rescale_grad_i = partial(rescale_grad, multiplier=bs / bs_i)
                update_wrapper(rescale_grad_i, rescale_grad)
                scores_i.register_hook(rescale_grad_i)

            with measure_time(self, f'mask_put_{i}'):
                scores[mask] = scores_i
                indices[mask] = indices_i
        if scores.dtype == torch.float16:
            scores = scores.float()
            scores[torch.isinf(scores[:, 0]), :] = 0.
        scores = F.softmax(scores, dim=1).type_as(input)
        indices.add_(head_index.unsqueeze(1), alpha=self.size)
        return scores, indices

    def _get_indices(self, query, subkeys):
        """
        Generate scores and indices for a specific head.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        bs = query.size(0)
        knn = self.knn
        half = self.k_dim // 2
        n_keys = len(subkeys[0])

        # split query for product quantization
        q1 = query[:, :half]                                                          # (bs,half)
        q2 = query[:, half:]                                                          # (bs,half)

        # compute indices with associated scores
        d1 = q1.view(bs, 1, half) - subkeys[0].view(1, n_keys, half)                  # (bs,n_keys,half)
        d2 = q2.view(bs, 1, half) - subkeys[1].view(1, n_keys, half)                  # (bs,n_keys,half)

        # divided by Wilson–Hilferty transformation coefficient
        # before summation to avoid FP16 overflow
        d1 = (d1 ** 2 / self.k_dim).sum(2)                                            # (bs,n_keys)
        d2 = (d2 ** 2 / self.k_dim).sum(2)                                            # (bs,n_keys)
        scores1 = -d1                                                                 # (bs,n_keys)
        scores2 = -d2                                                                 # (bs,n_keys)

        scores1, indices1 = scores1.topk(knn, dim=1)                                  # (bs,knn)
        scores2, indices2 = scores2.topk(knn, dim=1)                                  # (bs,knn)

        # cartesian product on best candidate keys
        all_scores = (
            scores1.view(bs, knn, 1).expand(bs, knn, knn) +
            scores2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)                                                                # (bs,knn**2)
        all_indices = (
            indices1.view(bs, knn, 1).expand(bs, knn, knn) * n_keys +
            indices2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)                                                                # (bs,knn**2)

        # select best scores with associated indices
        scores, best_indices = torch.topk(all_scores, k=knn, dim=1)                   # (bs,knn)
        indices = all_indices.gather(1, best_indices)                                 # (bs,knn)

        # standardize Chi-square distribution by Wilson–Hilferty transformation
        scores = -torch.sqrt(-scores)

        assert scores.shape == indices.shape == (bs, knn)
        return scores, indices


def eval_memory_and_head_usage(mem_att, mem_size, head_size):
    import numpy as np
    stats = eval_memory_usage(mem_att, mem_size * head_size)
    if head_size > 1:
        head_scores = np.zeros(head_size, dtype=np.float32)
        for weights, indices in mem_att:
            head_indices = indices.numpy() // mem_size
            np.add.at(head_scores, head_indices, 1)
        head_scores /= head_scores.sum()
        for i in range(head_size):
            stats[f'head_usage_{i}'] = head_scores[i].item()
        stats.update(dict(
            head_kl=float(kl_score(head_scores)),
            head_gini=float(gini_score(head_scores)),
        ))

    top_weights, tail_weights = map(torch.tensor,
                                    zip(*((weight[0], weight[-1]) for weights, _ in mem_att for weight in weights)))
    stats.update(dict(
        top_tail_ratio=top_weights.sum().item() / tail_weights.sum().item(),
    ))
    return stats
