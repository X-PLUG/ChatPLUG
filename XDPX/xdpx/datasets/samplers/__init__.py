import math
import os
import importlib
import torch
import numpy as np
from functools import partial
from xdpx.utils import register, numpy_seed

samplers = {}
register = partial(register, registry=samplers)


@register('default')
class SortedSampler(torch.utils.data.Sampler):
    """
    A sampler that supports distributed training and batch by length.
    """
    @classmethod
    def register(cls, options):
        ...

    def __init__(self, args, dataset, batch_size, shuffle, num_replicas, rank, update_freq=1, sort_key=None, seed=None):
        total_size = int(math.ceil(len(dataset) * 1.0 / num_replicas)) * num_replicas
        if sort_key and shuffle:
            keys = [sort_key(sample) for sample in dataset]
            sorted_indices = sorted(range(len(keys)), key=keys.__getitem__)
            bucket_len = batch_size * num_replicas * update_freq
            divide_point = len(sorted_indices) - len(sorted_indices) % bucket_len
            sorted_indices = sorted_indices[:divide_point]
            sorted_batch_indices = [sorted_indices[i: i + bucket_len] for i in
                                    range(0, len(sorted_indices), bucket_len)]
            # add extra samples to make it evenly divisible
            extra_samples = sorted_indices[divide_point:] + sorted_indices[:(total_size - len(sorted_indices))]
        else:
            sorted_batch_indices = None
            extra_samples = []

        self.args = args
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.update_freq = update_freq
        self.epoch = 0
        self.num_samples = total_size // self.num_replicas
        self.sorted_batch_indices = sorted_batch_indices
        self.extra_samples = extra_samples
        self.total_size = total_size
        self.seed = seed

    def __iter__(self):
        # deterministically shuffle based on epoch
        seed = (self.seed, self.epoch) if self.seed is not None else (None,)

        def shuffle_with_seed(batches):
            with numpy_seed(*seed):
                np.random.shuffle(batches)
            return batches

        if not self.sorted_batch_indices:
            if self.shuffle:
                indices = shuffle_with_seed(list(range(len(self.dataset))))
            else:
                indices = list(range(len(self.dataset)))

            # add extra samples to make it evenly divisible (in validation extra samples will be trimmed later)
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size
            # subsample
            indices = indices[self.rank: self.total_size: self.num_replicas]
            assert len(indices) == self.num_samples

            return iter(indices)

        assert self.shuffle
        batch_indices = shuffle_with_seed(list(range(len(self.sorted_batch_indices))))
        batch_start = self.rank * self.batch_size * self.update_freq

        indices = []
        for index in batch_indices:
            batch = self.sorted_batch_indices[index]
            indices.extend(batch[batch_start: batch_start + self.batch_size * self.update_freq])
        if self.extra_samples:
            extra_len = len(self.extra_samples) // self.num_replicas
            indices.extend(self.extra_samples[self.rank * extra_len: self.rank * extra_len + extra_len])
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('.' + module_name, __name__)
