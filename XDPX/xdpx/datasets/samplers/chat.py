# -*- coding: utf-8 -*- 
"""
@Time : 2022-09-20 12:21 
@Author : zhimiao.chh 
@Desc : 
"""

import torch
from functools import partial
from xdpx.utils import register,numpy_seed
from xdpx.options import Argument
import random
from . import register
import numpy as np
import math


@register('chat_mix_common')
class ChatMixCommonSampler(torch.utils.data.Sampler):

    @classmethod
    def register(cls, options):
        options.register(
            Argument('supervise_ratio', default=0.5)
        )

    def __init__(self, args, dataset, batch_size, shuffle, num_replicas, rank, update_freq=1, sort_key=None, seed=None):

        self.total_size = int(math.ceil(len(dataset) * 1.0 / num_replicas)) * num_replicas
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = self.total_size // self.num_replicas
        self.seed = seed

        def generate_idx(id_list,count):
            length = len(id_list)
            if length==0:
                return id_list
            divisor = count//length
            remainder = count%length
            result_list = id_list*divisor
            result_list += self.shuffle_with_seed(id_list)[:remainder]
            return result_list


        self.dataset_list = [i for i in range(len(dataset))]
        # when in train mode, do data sample
        if self.shuffle:
            sup_count = int(len(dataset)*args.supervise_ratio)
            unsup_count = len(dataset)-sup_count
            sup_id = []
            unsup_id = []
            for i,sample in enumerate(dataset):
                if len(sample['response'])>0:
                    sup_id.append(i)
                else:
                    unsup_id.append(i)
            sup_id = generate_idx(sup_id,sup_count)
            unsup_id = generate_idx(unsup_id,unsup_count)
            print(f"| sup_count:{len(sup_id)}  unsup_count:{len(unsup_id)}")
            dataset_list = sup_id+unsup_id
            self.dataset_list = self.shuffle_with_seed(dataset_list)

        assert len(self.dataset_list)==len(dataset)

    def shuffle_with_seed(self,batches):
        # deterministically shuffle based on epoch
        seed = (self.seed, self.epoch) if self.seed is not None else (None,)
        with numpy_seed(*seed):
            np.random.shuffle(batches)
        return batches

    def __iter__(self):
        indices = self.dataset_list

        # add extra samples to make it evenly divisible (in validation extra samples will be trimmed later)
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

@register('fid_chat_mix_common')
class FidChatMixCommonSampler(torch.utils.data.Sampler):

    @classmethod
    def register(cls, options):
        options.register(
            Argument('sup_count', type=int, required=True),
            Argument('unsup_count', type=int, required=True)
        )

    def __init__(self, args, dataset, batch_size, shuffle, num_replicas, rank, update_freq=1, sort_key=None, seed=None):

        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        # deterministically shuffle based on epoch
        self.seed = (seed, self.epoch) if self.seed is not None else (None,)
        self.batch_size = batch_size

        def generate_idx(id_list, count):
            length = len(id_list)
            if length == 0:
                return id_list
            divisor = count // length
            remainder = count % length
            result_list = id_list * divisor
            result_list += self.shuffle_with_seed(id_list)[:remainder]
            return result_list

        self.dataset_list = [i for i in range(len(dataset))]
        self.total_size = int(math.ceil(len(self.dataset_list) * 1.0 / num_replicas)) * num_replicas
        self.num_samples = self.total_size // self.num_replicas

        # when in train mode, do data sample
        if self.shuffle:
            sup_count = args.sup_count
            unsup_count = args.unsup_count
            sup_id = []
            unsup_id = []
            for i, sample in enumerate(dataset):
                if len(sample['response']) > 0:
                    sup_id.append(i)
                else:
                    unsup_id.append(i)
            sup_id = generate_idx(sup_id, sup_count)
            unsup_id = generate_idx(unsup_id, unsup_count)
            print(f"| sup_count:{len(sup_id)}  unsup_count:{len(unsup_id)}")

            self.sup_id_list = self.shuffle_with_seed(sup_id)
            self.unsup_id_list = self.shuffle_with_seed(unsup_id)

            total_id_num = len(self.sup_id_list)+len(self.unsup_id_list)
            # support distributed trainning and batch bucket according to sup or unsup
            self.total_size = math.ceil(total_id_num / (num_replicas*self.batch_size)) * (num_replicas*self.batch_size)
            self.num_samples = self.total_size // self.num_replicas

    def shuffle_with_seed(self, batches):
        # deterministically shuffle based on epoch
        seed = (self.seed, self.epoch) if self.seed is not None else (None,)
        with numpy_seed(*seed):
            np.random.shuffle(batches)
        return batches

    def __iter__(self):
        # validation
        if not self.shuffle:
            indices = self.dataset_list

            # add extra samples to make it evenly divisible (in validation extra samples will be trimmed later)
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size
            # subsample
            indices = indices[self.rank: self.total_size: self.num_replicas]
            assert len(indices) == self.num_samples

            return iter(indices)
        # training
        else:
            # unsup id: do downsampling
            unsup_id_size = len(self.unsup_id_list)//(self.num_replicas*self.batch_size)*(self.num_replicas*self.batch_size)
            unsup_id_list = self.unsup_id_list[:unsup_id_size]
            # sup id: do upsampling
            sup_id_size = self.total_size-unsup_id_size
            sup_id_list = self.sup_id_list+self.sup_id_list[:(sup_id_size-len(self.sup_id_list))]
            # subsample
            unsup_indices = unsup_id_list[self.rank: : self.num_replicas]
            sup_indices = sup_id_list[self.rank: : self.num_replicas]

            assert len(unsup_indices)//self.batch_size==len(unsup_indices)/self.batch_size
            assert len(sup_indices)//self.batch_size==len(sup_indices)/self.batch_size

            indices_bucket = []
            for i in range(0,len(unsup_indices),self.batch_size):
                indices_bucket.append(unsup_indices[i:i+self.batch_size])
            for i in range(0,len(sup_indices),self.batch_size):
                indices_bucket.append(sup_indices[i:i+self.batch_size])
            self.shuffle_with_seed(indices_bucket)

            indices = []
            for bucket in indices_bucket:
                indices.extend(bucket)

            assert len(indices) == self.num_samples

            return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


if __name__ == '__main__':
    ""
    seed = (1, 2)


    def shuffle_with_seed(batches):
        with numpy_seed(*seed):
            np.random.shuffle(batches)
        return batches

    a = [[1,2],[3,4],[5,6],[7,8]]
    shuffle_with_seed(a)
    print(a)