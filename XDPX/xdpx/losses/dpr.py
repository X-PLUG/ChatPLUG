import torch
import torch.nn as nn
from . import register
from .bert import BertLoss
from xdpx.options import Argument
from xdpx.tasks import Task, register as register_task
import json
from itertools import chain
from sklearn.metrics import accuracy_score
import numpy as np
from xdpx.loaders import loaders
from xdpx.utils import io, move_to_cuda
import traceback
from xdpx.utils.distributed_utils import is_master, should_barrier
import os
import numpy as np
import logging
from scipy.stats import spearmanr
import math
from typing import List
import torch.distributed as dist
from xdpx.utils import io, cache_file, get_train_subsets
from xdpx.utils.io_utils import tqdm, CallbackIOWrapper
import random
import torch.nn.functional as F

@register('bert_dpr')
class BertDPRLoss(BertLoss):
    """
    Contrastive Learning
    """

    @staticmethod
    def register(options):
        BertLoss.register(options)
        options.register(
            Argument('temperature', default=0.05),
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert reduce is True
        batch_size = int(sample['net_input']['question_input_ids'].size(0))
        z1, z2 = model(**sample['net_input'])
        # Gather all embeddings if using distributed training
        if dist.is_initialized() and model.training:
            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        logging_output = {}

        # infonce loss
        cos_sim = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1, eps=1e-4) / self.args.temperature
        max_p, indices = torch.max(cos_sim, dim=1)
        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)

        logging_output.update({
            'loss': loss.item(),
            'target': labels.tolist(),
            'pred': indices.tolist()
        })

        # loss and other stats are already averaged, so 1 is used as the sample_size
        return loss, batch_size, logging_output

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        target = list(chain.from_iterable(log['target'] for log in logging_outputs))[:sample_size]
        pred = list(chain.from_iterable(log['pred'] for log in logging_outputs))[:sample_size]

        agg_output = dict({
            'sample_size': sample_size
        })

        agg_output['loss'] = sum(log['loss'] for log in logging_outputs) / sample_size

        accuracy = accuracy_score(target, pred)
        agg_output['acc'] = accuracy

        if 'ntokens' in logging_outputs[0]:
            ntokens = sum(log['ntokens'] for log in logging_outputs)
            agg_output['ntokens'] = ntokens
        return agg_output

    def inference(self, model, sample):
        z1, z2 = model(sample['orig_queston_input_ids'], sample['orig_passage_input_ids'])
        return z1, z2
