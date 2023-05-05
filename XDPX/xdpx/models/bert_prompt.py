import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
from functools import lru_cache
from . import register, Model
from xdpx.processors import processors
from xdpx.processors.bert import BertProcessor
from xdpx.modules.activations import activations
from xdpx.modules.layer_norm import LayerNorm
from xdpx.utils import io, should_save_meta
from xdpx.options import Argument, Options
from .bert import BertForLanguageModeling, LMHead


@register('bert_prompt')
class BertPromptModel(BertForLanguageModeling):
    @staticmethod
    def register(options):
        options.add_global_constraint(
            lambda args: args.layer_lr_decay == 1 or args.keep_emb_lr
        )

    def __init__(self, args):
        super().__init__(args)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.cls = nn.ModuleDict()
        self.cls['predictions'] = LMHead(
            hidden_size=self.args.hidden_size,
            vocab_size=self.args.vocab_size,
            embedding_size=getattr(self.args, 'embedding_size', self.args.hidden_size),
            activation_fn=self.args.hidden_act,
            weight=self.get_embeddings().weight,
        )

    def forward(self, *args, masked_tokens=None, **kwargs):
        # use sequence output
        features = self.bert_forward(*args, **kwargs)[0]
        outputs = self.cls['predictions'](features, masked_tokens=masked_tokens)
        return outputs

    @property
    def name_map(self):
        name_map = super().name_map.copy()
        name_map.update({
            'heads.lm.bias': 'cls.predictions.bias',
            'heads.lm.weight': 'cls.predictions.decoder.weight',
            'heads.lm.dense.weight': 'cls.predictions.transform.dense.weight',
            'heads.lm.dense.bias': 'cls.predictions.transform.dense.bias',
            'heads.lm.layer_norm.weight': 'cls.predictions.transform.LayerNorm.weight',
            'heads.lm.layer_norm.bias': 'cls.predictions.transform.LayerNorm.bias',
        })
        return name_map

