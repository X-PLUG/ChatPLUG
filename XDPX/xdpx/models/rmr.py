import math
import torch.nn as nn
from xdpx.options import Argument
from xdpx.modules import MultiheadAttention
from xdpx.modules.activations import GELU
from .bert import BertForLanguageModeling
from . import register


@register('rmr_bert')
class RMRBert(BertForLanguageModeling):
    @staticmethod
    def register(options):
        options.register(
            Argument('top_hidden_size', default=256),
            Argument('top_heads', default=4),
            domain='Bert/RMR'
        )

    def __init__(self, args):
        super().__init__(args)
        self.top_tfm = MultiheadAttention(
            embed_dim=self.args.hidden_size,
            num_heads=self.args.top_heads,
            dropout=self.args.attention_probs_dropout_prob,
            # kdim=self.args.top_hidden_size,
            # vdim=self.args.top_hidden_size,
        )
        self.dropout = nn.Dropout(p=self.args.hidden_dropout_prob)
        self.top_ffn = nn.Sequential(
            nn.Linear(self.args.hidden_size, self.args.top_hidden_size),
            GELU(),
            nn.Dropout(p=self.args.hidden_dropout_prob),
            nn.Linear(self.args.top_hidden_size, 1)
        )

    def top_forward(self, x, mask):
        x = x.transpose(0, 1)
        x2, _ = self.top_tfm(x, x, x, key_padding_mask=~mask)
        x = (x + self.dropout(x2)) * math.sqrt(0.5)
        x = x.transpose(0, 1)
        return self.top_ffn(x)

    def forward(self, *args, masked_tokens=None, **kwargs):
        seq_features, agg_features = self.bert_forward(*args, **kwargs)[:2]
        seq_outputs = None
        if masked_tokens is not None:
            seq_outputs = self.cls['predictions'](seq_features, masked_tokens=masked_tokens)
        return seq_outputs, agg_features
