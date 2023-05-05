import torch
import torch.nn as nn
import torch.nn.functional as F
from ..projections import LinearProjection


class MaxPooling(nn.Module):
    def forward(self, x, mask, dim=1):
        # the following version will raise RuntimeError in pytorch 1.0 for bert_classificaiton_concat although no padding is pooled: 
        #   "one of the variables needed for gradient computation has been modified by an inplace operation"
        # return x.masked_fill_(~mask.byte(), -1e4).max(dim=1)[0]
        x = x * mask.float() + -1e4 * (1. - mask.float())
        return x.max(dim=dim)[0]


class AttnPooling(nn.Module): 
    def __init__(self, input_size, hidden_size=None, output_size=None):
        super().__init__()
        self.input_proj = LinearProjection(input_size, 1, bias=False) if hidden_size is None else \
            nn.Sequential(
                LinearProjection(input_size, hidden_size),
                nn.Tanh(),
                LinearProjection(hidden_size, 1, bias=False)
            )
        self.output_proj = LinearProjection(input_size, output_size) if output_size else lambda x: x

    def forward(self, x, mask):
        score = self.input_proj(x)
        score = score * mask.float() + -1e4 * (1. - mask.float())
        score = F.softmax(score, dim=1)
        features = self.output_proj(x)
        return torch.matmul(score.transpose(1, 2), features).squeeze(1)


class KeyAttnPooling(nn.Module): 
    def __init__(self, input_size, key_size=None, output_size=None):
        super().__init__()
        self.input_proj = LinearProjection(key_size, input_size) if key_size else lambda x: x
        self.output_proj = LinearProjection(input_size, output_size) if output_size else lambda x: x

    def forward(self, x, mask, key):
        key = self.input_proj(key)
        score = torch.bmm(x, key.unsqueeze(2))
        score = score * mask.float() + -1e4 * (1. - mask.float())
        score = F.softmax(score, dim=1)
        features = self.output_proj(x)
        return torch.matmul(score.transpose(1, 2), features).squeeze(1)
