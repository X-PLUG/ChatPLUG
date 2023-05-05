import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from ..projections import LinearProjection
from xdpx.utils.versions import torch_ge_150
if torch_ge_150():
    from torch.nn import MultiheadAttention
else:
    from .multihead_attention import MultiheadAttention


class Alignment(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(hidden_size)))

    def _attention(self, a, b):
        return torch.matmul(a, b.transpose(1, 2)) * self.temperature

    def forward(self, a, b, mask_a, mask_b):
        attn = self._attention(a, b)
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float())
        if version.parse(torch.__version__) < version.parse('1.2'):
            mask = mask.byte()
        else:
            mask = mask.bool()
        attn.masked_fill_(~mask, -1e4)
        attn_a = F.softmax(attn, dim=1)
        attn_b = F.softmax(attn, dim=2)
        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        feature_a = torch.matmul(attn_b, b)
        return feature_a, feature_b


class MappedAlignment(Alignment):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__(hidden_size)
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            LinearProjection(input_size, hidden_size, activation='gelu'),
        )

    def _attention(self, a, b):
        a = self.projection(a)
        b = self.projection(b)
        return super()._attention(a, b)
