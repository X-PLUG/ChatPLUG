import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as f
from ..activations import activations, activation_coeffs
from ..projections import LinearProjection


class ConvLayer(nn.Module):
    def __init__(self, dropout, hidden_size, kernel_sizes, enc_layers, input_size, activation, residual=False):
        super().__init__()
        self.dropout = dropout
        self.activation = activation
        self.encoders = nn.ModuleList([Conv1d(
                in_channels=input_size if i == 0 and not residual else hidden_size,
                out_channels=hidden_size, activation=activation,
                kernel_sizes=kernel_sizes) for i in range(enc_layers)])
        self.residual = residual
        if residual and input_size != hidden_size:
            self.proj = LinearProjection(input_size, hidden_size)

    def forward(self, x, mask):
        if hasattr(self, 'proj'):
            x = self.proj(x)
        x = x.transpose(1, 2)  # B x C x L
        mask = mask.transpose(1, 2)
        x_in = x
        for i, encoder in enumerate(self.encoders):
            if self.residual and i > 0:
                x = (x + x_in) / math.sqrt(2.)
            x_in = x
            x.masked_fill_(~mask, 0.)
            if i > 0:
                x = f.dropout(x, self.dropout, self.training)
            x = encoder(x)
        x = f.dropout(x, self.dropout, self.training)
        return x.transpose(1, 2)  # B x L x C


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, activation, kernel_sizes: List[int]):
        super().__init__()
        self.activation = activations[activation]
        activation_coeff = activation_coeffs[activation]
        out_channels = [out_channels // len(kernel_sizes) + (i < out_channels % len(kernel_sizes)) for i in range(len(kernel_sizes))]
        convs = []
        for kernel_size, out_channels in zip(kernel_sizes, out_channels):
            conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=int(math.ceil((kernel_size - 1) / 2)))
            nn.init.normal_(conv.weight, std=math.sqrt(1. / (in_channels * kernel_size) * activation_coeff))
            nn.init.zeros_(conv.bias)
            convs.append(nn.utils.weight_norm(conv))
        self.model = nn.ModuleList(convs)

    def forward(self, x):
        """x is [B x C x L]"""
        outs = []
        for encoder in self.model:
            out = encoder(x)
            if encoder.kernel_size[0] % 2 == 0:
                # remove the extra token introduced by an even kernel size
                out = out[:, :, 1:]
            outs.append(self.activation(out))

        return torch.cat(outs, dim=1)
