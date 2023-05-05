import math
import torch.nn as nn
from ..activations import activations, activation_coeffs
from .pkm import HashingMemory
from .lopkm import LocallyOptimizedHashingMemory


class LinearProjection(nn.Module):
    def __init__(self, in_features, out_features, activation='linear', bias=True):
        super().__init__()
        self.activation = activations[activation]
        activation_coeff = activation_coeffs[activation]
        linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.normal_(linear.weight, std=math.sqrt(1. / in_features) * activation_coeff)
        if bias:
            nn.init.zeros_(linear.bias)
        self.model = nn.utils.weight_norm(linear)

    def forward(self, x):
        return self.activation(self.model(x))
