import os
import math
import importlib

import torch
import torch.nn.functional as F


activations = {
    'relu': F.relu,
    'tanh': torch.tanh,
    'linear': lambda x: x,
}

activation_coeffs = {
    'relu': math.sqrt(2),
    'tanh': 5 / 3,
    'linear': 1.,
}


def register(name, value):
    def decorator(fn):
        if name in activations or name in activation_coeffs:
            raise LookupError(f"module {name} already registered.")
        assert isinstance(value, float) or isinstance(value, int)
        activations[name] = fn
        activation_coeffs[name] = value
        return fn
    return lambda fn: decorator(fn)


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('.' + module_name, __name__)

from .gelu import GELU
