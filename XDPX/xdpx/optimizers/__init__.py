import os
import importlib
from functools import partial
from xdpx.utils import register
from .optimizer import Optimizer
from .bmuf import BMUFOptimizer
from .fp16 import FP16Optimizer, MemoryEfficientFP16Optimizer

optimizers = {}
register = partial(register, registry=optimizers)


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('.' + module_name, __name__)
