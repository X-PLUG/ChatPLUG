import os
import importlib
from functools import partial
from xdpx.utils import register
from .. import Optimizer

lr_schedulers = {}
register = partial(register, registry=lr_schedulers)


@register('constant')
class LRScheduler:
    @staticmethod
    def register(options):
        pass

    def __init__(self, args, optimizer):
        super().__init__()
        if not isinstance(optimizer, Optimizer):
            raise ValueError('optimizer must be an instance of Optimizer')
        self.args = args
        self.optimizer = optimizer

    def state_dict(self):
        """Assume the scheduler is stateless (only depends on args)."""
        return {}

    def load_state_dict(self, state_dict):...

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.optimizer.get_lr()


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('.' + module_name, __name__)
