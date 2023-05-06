import math
from . import register, LRScheduler
from xdpx.options import Argument


@register('exponential_decay')
class ExponentialDecaySchedule(LRScheduler):
    @staticmethod
    def register(options):
        options.register(
            Argument('warmup_steps', default=0, doc='warmup the learning rate linearly for the first N updates'),
            Argument('decay_rate', default=0.95),
            Argument('decay_steps', default=1000),
            Argument('div_factor', default=25., doc='Determines the initial learning rate via initial_lr = max_lr / div_factor',
                # initial lr cannot be zero due to the limitation of "Optimizer.set_lr" implementation
                validate=lambda value: 1. < value < math.inf
            ), 
        )

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        self.lr = optimizer.get_lr()
        self.warmup_steps = args.warmup_steps
        self.decay_rate = args.decay_rate
        self.decay_steps = args.decay_steps
        self.inital_ratio = 1. / args.div_factor

    def step_update(self, step):
        """Update the learning rate after each update."""
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            ratio = self.inital_ratio + (1. - self.inital_ratio) * float(step) / self.warmup_steps
        else:
            ratio = self.decay_rate ** math.floor((step - self.warmup_steps) / self.decay_steps)
        lr = self.lr * ratio
        self.optimizer.set_lr(lr)
        return self.optimizer.get_lr()
