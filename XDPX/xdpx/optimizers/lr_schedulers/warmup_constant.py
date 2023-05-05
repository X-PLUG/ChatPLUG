import math
from . import register, LRScheduler
from xdpx.options import Argument


@register('warmup_constant')
class WarmupConstantSchedule(LRScheduler):
    @staticmethod
    def register(options):
        options.register(
            Argument('warmup_steps', default=0, type=int,
                doc='warmup the learning rate linearly for the first N updates'),
            Argument('div_factor', default=25., doc='Determines the initial learning rate via initial_lr = max_lr / div_factor',
                # initial lr cannot be zero due to the limitation of "Optimizer.set_lr" implementation
                validate=lambda value: 1. < value < math.inf
            ),  
        )

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        self.lr = optimizer.get_lr()
        self.warmup_steps = args.warmup_steps
        self.inital_ratio = 1. / args.div_factor

    def step_update(self, step):
        "Update the learning rate after each update."
        if step < self.warmup_steps:
            ratio = self.inital_ratio + (1. - self.inital_ratio) * float(step) / self.warmup_steps
        else:
            ratio = 1.0
        lr = self.lr * ratio
        self.optimizer.set_lr(lr)
        return self.optimizer.get_lr()
