import math
from . import register, LRScheduler
from typing import Union
from functools import partial
from xdpx.options import Argument, parse_relative
from xdpx.utils import get_total_steps


@register('one_cycle')
class OneCycleSchedule(LRScheduler):
    @staticmethod
    def register(options):
        options.register(
            Argument('warmup_steps', default=0, type=Union[int, float], post_process=parse_relative,
                     doc='warmup the learning rate linearly for the first N updates, can be a ratio (in float)'),
            Argument('anneal_strategy', default='linear',
                     validate=lambda value: value in ('linear', 'cosine', 'polynomial'),
                     children={
                        lambda value: value == 'polynomial': [
                            Argument('anneal_p', default=1.)
                        ]},
                     ),
            # initial lr cannot be zero due to the limitation of "Optimizer.set_lr" implementation
            Argument('div_factor', default=600., validate=lambda value: 1. < value < math.inf,
                     doc='Determines the initial learning rate via initial_lr = max_lr / div_factor'),
            Argument('cycle_momentum', default=False, children=[
                Argument('base_momentum', default=0.85),
            ]),
        )

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        self.lr = optimizer.get_lr()
        if args.cycle_momentum:
            self.m = optimizer.get_momentum()
            self.base_m = args.base_momentum
        self.total_steps = get_total_steps(args, runtime=True)
        self.warmup_steps = args.warmup_steps if type(args.warmup_steps) is int else int(self.total_steps * args.warmup_steps)
        self.anneal_steps = max(1, self.total_steps - self.warmup_steps)
        self.inital_lr = self.lr / args.div_factor
        if args.anneal_strategy == 'linear':
            self.anneal_func = self._annealing_linear
        elif args.anneal_strategy == 'cosine':
            self.anneal_func = self._annealing_cos
        elif args.anneal_strategy == 'polynomial':
            self.anneal_func = partial(self._annealing_poly, p=args.anneal_p)
        else:
            raise NotImplementedError

    @staticmethod
    def _annealing_cos(start, end, pct):
        """Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    @staticmethod
    def _annealing_linear(start, end, pct):
        """Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        return (end - start) * pct + start
    
    @staticmethod
    def _annealing_poly(start, end, pct, p):
        """Anneal from `start` to `end` as pct goes from 0.0 to 1.0 with polynomial p."""
        return end + (start - end) * (1 - pct) ** p

    def step_update(self, step):
        """Update the learning rate after each update."""
        if step > self.total_steps:
            print(f'| Warning: Step number {step} goes beyond the maximum number of steps in OneCycleSchedule.')
            return self.optimizer.get_lr()
        assert step >= 0
        if step < self.warmup_steps:
            lr = self.anneal_func(self.inital_lr, self.lr, float(step) / self.warmup_steps)
        else:
            lr = self.anneal_func(self.lr, self.inital_lr, float(step - self.warmup_steps) / self.anneal_steps)

        self.optimizer.set_lr(lr)
        if self.args.cycle_momentum:
            # momentum is cycled inversely to learning rate
            if step < self.warmup_steps:
                m = self.anneal_func(self.m, self.base_m, float(step) / self.warmup_steps)
            else:
                m = self.anneal_func(self.base_m, self.m, float(step - self.warmup_steps) / self.anneal_steps)
            self.optimizer.set_momentum(m)
        return self.optimizer.get_lr()
