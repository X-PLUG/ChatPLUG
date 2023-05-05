from functools import partial
from xdpx.options import Argument, parse_relative
from . import register, optimizers
from .optimizer import Optimizer


@register('pair')
class PairOptimizer(Optimizer):
    @staticmethod
    def register(options):
        with options.with_prefix('major_'):
            options.register(
                Argument('optimizer', default='adam', validate=lambda value: value in optimizers and value != 'pair',
                         register=lambda value: optimizers[value].register),
                domain='major_opt',
            )
        with options.with_prefix('second_'):
            options.register(
                Argument('optimizer', default='adam', validate=lambda value: value in optimizers and value != 'pair',
                         register=lambda value: optimizers[value].register),
                domain='second_opt',
            )
        options.register(
            Argument('second_learning_rate', type=float, required=True,
                     post_process=partial(parse_relative, reverse=False)),
            Argument('follow_lr_schedule', default=False),
            domain='second_opt',
        )
        options.add_global_constraint(lambda args: not hasattr(args, 'use_bmuf') or not args.use_bmuf)

    def __init__(self, args, params):
        super().__init__(args)
        if len(params) != 2:
            raise ValueError('`model.trainable_parameters()` should return a pair of param groups for pair optimizer.')
        major_params, second_params = params
        self._optimizer = optimizers[args.major_optimizer](args.strip_prefix('major_'), major_params)._optimizer
        self._2nd_optimizer = optimizers[args.second_optimizer](args.strip_prefix('second_'), second_params)._optimizer

    def state_dict(self):
        return {
            'major': self._optimizer.state_dict(),
            'second': self._2nd_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        super().load_state_dict(state_dict['major'], optimizer_overrides)
        self._2nd_optimizer.load_state_dict(state_dict['second'])

    @property
    def second_params(self):
        for param_group in self._2nd_optimizer.param_groups:
            for p in param_group['params']:
                yield p

    def backward(self, loss, **kwargs):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves."""
        if self.args.fp16 and self.args.fp16_backend == 'apex':
            from apex import amp
            with amp.scale_loss(loss, [self._optimizer, self._2nd_optimizer]) as scaled_loss:
                scaled_loss.backward(**kwargs)
        else:
            loss.backward(**kwargs)

    def multiply_grads(self, c):
        for params in (self.params, self.second_params):
            for p in params:
                if p.grad is not None:
                    p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm):
        # skip clipping/counting 2nd optimizer for now
        return super().clip_grad_norm(max_norm)

    def step(self, closure=None):
        """Performs a single optimization step."""
        self._optimizer.step(closure)
        self._2nd_optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self._optimizer.zero_grad()
        self._2nd_optimizer.zero_grad()

    def set_lr(self, lr):
        """Set the learning rate."""
        super().set_lr(lr)
        if self.args.follow_lr_schedule:
            base_lr = self._2nd_optimizer.param_groups[0]['lr']
            for param_group in self._2nd_optimizer.param_groups:
                param_group['lr'] *= lr / base_lr

    def set_momentum(self, m):
        super().set_momentum(m)
        if self.args.follow_lr_schedule:
            base_m = self._2nd_optimizer.param_groups[0]['betas'][0]
            for param_group in self._2nd_optimizer.param_groups:
                beta1, beta2 = param_group['betas']
                param_group['betas'] = (beta1 * m / base_m, beta2)
