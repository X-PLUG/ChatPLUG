import math
import torch
import torch.distributed as dist
from typing import List

from . import register
from .optimizer import Optimizer
from .fused_adam import get_fused_adam_class
from xdpx.options import Argument


@register('adam')
class Adam(Optimizer):
    """
    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """
    @staticmethod
    def register(options):
        options.register(
            Argument('adam_betas', default=[0.9, 0.999], type=List[float], validate=[
                lambda value: len(value) == 2,
                lambda value: all(0 < x < 1 for x in value),
            ]),
            Argument('adam_eps', default=1e-6, doc='epsilon for Adam optimizer'),
            Argument('weight_decay', default=0.),
            domain='adam_optimizer'
        )
        options.set_default('learning_rate', 1e-3)

    def __init__(self, args, params):
        super().__init__(args)
        fused_adam_cls = get_fused_adam_class()
        use_fused_adam = (
            fused_adam_cls is not None
            and torch.cuda.is_available()
        )
        if use_fused_adam:
            print('| Apex detected. Using FusedAdam.')
            print('| {}'.format(fused_adam_cls.__name__))
            self._optimizer = fused_adam_cls(params, **self.optimizer_config)
        else:
            self._optimizer = AdamW(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        return {
            'lr': self.args.learning_rate,
            'betas': self.args.adam_betas,
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
        }
    
    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)


@register('sparse_adam')
class SparseAdam(Optimizer):
    @staticmethod
    def register(options):
        options.register(
            Argument('adam_betas', default=[0.9, 0.999], type=List[float], validate=[
                lambda value: len(value) == 2,
                lambda value: all(0 < x < 1 for x in value),
            ]),
            Argument('adam_eps', default=1e-6, doc='epsilon for Adam optimizer'),
            domain='adam_optimizer'
        )
        options.set_default('learning_rate', 1e-3)
        options.add_global_constraint(lambda args: not hasattr(args, 'bmuf') or not args.bmuf)
        # apex for sparse tensors not implemented by now.
        options.add_global_constraint(lambda args: not hasattr(args, 'fp16') or not args.fp16)
        # distributed training for sparse tensors not implemented by now.
        options.add_global_constraint(lambda args: args.distributed_world_size == 1)

    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = torch.optim.SparseAdam(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        return {
            'lr': self.args.learning_rate,
            'betas': self.args.adam_betas,
            'eps': self.args.adam_eps,
        }


class AdamW(torch.optim.Optimizer):
    r"""Reference:  fairseq.optim.adam.Adam
    Implements Adam algorithm.

    This implementation is modified from torch.opticlip_grad_normm.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')


                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss