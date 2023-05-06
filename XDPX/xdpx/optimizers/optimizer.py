import math
import torch


class Optimizer:
    @staticmethod
    def register(options):
        pass

    def __init__(self, args):
        self.args = args
        self._optimizer = None

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, '_optimizer'):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of torch.optim.Optimizer')
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        raise NotImplementedError
    
    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                yield p
    
    def __getstate__(self):
        return self._optimizer.__getstate__()
    
    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        """Set the learning rate."""
        base_lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= lr / base_lr  # if the initial lrs differ, they scale accordingly

    def get_momentum(self):
        try:
            return self.optimizer.param_groups[0]['betas'][0]
        except KeyError:
            raise ValueError(f'Optimizer {self.__class__.__name__} does not support cycle_momentum.')

    def set_momentum(self, m):
        base_m = self.get_momentum()
        for param_group in self.optimizer.param_groups:
            beta1, beta2 = param_group['betas']
            param_group['betas'] = (beta1 * m / base_m, beta2)

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            # override learning rate, momentum, etc. with latest values
            for group in self.optimizer.param_groups:
                group.update(optimizer_overrides)

    def backward(self, loss, **kwargs):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves."""
        if self.args.fp16 and self.args.fp16_backend == 'apex':
            from apex import amp
            with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward(**kwargs)
        else:
            loss.backward(**kwargs)

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm. This behaves differently from fairseq.utils.clip_grad_norm_"""
        if max_norm > 0:
            return torch.nn.utils.clip_grad_norm_(self.params, max_norm)
        return math.sqrt(sum(p.grad.data.norm()**2 for p in self.params if p.grad is not None))

    def step(self, closure=None):
        """Performs a single optimization step."""
        self.optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self.optimizer.zero_grad()
    
    def average_params(self):
        pass

    @property
    def supports_memory_efficient_fp16(self):
        if hasattr(self.optimizer, 'supports_memory_efficient_fp16'):
            return self.optimizer.supports_memory_efficient_fp16
        return False
