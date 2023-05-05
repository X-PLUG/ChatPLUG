# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Difference between fairseq implementation:
- support multiple param groups
- compatible with pair optimizer wrapper
"""

import warnings
from itertools import chain
import torch
from . import Optimizer


class DynamicLossScaler(object):

    def __init__(
        self, init_scale=2.**15, scale_factor=2., scale_window=2000,
        tolerance=0.05, threshold=None,
    ):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self.threshold = threshold
        self._iter = 0
        self._last_overflow_iter = -1
        self._last_rescale_iter = -1
        self._overflows_since_rescale = 0

    def update_scale(self, overflow):
        iter_since_rescale = self._iter - self._last_rescale_iter
        if overflow:
            self._last_overflow_iter = self._iter
            self._overflows_since_rescale += 1
            pct_overflow = self._overflows_since_rescale / float(iter_since_rescale)
            if pct_overflow >= self.tolerance:
                self._decrease_loss_scale()
                self._last_rescale_iter = self._iter
                self._overflows_since_rescale = 0
        elif (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
            self._last_rescale_iter = self._iter
        self._iter += 1

    def _decrease_loss_scale(self):
        self.loss_scale /= self.scale_factor
        if self.threshold is not None:
            self.loss_scale = max(self.loss_scale, self.threshold)

    @staticmethod
    def has_overflow(grad_norm):
        # detect inf and nan
        if grad_norm == float('inf') or grad_norm != grad_norm:
            return True
        return False


class _FP16OptimizerMixin(object):

    def __init__(self, *args, **kwargs):
        # forward __init__ call to the next class in mro(method resolution order)
        super().__init__(*args, **kwargs)

    @property
    def has_flat_params(self):
        return torch.is_tensor(self.fp32_params)

    @classmethod
    def build_fp32_params(cls, params):
        # create FP32 copy of parameters and grads
        if torch.is_tensor(params[0]):
            fp32_params = []
            for p in params:
                p32 = torch.nn.Parameter(p.data.float())
                p32.grad = torch.zeros_like(p32.data)
                fp32_params.append(p32)
        else:
            # param group
            fp32_params = []
            for param_group in params:
                fp32_param_group = param_group.copy()
                fp32_param_group['params'] = []
                for p in param_group['params']:
                    p32 = torch.nn.Parameter(p.data.float())
                    p32.grad = torch.zeros_like(p32.data)
                    fp32_param_group['params'].append(p32)
                fp32_params.append(fp32_param_group)
        return fp32_params

    @classmethod
    def combine_param_groups(cls, params):
        if torch.is_tensor(params[0]):
            return params
        return [p for param_group in params for p in param_group['params']]

    def state_dict(self):
        """Return the optimizer's state dict."""
        state_dict = self.fp32_optimizer.state_dict()
        if self.scaler is not None:
            state_dict['loss_scale'] = self.scaler.loss_scale
        return state_dict

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        if 'loss_scale' in state_dict and self.scaler is not None:
            self.scaler.loss_scale = state_dict.pop('loss_scale')
        self.fp32_optimizer.load_state_dict(state_dict, optimizer_overrides)

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves.

        Compared to :func:`fairseq.optim.FairseqOptimizer.backward`, this
        function additionally dynamically scales the loss to avoid gradient
        underflow.
        """
        if self.scaler is not None:
            loss = loss * self.scaler.loss_scale
        loss.backward()
        self._needs_sync = True

    def _sync_fp16_grads_to_fp32(self, multiply_grads=1.):
        if self._needs_sync or multiply_grads != 1:
            if self.scaler is not None:
                # correct for dynamic loss scaler
                multiply_grads /= self.scaler.loss_scale

            # copy FP16 grads to FP32
            if self.has_flat_params:
                offset = 0
                for p in self.fp16_params:
                    if not p.requires_grad:
                        continue
                    grad_data = p.grad.data if p.grad is not None else p.data.new_zeros(p.data.shape)
                    numel = grad_data.numel()
                    self.fp32_params.grad.data[offset:offset+numel].copy_(grad_data.view(-1))
                    offset += numel
                self.fp32_params.grad.data.mul_(multiply_grads)
            else:
                for p, p32 in zip(self.fp16_params, self.fp32_params):
                    if not p.requires_grad:
                        continue
                    if p.grad is not None:
                        p32.grad.data.copy_(p.grad.data)
                        p32.grad.data.mul_(multiply_grads)
                    else:
                        p32.grad = torch.zeros_like(p.data, dtype=torch.float)

            self._needs_sync = False

    def multiply_grads(self, c):
        """Multiplies grads by a constant ``c``."""
        if self._needs_sync:
            self._sync_fp16_grads_to_fp32(c)
        elif self.has_flat_params:
            self.fp32_params.grad.data.mul_(c)
        else:
            for p32 in self.fp32_params:
                p32.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm and updates dynamic loss scaler."""
        self._sync_fp16_grads_to_fp32()
        grad_norm = clip_grad_norm_(self.fp32_params, max_norm, aggregate_norm_fn)

        # detect overflow and adjust loss scale
        if self.scaler is not None:
            overflow = DynamicLossScaler.has_overflow(grad_norm)
            prev_scale = self.scaler.loss_scale
            self.scaler.update_scale(overflow)
            if overflow:
                if self.scaler.loss_scale <= self.min_loss_scale:
                    # Use FloatingPointError as an uncommon error that parent
                    # functions can safely catch to stop training.
                    self.scaler.loss_scale = prev_scale
                    raise FloatingPointError((
                        'Minimum loss scale reached ({}). Your loss is probably exploding'
                        'or there\'s NaN in half precision. See the report of NaNDetector.'
                    ).format(self.min_loss_scale))
                raise OverflowError('setting loss scale to: ' + str(self.scaler.loss_scale))

        return grad_norm

    def step(self, closure=None):
        """Performs a single optimization step."""
        self._sync_fp16_grads_to_fp32()
        self.fp32_optimizer.step(closure)

        # copy FP32 params back into FP16 model
        if self.has_flat_params:
            offset = 0
            for p in self.fp16_params:
                if not p.requires_grad:
                    continue
                numel = p.data.numel()
                p.data.copy_(self.fp32_params.data[offset:offset+numel].view_as(p.data))
                offset += numel
        else:
            for p, p32 in zip(self.fp16_params, self.fp32_params):
                if not p.requires_grad:
                    continue
                p.data.copy_(p32.data)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.fp16_params:
            p.grad = None
        if self.has_flat_params:
            self.fp32_params.grad.zero_()
        else:
            for p32 in self.fp32_params:
                p32.grad.zero_()
        self._needs_sync = False


class FP16Optimizer(_FP16OptimizerMixin, Optimizer):
    """
    Wrap an *optimizer* to support FP16 (mixed precision) training.
    """

    def __init__(self, args, params, fp32_optimizer, fp32_params):
        super().__init__(args)
        self.fp16_params = params
        self.fp32_optimizer = fp32_optimizer
        self.fp32_params = fp32_params

        if getattr(args, 'fp16_scale_window', None) is None:
            data_parallel_size = args.distributed_world_size
            scale_window = int(2**14 / data_parallel_size / args.update_freq)
        else:
            scale_window = args.fp16_scale_window

        if not getattr(args, 'bf16', False):
            self.scaler = DynamicLossScaler(
                init_scale=args.fp16_init_scale,
                scale_window=scale_window,
                tolerance=args.fp16_scale_tolerance,
                threshold=args.threshold_loss_scale,
            )
            self.min_loss_scale = self.args.min_loss_scale
        else:
            # disable loss scaling for bfloat16
            self.scaler = None

    @classmethod
    def build_optimizer(cls, args, params):
        """
        Args:
            args (argparse.Namespace): fairseq args
            params (iterable): iterable of parameters to optimize
        """
        from . import optimizers

        if args.optimizer != 'pair':
            # create FP32 copy of parameters and grads
            fp32_params = cls.build_fp32_params(params)
            # build FP32 optimizer with original param groups
            fp32_optimizer = optimizers[args.optimizer](args, fp32_params)
            # build FP16 optimizer with a unified param group
            params = cls.combine_param_groups(params)
            fp32_params = cls.combine_param_groups(fp32_params)
        else:
            major_params, second_params = params
            fp32_major_params = cls.build_fp32_params(major_params)
            fp32_second_params = cls.build_fp32_params(second_params)
            fp32_optimizer = optimizers[args.optimizer](args, [fp32_major_params, fp32_second_params])
            params = cls.combine_param_groups(major_params) + cls.combine_param_groups(second_params)
            fp32_params = cls.combine_param_groups(fp32_major_params) + cls.combine_param_groups(fp32_second_params)
        return cls(args, params, fp32_optimizer, fp32_params)

    @property
    def optimizer(self):
        return self.fp32_optimizer.optimizer

    @property
    def optimizer_config(self):
        return self.fp32_optimizer.optimizer_config

    def get_lr(self):
        return self.fp32_optimizer.get_lr()

    def set_lr(self, lr):
        self.fp32_optimizer.set_lr(lr)


class _MemoryEfficientFP16OptimizerMixin(object):

    def __init__(self, *args, **kwargs):
        # forward __init__ call to the next class in mro(method resolution order)
        super().__init__(*args, **kwargs)

    @property
    def has_flat_params(self):
        return False

    def state_dict(self):
        """Return the optimizer's state dict."""
        state_dict = self.wrapped_optimizer.state_dict()
        if self.scaler is not None:
            state_dict['loss_scale'] = self.scaler.loss_scale
        return state_dict

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        if 'loss_scale' in state_dict and self.scaler is not None:
            self.scaler.loss_scale = state_dict['loss_scale']

        self.wrapped_optimizer.load_state_dict(state_dict, optimizer_overrides)

        # Hack: PyTorch automatically casts the optimizer state to match the
        # type of the current parameters. But with --memory-efficient-fp16 the
        # params are FP16 while the optimizer state is FP32 and we don't want
        # to cast. A workaround is to manually copy back the original state
        # after the optimizer has been loaded.
        groups = self.optimizer.param_groups
        saved_groups = state_dict['param_groups']
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain(*(g['params'] for g in saved_groups)),
                chain(*(g['params'] for g in groups))
            )
        }
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                self.optimizer.state[param] = v

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves.

        Compared to :func:`fairseq.optim.FairseqOptimizer.backward`, this
        function additionally dynamically scales the loss to avoid gradient
        underflow.
        """
        if self.scaler is not None:
            loss = loss * self.scaler.loss_scale
            self._grads_are_scaled = True
            self._multiply_factor = 1
        loss.backward()

    def _unscale_grads(self):
        if self._grads_are_scaled:
            self._grads_are_scaled = False

            # correct for dynamic loss scaler
            self.wrapped_optimizer.multiply_grads(self._multiply_factor / self.scaler.loss_scale)
            self._multiply_factor = 1
        else:
            assert self._multiply_factor == 1

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        if self._grads_are_scaled:
            self._multiply_factor *= c
        else:
            self.wrapped_optimizer.multiply_grads(c)

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm and updates dynamic loss scaler."""

        # detect overflow and adjust loss scale
        if self.scaler is not None:
            scale = self._multiply_factor / self.scaler.loss_scale
            grad_norm = self.wrapped_optimizer.clip_grad_norm(0, aggregate_norm_fn) * scale
            grad_norm_cpu = float(grad_norm)
            if grad_norm_cpu > max_norm:
                self._multiply_factor *= max_norm / grad_norm_cpu
            overflow = DynamicLossScaler.has_overflow(grad_norm_cpu)
            prev_scale = self.scaler.loss_scale
            self.scaler.update_scale(overflow)
            if overflow:
                if self.scaler.loss_scale <= self.min_loss_scale:
                    # Use FloatingPointError as an uncommon error that parent
                    # functions can safely catch to stop training.
                    self.scaler.loss_scale = prev_scale
                    raise FloatingPointError((
                        'Minimum loss scale reached ({}). Your loss is probably exploding. '
                        'Try lowering the learning rate, using gradient clipping or '
                        'increasing the batch size.'
                    ).format(self.min_loss_scale))
                raise OverflowError('setting loss scale to: ' + str(self.scaler.loss_scale))
        else:
            self._unscale_grads()
            grad_norm = self.wrapped_optimizer.clip_grad_norm(max_norm, aggregate_norm_fn)

        return grad_norm

    def step(self, closure=None):
        """Performs a single optimization step."""
        if self.supports_step_with_scale and self._grads_are_scaled:
            scale = self._multiply_factor / self.scaler.loss_scale
            self.wrapped_optimizer.step(closure, scale=scale)
        else:
            self._unscale_grads()
            self.wrapped_optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self.wrapped_optimizer.zero_grad()
        self._grads_are_scaled = False


class MemoryEfficientFP16Optimizer(_MemoryEfficientFP16OptimizerMixin, Optimizer):
    """
    Wrap an *optimizer* to support FP16 (mixed precision) training.

    Compared to :class:`fairseq.optim.FP16Optimizer`, this version does not
    maintain an FP32 copy of the model. We instead expect the optimizer to
    convert the gradients to FP32 internally and sync the results back to the
    FP16 model params. This significantly reduces memory usage but slightly
    increases the time spent in the optimizer.

    Since this wrapper depends on specific functionality in the wrapped
    optimizer (i.e., on-the-fly conversion of grads to FP32), only certain
    optimizers can be wrapped. This is determined by the
    *supports_memory_efficient_fp16* property.
    """

    def __init__(self, args, params, optimizer):
        if not optimizer.supports_memory_efficient_fp16:
            raise ValueError(
                'Unsupported optimizer: {}'.format(optimizer.__class__.__name__)
            )

        super().__init__(args)
        self.wrapped_optimizer = optimizer

        if getattr(args, 'fp16_scale_window', None) is None:
            data_parallel_size = int(args.distributed_world_size)
            scale_window = 2**14 / data_parallel_size / args.update_freq
        else:
            scale_window = args.fp16_scale_window

        if not getattr(args, 'bf16', False):
            self.scaler = DynamicLossScaler(
                init_scale=args.fp16_init_scale,
                scale_window=scale_window,
                tolerance=args.fp16_scale_tolerance,
                threshold=args.threshold_loss_scale,
            )
            self.min_loss_scale = self.args.min_loss_scale
        else:
            # disable loss scaling for bfloat16
            self.scaler = None

    @classmethod
    def build_optimizer(cls, args, params):
        """
        Args:
            args (argparse.Namespace): fairseq args
            params (iterable): iterable of parameters to optimize
        """
        from . import optimizers
        fp16_optimizer = optimizers[args.optimizer](args, params)
        return cls(args, params, fp16_optimizer)

    @property
    def optimizer(self):
        return self.wrapped_optimizer.optimizer

    @property
    def optimizer_config(self):
        return self.wrapped_optimizer.optimizer_config

    def get_lr(self):
        return self.wrapped_optimizer.get_lr()

    def set_lr(self, lr):
        self.wrapped_optimizer.set_lr(lr)


try:
    from amp_C import multi_tensor_l2norm
    multi_tensor_l2norm_available = True
except ImportError:
    multi_tensor_l2norm_available = False


def multi_tensor_total_norm(grads, chunk_size=2048*32) -> torch.Tensor:
    per_device_grads = {}
    norms = []
    for grad in grads:
        device = grad.device
        cur_device_grads = per_device_grads.get(device)
        if cur_device_grads is None:
            cur_device_grads = []
            per_device_grads[device] = cur_device_grads
        cur_device_grads.append(grad)
    for device in per_device_grads.keys():
        cur_device_grads = per_device_grads[device]
        if device.type == "cuda":
            # TODO(msb) return has_inf
            has_inf = torch.zeros((1, 1), dtype=torch.int, device=device)
            with torch.cuda.device(device):
                norm = multi_tensor_l2norm(chunk_size, has_inf, [cur_device_grads], False)
                norms.append(norm[0])
        else:
            norms += [torch.norm(g, p=2, dtype=torch.float32) for g in cur_device_grads]
    total_norm = torch.norm(torch.stack(norms))
    return total_norm


def clip_grad_norm_(params, max_norm, aggregate_norm_fn=None) -> torch.Tensor:
    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(params)
    grads = [p.grad.detach() for p in filter(lambda p: p.grad is not None, params)]
    if len(grads) == 0:
        if len(params) > 0:
            return params[0].new_tensor(0.)
        else:
            return torch.tensor(0.)

    if len(grads) == 1:
        total_norm = torch.norm(grads[0], p=2, dtype=torch.float32)
    else:
        if multi_tensor_l2norm_available:
            total_norm = multi_tensor_total_norm(grads)
        else:
            warnings.warn(
                "amp_C fused kernels unavailable, disabling multi_tensor_l2norm; "
                "you may get better performance by installing NVIDIA's apex library"
            )
            total_norm = torch.norm(
                torch.stack([torch.norm(g, p=2, dtype=torch.float32) for g in grads])
            )

    if aggregate_norm_fn is not None:
        total_norm = aggregate_norm_fn(total_norm)

    if max_norm > 0:
        max_norm = float(max_norm)
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp_(max=1)
        for g in grads:
            g.mul_(clip_coef)
    return total_norm
