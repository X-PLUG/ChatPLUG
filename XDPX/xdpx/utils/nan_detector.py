# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class NanDetector:
    """
        Detects the first NaN or Inf in forward and/or backward pass and logs, together with the module name.
        If operation that causes NaN is done by nn.Parameter, it cannot be detected by NanDetector.
    """

    def __init__(self, model, loss, forward=True, backward=True):
        self.bhooks = []
        self.fhooks = []
        self.forward = forward
        self.backward = backward
        self.reset()

        for name, mod in model.named_modules():
            mod.__module_name = name
            self.add_hooks(mod)
        loss.__module_name = loss.__class__.__name__
        self.add_hooks(loss)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def add_hooks(self, module):
        if self.forward:
            self.fhooks.append(module.register_forward_hook(self.fhook_fn))
        if self.backward:
            self.bhooks.append(module.register_backward_hook(self.bhook_fn))

    def reset(self):
        self.has_printed_f = False
        self.has_printed_b = False

    def _detect(self, tensor, name, backward):
        err = None
        with torch.no_grad():
            if torch.isnan(tensor).any():
                err = "NaN"
            elif torch.isinf(tensor).any():
                err = "Inf"
        if err is not None:
            err = f"{err} detected in output of {name}, shape: {tensor.shape}, {'backward' if backward else 'forward'}"
            err += '\n' + str(tensor)
            inf_cnt = torch.isinf(tensor).sum().item()
            nan_cnt = torch.isnan(tensor).sum().item()
            inf_msg = str(inf_cnt) + (f'/{tensor.numel()}' if inf_cnt else '')
            nan_msg = str(nan_cnt) + (f'/{tensor.numel()}' if nan_cnt else '')
            err += '\n' + f'{inf_msg} Inf; {nan_msg} Nan; ' + '\n'
        return err

    def _apply(self, module, inp, x, backward):
        if torch.is_tensor(x):
            if isinstance(inp, tuple) and len(inp) > 0:
                inp = inp[0]
            err = self._detect(x, module.__module_name, backward)
            if err is not None:
                if torch.is_tensor(inp) and not backward:
                    err += (
                        f" input max: {inp.max().item()}, input min: {inp.min().item()}"
                    )

                has_printed_attr = 'has_printed_b' if backward else 'has_printed_f'
                print('|', err)
                setattr(self, has_printed_attr, True)
        elif isinstance(x, dict):
            for v in x.values():
                self._apply(module, inp, v, backward)
        elif isinstance(x, list) or isinstance(x, tuple):
            for v in x:
                self._apply(module, inp, v, backward)

    def fhook_fn(self, module, inp, output):
        if not self.has_printed_f:
            self._apply(module, inp, output, backward=False)

    def bhook_fn(self, module, inp, output):
        if not self.has_printed_b:
            self._apply(module, inp, output, backward=True)

    def close(self):
        for hook in self.fhooks + self.bhooks:
            hook.remove()
