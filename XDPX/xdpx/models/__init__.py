import os
import importlib
import copy
import inspect
from contextlib import contextmanager
from collections import OrderedDict
from functools import lru_cache
import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import partial
from typing import Union, List
from xdpx.options import Argument
from xdpx.utils import register, distributed_utils, io, parse_model_path, validate_url, download_from_url

models = {}
register = partial(register, registry=models)


@lru_cache(maxsize=8)
def _validate_pretrain_path(pretrained_model, from_tf):
    value = pretrained_model
    if not value:
        return True
    if value.startswith('http'):
        if from_tf:
            return False, f'Currently loading a zipfile containing tf checkpoints is not supported. Invalid url\n{value}'
        return validate_url(value), f'Exception when trying to access the url\n{value}'
    if from_tf:
        return io.exists(value + '.meta') and io.exists(value + '.index') and \
               io.exists(value + '.data-00000-of-00001'), f'invalid Tensorflow checkpoint path {value}'
    if io.isdir(value):
        return False, f'pretrained_model must be a path to a file, not a directory: {value}'
    return io.exists(value), f'pretrained_model path does not exist: {value}'


class Model(nn.Module):
    @staticmethod
    def register(options):
        options.register(
            Argument(
                'pretrained_model',
                doc='pytorch state_dict file or tensorflow checkpoint path, '
                    'can use <best> <last> to refer to the best or last pytorch checkpoint',
                # must use independent post_process here to support "children"
                post_process=lambda val: parse_model_path(val, None), type=str,
                children=[
                    Argument('strict_size', default=True,
                             doc='otherwise will ignore size mismatch when loading the pretrained model'),
                ]
            ),
            Argument('from_tf', default=False, children=[
                Argument('tf_format', default='checkpoint', validate=lambda val: val in ('checkpoint', 'savedmodel'),
                         children={lambda val: val == 'savedmodel': [
                             Argument('signature_def_key', default='serving_default'),
                         ]}),
            ]),
        )
        options.register(
            Argument('auto_model',
                     doc='bert configs related to HuggingFace backend, if auto_model is not None, related options in args will be overwrittern'),
            domain='huggingface',
        )

        def validate_pretrain_path(args):
            return _validate_pretrain_path(args.pretrained_model, args.from_tf)

        options.add_global_constraint(validate_pretrain_path)

    def __init__(self, args):
        super().__init__()
        self.args = args

    @classmethod
    def build(cls, args):
        model = cls(args)
        from .bert import Bert
        if hasattr(args, '__cmd__') and args.__cmd__ == 'train' \
                and hasattr(args, 'resume') and args.resume is False:
            # if issubclass(cls, Bert) and args.auto_model:
            #     from transformers import AutoModel
            #     model.bert = AutoModel.from_pretrained(args.auto_model)
            #     print(f'| Weights loaded for {cls.__name__} from huggingface: {args.auto_model}.')
            # elif args.pretrained_model:
            if args.pretrained_model:
                model_path = args.pretrained_model
                if args.pretrained_model.startswith('http'):
                    model_path = download_from_url(args.pretrained_model)
                model.load(model_path)

        return model

    @staticmethod
    def build_model_class(model_name):

        class CombinedModule(models[model_name]):
            @classmethod
            def register(cls, options):
                # remove duplicated register hooks while keeping the mro order
                for hook in OrderedDict((module.register, None) for module in cls.__mro__[-1:0:-1]
                                        if hasattr(module, 'register')).keys():
                    hook(options)

        return CombinedModule

    @property
    def dummy_inputs(self):
        """for TorchScript tracing. Return a tuple which is the model input."""
        raise NotImplementedError

    def dummy_tf_inputs(self, inputs=None):
        """for comparing tf savedmodel results"""
        raise NotImplementedError

    def trainable_parameters(self) -> List[Union[torch.Tensor, dict]]:
        """for torch.optim.Optimizer"""
        return [param for param in self.parameters() if param.requires_grad]

    @property
    def name_map(self):
        return {}

    def customized_name_mapping(self, state_dict):
        return state_dict

    def _load_and_parse_state_dict(self, path):
        with io.open(path, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            if 'model' in state_dict:
                state_dict = state_dict['model']
            return state_dict

    def load(self, path, from_tf=False):
        if from_tf:
            from xdpx.utils.tf_utils import load_vars_from_tf_checkpoint, load_vars_from_tf_savedmodel
            if self.args.tf_format == 'checkpoint':
                tf_vars = load_vars_from_tf_checkpoint(path)
            else:
                tf_vars = load_vars_from_tf_savedmodel(path, signature_def_key=self.args.signature_def_key)
            return self.load_from_tf(tf_vars)
        state_dict = self._load_and_parse_state_dict(path)
        # update state dict
        for old_name, new_name in self.name_map.items():
            if old_name in state_dict:
                state_dict[new_name] = state_dict.pop(old_name)
        state_dict = self.customized_name_mapping(state_dict)
        # avoid using set subtraction to keep the original order
        model_state_dict = self.state_dict()
        model_keys = model_state_dict.keys()
        missing_keys = [key for key in model_keys if key not in state_dict]
        unexpected_keys = [key for key in state_dict if key not in model_keys]
        mismatched_keys = [key for key in model_keys if key in state_dict and
                           state_dict[key].shape != model_state_dict[key].shape]
        if self.args.strict_size and mismatched_keys:
            raise RuntimeError(f'Found size mismatch when strict_size=True\n' +
                               '\n'.join(f'\t{key}: expected {str(model_state_dict[key].shape)},'
                                         f' found {str(state_dict[key].shape)}' for key in mismatched_keys))
        for key in mismatched_keys:
            del state_dict[key]
        all_mismatch = len(missing_keys) + len(mismatched_keys) == len(model_keys)
        if not all_mismatch:
            self.load_state_dict(state_dict, strict=False)
            print(f'| Weights loaded for {self.__class__.__name__} from {path}.')
        if missing_keys:
            print(f'| Missing keys:\n|\t' + '\n|\t'.join(missing_keys))
        if unexpected_keys:
            print(f'| Unexpected keys:\n|\t' + '\n|\t'.join(unexpected_keys))
        if mismatched_keys:
            print(f'| Mismatched keys:\n|\t' + '\n|\t'.join(mismatched_keys))
        print()
        if all_mismatch:
            raise RuntimeError(f'Checkpoint mismatch: None of the parameters can be loaded from {path}')
        return self

    def load_from_tf(self, path):
        raise NotImplementedError(f'Model {self.__class__.__name__} does not implement loading from tf')

    def load_into_tf(self, sess, strict=True) -> dict:
        """return predict_signature"""
        raise NotImplementedError(f'Model {self.__class__.__name__} does not implement export to tf')

    def get_embeddings(self):
        """returns the embedding module of the model"""
        raise NotImplementedError

    def load_embeddings(self, module_name=None):
        embedding_path = os.path.join(self.args.data_dir, 'embeddings.pt')
        if hasattr(self.args, '__cmd__') and self.args.__cmd__ == 'train' and io.exists(embedding_path):
            if module_name is not None:
                module = getattr(self, module_name)
            else:
                module = self.get_embeddings()
            with io.open(embedding_path, 'rb') as f:
                embeddings = torch.load(f)
            module.set_(embeddings)
            print('| pretrained embeddings loaded.')


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('.' + module_name, __name__)

for file in os.listdir(os.path.join(os.path.dirname(__file__),'fewshot')):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('.fewshot.' + module_name, __name__)


def DistributedDataParallel(args, model, process_group=None):
    """
    Wrap a *model* to support distributed data parallel training.

    This is similar to the built-in DistributedDataParallel, but allows
    additional configuration of the DistributedDataParallel class to
    use, and also provides easier access to the wrapped model by
    forwarding requests for missing attributes to the wrapped model.

    Args:
        args (argparse.Namespace): fairseq args
        model (BaseFairseqModel): model to wrap
    """
    assert isinstance(model, nn.Module)
    if args.ddp_backend == 'no_c10d':
        ddp_class = LegacyDistributedDataParallel
        init_kwargs = dict(
            module=model,
            world_size=args.distributed_world_size,
            buffer_size=2 ** 28,
            process_group=process_group,
        )
    else:
        """
        This implementation mysteriously hangs when training masked_lm on pytorch 1.0 (always hangs, reproducably)
        Temporarily use "LegacyDistributedDataParallel" which is "slower but more robust"
        """
        assert args.ddp_backend == 'c10d'
        ddp_class = nn.parallel.DistributedDataParallel
        init_kwargs = dict(
            module=model,
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=False,
            bucket_cap_mb=args.bucket_cap_mb,
            process_group=process_group,
        )
        # Maintain backward compatibility
        if 'check_reduction' in inspect.getargspec(ddp_class)[0]:
            init_kwargs['check_reduction'] = True
        if 'find_unused_parameters' in inspect.getargspec(ddp_class)[0]:
            # available since pytorch 1.2
            init_kwargs['find_unused_parameters'] = True

    class _DistributedModel(ddp_class):
        """Extend DistributedDataParallel to check for missing
        attributes in the wrapped module."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getattr__(self, name):
            wrapped_module = super().__getattr__('module')
            if hasattr(wrapped_module, name):
                return getattr(wrapped_module, name)
            return super().__getattr__(name)

    return _DistributedModel(**init_kwargs)


class LegacyDistributedDataParallel(nn.Module):
    """
    Reference: https://github.com/pytorch/fairseq/blob/master/fairseq/legacy_distributed_data_parallel.py

    Implements distributed data parallelism at the module level.

    A simplified version of :class:`torch.nn.parallel.DistributedDataParallel`.
    This version uses a c10d process group for communication and does not
    broadcast buffers.

    Args:
        module (~torch.nn.Module): module to be parallelized
        world_size (int): number of parallel workers
        process_group (optional): the c10d process group to be used for
            distributed data all-reduction. If None, the default process group
            will be used.
        buffer_size (int, optional): number of elements to buffer before
            performing all-reduce (default: 256M).
    """

    def __init__(self, module, world_size, process_group=None, buffer_size=2 ** 28):
        super().__init__()

        self.module = module
        self.world_size = world_size
        self.process_group = process_group

        # Never use a bigger buffer than the number of model params
        self.buffer_size = min(buffer_size, sum(p.numel() for p in module.parameters()))
        self.buffer = None

        # Flag used by the NCCL backend to make sure we only reduce gradients
        # one time in the execution engine
        self.need_reduction = False

        # We can also forcibly accumulate grads locally and only do the
        # all-reduce at some later time
        self.accumulate_grads = False

        # For NCCL backend, since every single NCCL call is asynchoronous, we
        # therefore directly enqueue all the NCCL reduction calls to the
        # default CUDA stream without spawning up other reduction threads.
        # This achieves the best performance.
        self._register_grad_hook()

    def __getstate__(self):
        attrs = copy.copy(self.__dict__)
        return attrs

    def __setstate__(self, state):
        super().__setstate__(state)
        self._register_grad_hook()

    @contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        old_accumulate_grads = self.accumulate_grads
        self.accumulate_grads = True
        yield
        self.accumulate_grads = old_accumulate_grads

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _register_grad_hook(self):
        """
        This function registers the callback all-reduction function for the
        NCCL backend. All gradients will be all reduced in one single step.
        The NCCL reduction will directly be enqueued into the default CUDA
        stream. Therefore, no synchronization is needed.
        """

        def all_reduce(params):
            buffer = self.buffer
            nonzero_buffer = False
            if len(params) > 1:
                offset = 0
                for p in params:
                    sz = p.numel()
                    if p.grad is not None:
                        buffer[offset:offset + sz].copy_(p.grad.data.view(-1))
                        nonzero_buffer = True
                    else:
                        buffer[offset:offset + sz].zero_()
                    offset += sz
            else:
                # we only have a single grad to all-reduce
                p = params[0]
                if p.grad is not None:
                    buffer = p.grad.data
                    nonzero_buffer = True
                elif p.numel() <= self.buffer.numel():
                    buffer = buffer[:p.numel()]
                    buffer.zero_()
                else:
                    buffer = torch.zeros_like(p)

            if nonzero_buffer:
                buffer.div_(self.world_size)

            distributed_utils.all_reduce(buffer, self.process_group)

            # copy all-reduced grads back into their original place
            offset = 0
            for p in params:
                sz = p.numel()
                if p.grad is not None:
                    p.grad.data.copy_(buffer[offset:offset + sz].view_as(p))
                else:
                    p.grad = buffer[offset:offset + sz].view_as(p).clone()
                offset += sz

        def reduction_fn():
            # This function only needs to be called once
            if not self.need_reduction or self.accumulate_grads:
                return
            self.need_reduction = False

            if self.buffer is None:
                self.buffer = next(self.module.parameters()).new(self.buffer_size)

            # All-reduce the gradients in buckets
            offset = 0
            buffered_params = []
            for param in self.module.parameters():
                if not param.requires_grad:
                    continue
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                if param.grad.requires_grad:
                    raise RuntimeError("DistributedDataParallel only works "
                                       "with gradients that don't require "
                                       "grad")
                sz = param.numel()
                if sz > self.buffer.numel():
                    # all-reduce big params directly
                    all_reduce([param])
                else:
                    if offset + sz > self.buffer.numel():
                        all_reduce(buffered_params)
                        offset = 0
                        buffered_params.clear()
                    buffered_params.append(param)
                    offset += sz

            if len(buffered_params) > 0:
                all_reduce(buffered_params)

        # Now register the reduction hook on the parameters
        for p in self.module.parameters():

            def allreduce_hook(*unused):
                self.need_reduction = True
                Variable._execution_engine.queue_callback(reduction_fn)

            if p.requires_grad:
                p.register_hook(allreduce_hook)
