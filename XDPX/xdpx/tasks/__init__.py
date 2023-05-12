import json
import os
import importlib
import torch
import numpy as np
from functools import partial

from torch.utils.data import DataLoader
from xdpx.utils import register, io, cache_file, get_train_subsets, numpy_seed
from xdpx.utils.io_utils import tqdm, CallbackIOWrapper
from xdpx.processors import processors
from xdpx.options import Argument
from xdpx.datasets import datasets
from xdpx.datasets.samplers import samplers

from torch.cuda.amp import autocast


tasks = {}
register = partial(register, registry=tasks)


@register('default')
class Task:
    TRAIN = 'train'
    VALID = 'valid'

    @classmethod
    def register(cls, options):
        from xdpx.models import models, Model
        from xdpx.losses import losses
        options.register(
            Argument('model', required=True, validate=lambda value: value in models,
                     register=lambda value: Model.build_model_class(value).register),
            domain='model',
        )
        options.register(
            Argument('loss', required=True, validate=lambda value: value in losses,
                     register=lambda value: losses[value].register),
            domain='loss'
        )
        cls.register_dataset_options(options)

    @classmethod
    def register_dataset_options(cls, options):
        options.register(
            Argument('dataset', default='default', validate=lambda value: value in datasets,
                     register=lambda value: datasets[value].register),
            domain='dataset'
        )
        options.register(
            Argument('dataset_format', default='torch', validate=lambda value: value in ('torch', 'jsonl')),
            domain='dataset'
        )
        options.register(
            Argument('sampler', default='default', validate=lambda value: value in samplers,
                     register=lambda value: samplers[value].register),
            domain='dataset/sampler'
        )
        options.register(
            Argument('batch_by_len', default=False),
        )

        def validate_batch_size(args):
            actual_bsz = args.batch_size * args.update_freq
            if args.lazy_load:
                min_data_sz = min(args.data_size[name] for name in get_train_subsets(args))
            else:
                min_data_sz = sum(args.data_size[name] for name in get_train_subsets(args))
            return actual_bsz < min_data_sz, \
                   f'actual batch size {actual_bsz} can not be larger than (minimum) training data size {min_data_sz}'

        options.add_global_constraint(validate_batch_size)
    
    def __init__(self, args):
        self.args = args
        self.processor = processors[args.processor](args)
        self.datasets = {}
        self.data_loaders = {}
        self.num_updates = 0
    
    def build_model(self, args):
        from xdpx.models import models
        return models[args.model].build(args)
    
    def build_loss(self, args):
        from xdpx.losses import losses
        return losses[args.loss](args)

    def load_dataset(self, splits, is_train, reload=False):
        split_name = self.TRAIN if is_train else self.VALID
        if isinstance(splits, str):
            splits = [splits]
        if split_name in self.data_loaders and all(split in self.datasets for split in splits):
            if reload:
                print(f'| Reload dataset {split_name} with {splits}')
            else:
                return self.data_loaders[split_name]
        
        data = []
        for split in splits:
            if split not in self.datasets:
                if self.args.dataset_format == 'torch':
                    path = os.path.join(self.args.data_dir, f'{split}.pt')
                    if self.args.cache_train_file:
                        path = cache_file(path)
                    if os.path.exists(path):
                        md5sum = io.md5(path)
                        print(f'| Loading dataset {split} ({md5sum})')
                    data_fsize = io.size(path)
                    with tqdm(total=data_fsize, unit='B', unit_scale=True, unit_divisor=1024,
                              leave=data_fsize > 100 * 1024 ** 2,  # 100M
                              desc=f'reading {split}') as t, io.open(path, 'rb') as obj:
                        obj = CallbackIOWrapper(t.update, obj, "read")
                        data_i = torch.load(obj)
                    self.datasets[split] = data_i
                elif self.args.dataset_format == 'jsonl':
                    path = os.path.join(self.args.data_dir, f'{split}.jsonl')
                    if self.args.cache_train_file:
                        path = cache_file(path)
                    if os.path.exists(path):
                        md5sum = io.md5(path)
                        print(f'| Loading dataset {split} ({md5sum})')
                    num_lines = io.lines(path)
                    data_i = []
                    with io.open(path) as f:
                        for item in tqdm(f, total=num_lines):
                            data_i.append(json.loads(item))
                    self.datasets[split] = data_i
            data.extend(self.datasets[split])
        data = self.build_dataset(data, is_train)
        self.data_loaders[split_name] = data
        return data
    
    def build_dataset(self, data: list, is_train: bool) -> DataLoader:
        dataset_class = datasets[getattr(self.args, 'dataset', 'default')]
        sampler_class = samplers[getattr(self.args, 'sampler', 'default')]
        data = dataset_class(data, is_train)

        if self.args.distributed_world_size > 1:
            import torch.distributed as dist
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        else:
            num_replicas = 1
            rank = 0
        sampler = sampler_class(self.args,
            data, self.args.batch_size, shuffle=is_train, seed=self.args.seed,
            num_replicas=num_replicas, rank=rank, update_freq=self.args.update_freq if is_train else 1,
            sort_key=self.processor.text_length if self.args.batch_by_len else None
        )
        if isinstance(sampler, torch.utils.data.BatchSampler):
            data = DataLoader(
                data, collate_fn=self.processor.collate, batch_sampler=sampler
            )
        else:
            data = DataLoader(
                data, batch_size=self.args.batch_size, drop_last=is_train, pin_memory=True,
                collate_fn=self.processor.collate, sampler=sampler
            )
        return data

    def set_epoch(self, epoch, reload=False):
        if self.args.lazy_load:
            subsets = get_train_subsets(self.args, reload=True)
            if epoch >= len(subsets):
                with numpy_seed(self.args.seed, epoch // len(subsets)):
                    np.random.shuffle(subsets)
                split = subsets[epoch % len(subsets)]
            else:
                split = subsets[epoch]
            if split not in self.datasets:
                # clear previous loaded train subsets to save memory
                self.datasets = {self.args.valid_subset: self.datasets[self.args.valid_subset]}
                self.load_dataset(split, is_train=True, reload=reload)
        elif reload:
            self.load_dataset(get_train_subsets(self.args), is_train=True, reload=True)
        loader = self.data_loaders[self.TRAIN]
        if hasattr(loader, 'sampler'):
            sampler = loader.sampler
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
        self.processor.set_epoch(epoch)

    def train_step(self, sample, model, loss, optimizer, num_updates=0):
        model.train()
        loss.train()
        if self.args.bf16:
            with autocast(dtype=torch.bfloat16):
                loss, sample_size, logging_output = loss(model, sample)
        else:
            loss, sample_size, logging_output = loss(model, sample)

        if self.args.deepspeed_zero_stage>0:
            model.backward(loss)
        else:
            optimizer.backward(loss)

        logging_output['ntokens'] = sample['ntokens']

        if self.args.inspect_gradient and num_updates % self.args.eval_interval == 0:
            from xdpx.visualize import plot_grad_flow
            plot_grad_flow(
                model.named_parameters(),
                os.path.join(self.args.save_dir, 'plots', 'gradients', f'{num_updates}.' + self.args.figext)
            )
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, loss):
        model.eval()
        loss.eval()
        with torch.no_grad():
            if self.args.bf16:
                with autocast(dtype=torch.bfloat16):
                    loss, sample_size, logging_output = loss(model, sample)
            else:
                loss, sample_size, logging_output = loss(model, sample)
        return loss, sample_size, logging_output

    def inference_step(self, sample, model, loss):
        model.eval()
        loss.eval()
        with torch.no_grad():
            pred, prob = loss.inference(model, sample)
        target_prob = [p[i] for p, i in zip(prob, pred)]
        pred = [self.processor.target_map[i] for i in pred]
        return pred, prob, target_prob

    @property
    def inference_header(self):
        return 'pred prob target_prob'.split()

    def distill_step(self, sample, model, loss):
        "get logits of given samples"
        model.eval()
        loss.eval()
        with torch.no_grad():
            logits = loss.distill(model, sample)
        if 'target' not in sample:
            with torch.no_grad():
                pred, prob = loss.inference(model, sample)
            pred = [self.processor.target_map[i] for i in pred]
            return logits, pred
        return (logits,)

    def update_step(self, num_updates):
        """Task level update when number of update increases. This is called after optimization step and
           learning rate update of each step"""
        self.num_updates = num_updates

    def aggregate_logging_outputs(self, logging_outputs, sample_size, loss, max_count=None):
        return loss.aggregate_logging_outputs(logging_outputs, sample_size, max_count)
    
    @staticmethod
    def build_task_class(tasks):
        raise DeprecationWarning('''
            Please use `from xdpx.tasks import tasks; task = tasks[args.task](args)` instead of 
            `task = Task.build_task_class(args.task)(args)` to build the task.
        ''')


@register('chat')
class ChatTask(Task):

    def inference_step(self, sample, model, loss):
        model.eval()
        loss.eval()
        with torch.no_grad():
            index = loss.inference(model, sample)
        tokens, texts = [], []
        for s in index:
            s_ = []
            for _ in s:
                if _ < 1:
                    continue
                s_.append(_)
            tokens_ = self.processor.decode(s_)
            tokens.append(' '.join(tokens_[1:]))
            texts.append(''.join(tokens_[1:]))
        return tokens, texts

    @property
    def inference_header(self):
        return 'tokens generation'.split()


# some of model-specific tasks are defined in models, so we import them here.
from .. import models, losses
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('.' + module_name, __name__)
