import os
import re
import sys
import math
import time
import random
import contextlib
import torch
import pandas as pd
from functools import partial
from tqdm import tqdm
from functools import lru_cache
from multiprocessing import Process
from typing import List, Dict, Union, Optional, Any
from itertools import chain
from .options import Argument, parse_relative, Arg
from . import utils
from .utils import (
    distributed_utils, io, persistent_run, cache_file,
    parse_model_path, convert_to_native, delayed_flush, get_total_steps,
    default_cache_root, get_train_subsets, pformat_dataframe,
)
from xdpx.utils.versions import torch_ne_13
from .utils.nan_detector import NanDetector
from .logger import Logger
from .tasks import tasks, Task
from .processors import processors
from .optimizers import optimizers, BMUFOptimizer
from .optimizers.lr_schedulers import lr_schedulers
from .visualize import ref_dir_arg, figext_arg
from .utils import EMA

import deepspeed

class Trainer:
    """
    Reference: https://github.com/pytorch/fairseq/blob/master/fairseq/trainer.py
    Reference date: Oct 1st, 2019
    2nd Reference data: March 27, 2020 (d37fdee3da996e1533eb4fe4b68a3683c8ce987b)

    Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """
    @staticmethod
    def auto_save_dir(val, args: Arg):
        if hasattr(args, 'auto_suffix') and args.auto_suffix:
            suffix = 1
            while io.exists(val):
                m = re.search(r'\((\d+)\)$', val)
                if m:
                    val = val[:m.start()]
                    suffix = int(m.group(1)) + 1
                val = val.rstrip('/') + f'({suffix})'
        return val

    @staticmethod
    def register(options):
        options.register(
            Argument('data_dir', required=True,
                     doc='path to preprocessed data and resource files from the preprocessing step',
                     validate=lambda value: (cached_exists(os.path.join(value, 'meta.hjson')),
                                             'data_dir does not exist or does not contain processed data')),
            Argument('save_dir', required=True, unique=True, type=str, validate=[
                lambda value: io.is_writable(value),
            ], post_process=Trainer.auto_save_dir),
            Argument('overwrite', default=False, doc='whether to overwrite save_dir if exists.', children={
                lambda value: not value: [
                    Argument('auto_suffix', default=False, doc='add auto suffix to save_dir if exists.'),
                ]
            }),
            Argument('resume', default=False, doc='resume previous training'),
            Argument('max_epoch', type=int),
            Argument('max_update', type=int, post_process=parse_relative),
            Argument('batch_size', type=int, required=True),
            Argument('update_freq', default=1, doc='Update parameters every N batches', children={
                lambda value: value > 1: [
                    Argument('force_sync', default=False,
                             doc='force to synchronize gradient in distributed training in each train step.')
                ]
            }),
            Argument('auto_ga', default=False,
                     doc='automatically turn on gradient accumulation when OOM occurs.'),
            Argument('max_loss', type=float, default=200., doc='force stop unstable training when loss exceeds max_loss'),
            Argument('min_lr', type=float, default=0.0),
            Argument('seed', default=1, type=int, doc='to disable random seed, use None;'),

            Argument('cache_train_file', default=True,
                     doc='better to cache train files when using remote path with multiple training subsets'),
            Argument('lazy_load', default=False, doc='load one shard at a time. Shuffle only within each shard'),
            Argument('data_size', type=Dict[str, int], doc='auto set by prepro stats'),
            Argument('ema_decay', default=1.0, type=float, doc='ema decay rate'),
            domain='trainer',
        )

        def get_train_steps(train_steps, args: Arg):
            if isinstance(train_steps, float):
                total_steps = get_total_steps(args, runtime=False, calibrate=False)
                train_steps = int(round(train_steps * total_steps))
            return train_steps

        options.register(
            Argument('train_subset', default='train', type=Union[str, List[str]],
                     doc='supports Unix filename pattern matching for multiple files'),
            Argument('exclude_valid_from_train', default=False),
            Argument('train_steps', type=Union[int, float], post_process=get_train_steps,
                     doc='if not None, train for only these steps. '
                         '(while lr_schedule still follow max_epoch or max_update)'),
            domain='trainer',
        )

        options.register(
            Argument('log_interval', default=10, post_process=parse_relative, type=int,
                     validate=lambda value: value > 0),
            Argument('log_file', default='log.txt', type=Optional[str], doc='log filename under "save_dir"'),
            ref_dir_arg('viz_ref_dir'),
            figext_arg('figext'),
            domain='trainer/log'
        )
        options.register(
            Argument('eval_interval', type=int, required=True, validate=lambda value: value > 0 or value == -1,
                     post_process=parse_relative, doc='-1 means just eval at the end of training'),
            Argument('save', default=True, children=[
                Argument('save_best_only', default=False),
                Argument('save_above_score', type=float),
                # In some cases the checkpoint will be very large, e.g. when Adam is used as the optimizer
                # the 1st & 2nd momentum of each param has to be saved and the checkpoint size will be 3x as a result.
                Argument('save_full_checkpoint', default=False,
                         doc='save full checkpoint to support resumed training in the future.'),
                Argument('save_last_only', default=False, doc='if save_best_only is also true, save best & last'),
            ]),
            Argument('tolerance', type=int, validate=lambda value: value is None or value > 0,
                     post_process=parse_relative),
            Argument('min_steps', type=int, default=1, post_process=parse_relative,
                     doc='minimum steps regardless of tolerance'),
            Argument('major_metric', required=True, doc='major metric for early stopping and display'),
            Argument('ascending_metric', default=True, doc='whether the major metric is the higher the better'),
            Argument('eval_interval_warmup', default=0, type=int, post_process=parse_relative,
                     doc='eval (& maybe save) less frequently in earlier steps',
                     children=[
                         Argument('eval_interval_warmup_mutiplier', default=20.),
                     ]),
            Argument('valid_subset', default='dev'),
            Argument('max_eval_steps', type=int, validate=lambda value: value is None or value > 0),
            Argument('inspect_gradient', default=False),
            domain='trainer/evaluation',
        )
        options.register(
            Argument('learning_rate', type=float, required=True),
            Argument('clip_norm', default=5., doc='gradient norm clipping'),
            domain='optimization',
        )
        options.register(
            Argument('optimizer', default='adam',
                     validate=lambda value: (value in optimizers, f'Unknown optimizer: {value}'),
                     register=lambda value: optimizers[value].register),
            domain='optimization/optimizer',
        )
        options.register(
            Argument('lr_scheduler', default='constant',
                     validate=lambda value: (value in lr_schedulers, f'Unknown lr_scheduler {value}'),
                     register=lambda value: lr_schedulers[value].register),
            domain='optimization/lr_scheduler'    
        )

        options.register(
            Argument(
                'task',
                validate=lambda value: (value in tasks, f'Unknown task: {value}'),
                register=lambda value: tasks[value].register, 
                default='default',
                type=str,
            ),
            domain='task',
        )
        options.register(
            Argument(
                'processor', required=True, 
                validate=lambda value: (value in processors.keys(), f'Unknown processor {value}'),
                register=lambda value: processors[value].register
            ),
            domain='processor'
        )
        
        def infer_init_method():
            if all(key in os.environ for key in ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK']) \
                    and (int(os.environ['WORLD_SIZE']) > 1):
                print('| MASTER_ADDR', os.environ['MASTER_ADDR'])
                print('| MASTER_PORT', os.environ['MASTER_PORT'])
                print('| WORLD_SIZE', os.environ['WORLD_SIZE'])
                print('| RANK', os.environ['RANK'])
                if 'VISIBLE_DEVICE_LIST' in os.environ:
                    print('| VISIBLE_DEVICE_LIST', os.environ['VISIBLE_DEVICE_LIST'])
                return 'env://'
            elif torch.cuda.device_count() > 1:
                port = random.randint(10000, 20000)
                return 'tcp://localhost:{port}'.format(port=port)
            return None
        
        def infer_world_size(value, args: Arg):
            if args.cuda and args.distributed_init_method and args.distributed_init_method.startswith('env'):
                if not torch_ne_13():
                    return int(os.environ['WORLD_SIZE'])  # docker fusion
                return int(os.environ['WORLD_SIZE']) * value
            return value
        
        def infer_dist_rank(value, args: Arg):
            if args.cuda and args.distributed_init_method and args.distributed_init_method.startswith('env'):
                return int(os.environ['RANK'])
            return value

        options.register(
            Argument('cuda', default=torch.cuda.is_available(), children=[
                    Argument('distributed_init_method', default=infer_init_method(), children=[
                        Argument('distributed_backend', default='nccl'),
                        Argument('bucket_cap_mb', default=25, doc='bucket size (in MB) for reduction'),
                        Argument('ddp_backend', default='no_c10d', doc='DistributedDataParallel backend', 
                                 validate=lambda value: value in 'c10d no_c10d'.split()),
                        Argument('use_bmuf', default=False,
                                 doc='specify global optimizer for syncing models on different GPUs/shards',
                                 register=lambda value: BMUFOptimizer.register if value else lambda x: ...),
                    ]),
                    Argument('empty_cache_freq', default=0,
                             doc='how often to clear the PyTorch CUDA cache (0 to disable)'),
                ]
            ),
            Argument('distributed_world_size', type=int, default=max(1, torch.cuda.device_count()),
                     post_process=infer_world_size),
            Argument('distributed_rank', type=int, default=0, post_process=infer_dist_rank),
            Argument('device_id', type=int, doc='set automatically'),
            domain='trainer/distributed',
        )
        options.register(
            # fp16 mixed precision training is only beneficial on GPUs with compute capability >=7, 
            # all the constant values (Adam betas, masked softmax, etc) should not go beyond
            # the FP16 range: (6e-8, 65504)
            Argument('fp16', default=False, doc='mixed precision training', children=[
                Argument('fp16_backend', default='fairseq', validate=lambda val: val in ('apex', 'fairseq'), children={
                    lambda val: True: [
                        Argument('min_loss_scale', default=0.1,
                                 doc='minimum FP16 loss scale, after which training is stopped'),
                    ],
                    lambda val: val == 'apex': [
                        Argument('opt_level', default='O1', validate=lambda value: value in 'O0 O1 O2 O3'.split(),
                                 doc='APEX opt_level, see https://nvidia.github.io/apex/amp.html#opt-levels'),
                    ],
                    lambda val: val == 'fairseq': [  # similar to apex amp O2 level, with more fine-grained control
                        Argument('memory_efficient_fp16', default=False),
                        Argument('fp16_init_scale', default=2 ** 7),
                        Argument('fp16_scale_tolerance', default=0.0,
                                 doc='pct of updates that can overflow before decreasing the loss scale'),
                        Argument('threshold_loss_scale', type=int, doc='threshold FP16 loss scale from below'),
                    ]
                }),
            ]),
            domain='trainer/fp16'
        )

        options.register(
            Argument('bf16', default=False),
            domain='trainer/bf16',
        )

        # deepspeed config
        options.register(
            Argument('deepspeed_save_dir', required=True),
            Argument('deepspeed_zero_stage', type=int, default=0),
            Argument('deepspeed_bf16', default=False),
            Argument('deepspeed_fp16', default=False),
            domain='deepspeed',
        )

        def check_power_8(args, name):
            if not args.fp16 or not hasattr(args, name):
                return True
            return not args.fp16 or getattr(args, name) % 8 == 0, \
                f'argument "{name}" has to be power of 8 in fp16 training'
        # remove check_power_8 global constraint for batch_size, max_len
        # for param in 'batch_size hidden_size max_len'.split():
        for param in 'hidden_size'.split():
            options.add_global_constraint(
                partial(check_power_8, name=param),
            )
        options.add_global_constraint(
            lambda args: (args.valid_subset in args.data_size,
                          f'valid_subset "{args.valid_subset}" not found')
        )

        options.add_global_constraint(
            lambda args: (not (args.resume and args.overwrite), 'resume and overwrite cannot both be true')
        )
        options.add_global_constraint(
            lambda args: (not args.resume or io.exists(args.save_dir),
                          f'save_dir "{args.save_dir}" does not exist when resuming')
        )
        options.add_global_constraint(
            lambda args: (args.overwrite or args.resume or not cached_exists(args.save_dir),
                          f'save_dir "{args.save_dir}" exists, set "overwrite" to true to overwrite save_dir '
                          f'or set "auto_suffix" to true to avoid collision.')
        )
        options.add_global_constraint(
            lambda args: (args.max_epoch or args.max_update, 'either max_epoch or max_update should be set')
        )
        options.add_global_constraint(
            lambda args: (len(get_train_subsets(args)) > 0, f'train_subset {args.train_subset} not found.')
        )
        options.add_global_constraint(
            lambda args: (not (args.eval_interval == -1 and args.eval_interval_warmup > 0),
                          'cannot use eval_interval_warmup when eval_interval is disabled')
        )

        def check_min_steps(args):
            total_steps = get_total_steps(args, runtime=False)
            return (
                args.min_steps <= total_steps,
                f'min_steps {args.min_steps} cannot be larger than total steps {total_steps}.'
            )
        options.add_global_constraint(check_min_steps)

    def __init__(self, args, task, model, loss, state_dict=None):
        self.args = args
        self.task: Task = task
        self.logger = Logger(args)
        self.save_jobs = {}

        if args.cuda and not torch.cuda.is_available():
            raise RuntimeError(r'CUDA not available when {cuda: true}')
        self.cuda = args.cuda
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self._loss = loss
        self._model = model
        if args.resume:
            # resume before optimizer initialization to ensure correctness in corner cases
            prev_path = parse_model_path('<last>', args)
            with io.open(prev_path, 'rb') as f:
                model_state_dict = torch.load(f, map_location='cpu')
            # self._model.load_state_dict(model_state_dict, strict=True)
            self._model.load(prev_path)
        if args.fp16 and args.fp16_backend == 'fairseq':
            self._model = self._model.half()
            self._loss = self._loss.half()
        # copy model and loss to current device
        self._loss = self._loss.to(device=self.device)
        self._model = self._model.to(device=self.device)
        self.ema = EMA(self._model, args.ema_decay)
        
        # build optimizer
        params = self._model.trainable_parameters()

        if args.fp16:
            if torch.cuda.get_device_capability(0)[0] < 7:
                print(
                    "NOTE: your device does NOT support faster training with fp16, "
                    "please switch to FP32 which is likely to be faster"
                )
            if args.fp16_backend == 'apex':
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        'Please install apex from https://www.github.com/nvidia/apex to use fp16 training.')
                self._optimizer = optimizers[self.args.optimizer](self.args, params)
                apex_params = dict(
                    opt_level=args.opt_level,
                    min_loss_scale=args.min_loss_scale
                )
                if self.args.optimizer != 'pair':
                    self._model, self._optimizer._optimizer = amp.initialize(self._model, self._optimizer.optimizer,
                                                                             **apex_params)
                else:
                    self._model, [self._optimizer._optimizer, self._optimizer._2nd_optimizer] = amp.initialize(
                        self._model, [self._optimizer.optimizer, self._optimizer._2nd_optimizer], **apex_params
                    )
            elif args.fp16_backend == 'fairseq':
                from xdpx.optimizers import FP16Optimizer, MemoryEfficientFP16Optimizer
                if self.args.memory_efficient_fp16:
                    self._optimizer = MemoryEfficientFP16Optimizer.build_optimizer(self.args, params)
                else:
                    self._optimizer = FP16Optimizer.build_optimizer(self.args, params)
            else:
                raise NotImplementedError
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                print('| NOTICE: your device may support faster training with fp16: true')
            self._optimizer = optimizers[self.args.optimizer](self.args, params)
        
        if args.distributed_world_size > 1 and args.use_bmuf:
            self._optimizer = BMUFOptimizer(self.args, self._optimizer)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_schedulers[self.args.lr_scheduler](self.args, self.optimizer)
        self._lr_scheduler.step_update(0)

        if args.resume:
            assert state_dict
            self._loss.load_state_dict(state_dict['loss_state_dict'])
            # self._optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self._lr_scheduler.load_state_dict(state_dict['lr_scheduler_state_dict'])
            self._train_start_i = state_dict.get('train_i')
            if args.fp16 and args.fp16_backend == 'apex':
                from apex import amp
                amp.load_state_dict(state_dict['amp'])
            state = state_dict['trainer']
        else:
            self._train_start_i = -1
            state = {}

        self._num_updates = state.get('num_updates', 0)
        self._epoch = state.get('epoch', 0)
        self._current_score = state.get('current_score', math.inf * (-1. if args.ascending_metric else 1.))
        self._best_score = state.get('best_score', self._current_score)
        self._best_step = state.get('best_step', 0)
        self._prev_best = state.get('best_step', 0)
        self._last_eval_step = state.get('last_eval_step', 0)
        self._wrapped_loss = None
        self._wrapped_model = None
        self._last_save = state.get('num_updates', None)  # last_save is resumed one
        self.start_step = self._num_updates

        self._grad_norm_buf = None
        self._oom_buf = None
        self._inconsistent_steps = 0
        if self.cuda and args.distributed_world_size > 1:
            self._grad_norm_buf = torch.cuda.DoubleTensor(args.distributed_world_size)
            if args.auto_ga:
                self._oom_buf = torch.cuda.LongTensor(args.distributed_world_size)

        ds_config = {
            "train_micro_batch_size_per_gpu": self.args.batch_size,
            "bfloat16": {
                "enabled": self.args.deepspeed_bf16
            },
            "fp16": {
                "enabled": self.args.deepspeed_fp16
            },
            "zero_optimization": {
                "stage": self.args.deepspeed_zero_stage,
                "overlap_comm": False
            },
            "zero_allow_untested_optimizer": self.args.deepspeed_zero_stage>0,
            # consider excuting model parallelism and using torch.utils.checkpoint in nn.modules
            "activation_checkpointing": {
                "partition_activations": True
            }
        }
        self.p_optimizer = self._optimizer
        self._optimizer = self.p_optimizer._optimizer
        self._model, self._optimizer, _, _ = deepspeed.initialize(model=self._model,
                                                                optimizer=self._optimizer,
                                                                config=ds_config)
        if args.resume:
            _, client_sd = self._model.load_checkpoint(args.deepspeed_save_dir, self._num_updates)

    def state_dict(self):
        return dict(
            num_updates=self._num_updates,
            epoch=self._epoch,
            current_score=self._current_score,
            best_score=self._best_score,
            best_step=self._best_step,
            last_eval_step=self._last_eval_step,
        )
    
    @property
    def train_start_i(self):
        start_i, self._train_start_i = self._train_start_i, -1
        return start_i
    
    def set_epoch(self, epoch):
        self._epoch = epoch
        self.task.set_epoch(epoch)
        self.logger.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch
    
    def train_step(self, samples):
        """Do forward, backward and parameter update."""
        self.logger.tik()
        self._set_seed()
        self.model.train()
        self.loss.train()
        self.zero_grad()

        # forward and backward pass
        logging_outputs, sample_size = [], 0
        for i, sample in enumerate(samples):
            sample = self._prepare_sample(sample)

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                return contextlib.ExitStack()
                # if (
                #     self.args.distributed_world_size > 1
                #     and hasattr(self.model, 'no_sync')
                #     and (not hasattr(self.args, 'force_sync') or not self.args.force_sync)
                #     and i < len(samples) - 1
                # ):
                #     return self.model.no_sync()
                # else:
                #     return contextlib.ExitStack()  # dummy contextmanager

            try:
                with maybe_no_sync():
                    # forward and backward
                    loss, sample_size_i, logging_output = self.task.train_step(
                        sample, self.model, self.loss, self.optimizer, self.get_num_updates()
                    )
                    if self.should_check_nan(loss):
                        # re-run the forward and backward pass with hooks attached to print out where it fails
                        with NanDetector(self.model, self.loss):
                            self.task.train_step(
                                sample, self.model, self.loss, self.optimizer, self.get_num_updates()
                            )
                        raise RuntimeError(f'nan detected in forward pass {i}.')
                        # print((f'| nan detected in forward pass {i}.'))
                        # sample_str = str(sample)
                        # print(f'| {sample_str}')
                    del loss
                logging_outputs.append(logging_output)
                sample_size += sample_size_i
                # emptying the CUDA cache after the first step can
                # reduce the chance of OOM
                if self.cuda and self.get_num_updates() == 0:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    self._log_oom(e)
                raise e

        if self.sync_oom():
            # another worker encouters OOM while the current worker does not
            raise self.HandleOOM()

        if torch.is_tensor(sample_size):
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        # gather logging outputs from all replicas
        if self._sync_stats():
            logging_outputs, (sample_size, ) = self._aggregate_logging_outputs(
                logging_outputs, sample_size
            )
        logging_output = self.task.aggregate_logging_outputs(
            logging_outputs, int(sample_size), self.get_loss()
        )

        try:
            # normalize grads by sample size
            if sample_size > 0:
                if self.args.distributed_world_size == 1 or not self.args.use_bmuf:
                    # default behavior is simply averaging gradients between nodes, we need to recompute the average using sample_size
                    self.p_optimizer.multiply_grads(self.args.distributed_world_size / float(sample_size))
                else:  # BMUF needs to check sample size
                    num = self.args.distributed_world_size if self._sync_stats() else 1
                    self.optimizer.multiply_grads(num / sample_size)
            logging_output['sample_size'] = sample_size

            # trainer takes a global step before grad clipping in case a FloatPointError is raised
            self.set_num_updates(self.get_num_updates() + 1)

            # clip grads
            grad_norm = self.p_optimizer.clip_grad_norm(self.args.clip_norm)
            if self.should_check_nan(grad_norm):
                with NanDetector(self.model, self.loss, forward=False):
                    self.task.train_step(
                        sample, self.model, self.loss, self.optimizer, self.get_num_updates()
                    )
                raise RuntimeError('nan or inf detected in backward pass.')
                # print(f'| nan or inf detected in backward pass.')
                # return None, None

            # check that grad norms are consistent across workers
            consistent = True
            # if self.args.distributed_world_size > 1 and not self.args.use_bmuf:
            #     consistent = self._check_grad_norms(grad_norm)

            logging_output['grad_norm'] = float(grad_norm)
            logging_output['lr'] = self.get_lr()
            logging_output['step'] = self.get_num_updates()

            # take an optimization step
            if consistent:
                self.optimizer.step()
                self.ema.update()
                self._inconsistent_steps = 0
            else:
                sys.stderr.write(f'[step {self._num_updates}] gradient inconsistent. Skip optimization.\n')
                self._inconsistent_steps += 1
                if self._inconsistent_steps >= 4:
                    raise RuntimeError('Consecutive inconsistent gradients between workers detected.')

            # task specific update per step
            self.task.update_step(self._num_updates)

            # clear CUDA cache to reduce memory fragmentation
            if (
                self.args.cuda and
                self.args.empty_cache_freq > 0
                and (
                    (self.get_num_updates() + self.args.empty_cache_freq - 1)
                    % self.args.empty_cache_freq
                ) == 0
            ):
                torch.cuda.empty_cache()
        except FloatingPointError:
            # re-run the forward and backward pass with hooks attached to print out where it fails
            print('| NaN gradients:')
            stats = []
            for name, p in self.model.named_parameters():
                grad = p.grad
                if grad is not None:
                    isinf = torch.isinf(grad).any()
                    isnan = torch.isnan(grad).any()
                    if isinf or isnan:
                        stats.append([name,
                                      torch.isinf(grad).sum().item() if isinf else 0,
                                      torch.isnan(grad).sum().item() if isnan else 0,
                                      str(p.size())])
            print('| \n' + pformat_dataframe(pd.DataFrame(stats, columns=['param', 'inf', 'nan', 'size'])))
            print('| sample: {}'.format(sample))

            with NanDetector(self.model, self.loss):
                self.task.train_step(
                    sample, self.model, self.loss, self.optimizer, self.get_num_updates(),
                )
            raise
        except OverflowError as e:
            print('NOTE: overflow detected, ' + str(e))
            self.zero_grad()
            logging_output = None
        except RuntimeError as e:
            if 'out of memory' in str(e):
                self._log_oom(e)
                print('| OOM during optimization, irrecoverable')
            raise e

        if logging_output:
            if self.args.fp16:
                if self.args.fp16_backend == 'fairseq':
                    logging_output['loss_scale'] = self.optimizer.scaler.loss_scale
                else:
                    from apex import amp
                    logging_output['loss_scale'] = amp.state_dict()['loss_scaler0']['loss_scale']
            summary = self.logger.update(logging_output)
        else:
            summary = None
        return logging_output, summary

    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():

            sample = self._prepare_sample(sample)

            try:
                _loss, sample_size, logging_output = self.task.valid_step(
                    sample, self.model, self.loss
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if not raise_oom:
                        print('| ran out of memory in validation step, retrying batch')
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = None  # free some memory
                        if self.cuda:
                            torch.cuda.empty_cache()
                        return self.valid_step(sample, raise_oom=True)
                raise e
                
            logging_outputs = [logging_output]

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_outputs, (sample_size, ) = self._aggregate_logging_outputs(
                logging_outputs, sample_size,
            )

        return logging_outputs, sample_size

    def validate(self, data):
        self._last_eval_step = self._num_updates
        logging_outputs, sample_size = [], 0
        for i, sample in enumerate(tqdm(data, desc='evaluate', leave=False)):
            if self.args.max_eval_steps and i >= self.args.max_eval_steps:
                break
            logging_output, sample_size_i = self.valid_step(sample)
            logging_outputs.extend(logging_output)
            sample_size += sample_size_i
        logging_output = self.task.aggregate_logging_outputs(
            logging_outputs, int(sample_size), self.get_loss(), 
            max_count=self.args.data_size[self.args.valid_subset]
        )
        summary, score = self.logger.summarize(self.get_num_updates(), logging_output)
        self._current_score = score
        if (score > self.best_score) == self.args.ascending_metric:
            self.best_score = score
        return summary, score
    
    save_prefix = 'checkpoint'

    def reach_train_steps(self) -> bool:
        return self.args.train_steps and self._num_updates >= self.start_step + self.args.train_steps

    def reach_last_step(self, i: int) -> bool:
        reach_max_update = self.args.max_update and self._num_updates >= self.args.max_update
        reach_max_epoch = (self.args.max_epoch and self.epoch == self.args.max_epoch - 1 and
                           i >= (len(self.task.data_loaders[self.task.TRAIN]) // self.args.update_freq
                                 * self.args.update_freq - self.args.update_freq))
        reach_train_steps = self.reach_train_steps()
        return reach_max_update or reach_max_epoch or reach_train_steps

    def should_eval(self) -> bool:
        if self.args.eval_interval == -1:
            return False
        if self._num_updates > self.args.eval_interval_warmup:
            return self._num_updates % self.args.eval_interval == 0
        else:
            return self._num_updates % (self.args.eval_interval_warmup_mutiplier * self.args.eval_interval) == 0

    def should_save(self, i) -> bool:
        if not self.args.save:
            return False
        if self.args.save_last_only and self.reach_last_step(i):
            return True
        if self.args.save_above_score is not None and \
                (self._current_score >= self.args.save_above_score) != self.args.ascending_metric:
            return False
        if self.args.save_best_only:
            return self._num_updates == self.best_step
        return not self.args.save_last_only

    def should_check_nan(self, indicator):
        if torch.is_tensor(indicator):
            # in a forward pass, loss as a tensor is provided
            has_nan = torch.isnan(indicator) or torch.isinf(indicator)
        else:
            has_nan = math.isnan(indicator) or math.isinf(indicator)
            if has_nan and self.args.fp16:
                # if nan is encountered in backward passes, ignore them in FP16 training before the
                # loss_scale reach its limit
                assert self.args.fp16_backend == 'apex'
                from apex import amp
                loss_scale = amp.state_dict()['loss_scaler0']['loss_scale']
                reach_scale_limit = loss_scale / 2 < self.args.min_loss_scale
                return reach_scale_limit
        return has_nan

    def reach_tolerance(self):
        return self.args.tolerance \
               and self._num_updates > self.args.eval_interval_warmup \
               and self._num_updates >= self.args.min_steps \
               and self._num_updates - self.best_step > self.args.tolerance

    @staticmethod
    def save_pattern(num_updates):
        return f'{Trainer.save_prefix}-{num_updates}.pt'

    @staticmethod
    def get_step_from_path(path):
        return int(re.search(rf'{Trainer.save_prefix}-(\d+).pt', path).group(1))

    def save_checkpoint(self, i):
        num_updates = self.get_num_updates()
        if not self.should_save(i):
            return
        if distributed_utils.is_master(self.args):
            cast_precision = False
            if not self.args.fp16 or self.args.fp16_backend == 'apex':
                states = self.get_model().state_dict()
            else:
                cast_precision = True
                fp32_model = self._model.float()
                params = fp32_model.trainable_parameters()
                if self.args.optimizer != 'pair':
                    params = self._optimizer.combine_param_groups(params)
                else:
                    major_params, second_params = params
                    params = self._optimizer.combine_param_groups(major_params) + \
                        self._optimizer.combine_param_groups(second_params)
                assert len(params) == len(self._optimizer.fp32_params)
                for p, p32 in zip(params, self._optimizer.fp32_params):
                    p.data.copy_(p32.data)
                states = fp32_model.state_dict()

            filename = os.path.join(self.args.save_dir, self.save_pattern(num_updates))
            cache = cache_file(filename, dry=True, clear_cache=True)
            try:
                with io.open(cache, 'wb') as f:
                    torch.save(states, f)
                if self.args.save_full_checkpoint:
                    train_states = {
                        'args': vars(self.args),
                        'train_i': i,
                        'trainer': self.state_dict(),
                        'loss_state_dict': self.get_loss().state_dict(),
                        # 'optimizer_state_dict': self.optimizer.state_dict(),  # this item will be large for optims like Adam
                        'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                    }
                    if self.args.fp16 and self.args.fp16_backend == 'apex':
                        from apex import amp
                        train_states['amp'] = amp.state_dict()
                    # postfix "pts" stands for "PyTorch State"
                    with io.open(cache + 's', 'wb') as f:
                        torch.save(train_states, f)
            except RuntimeError as e:
                if 'No space left on device' in str(e):
                    io.rmtree(default_cache_root)
                    return self.save_checkpoint(i)
                raise e

            if cast_precision:
                self._model.half()

            if cache != filename:
                self.save_jobs[num_updates] = upload_models(cache, filename)
                if self.args.save_full_checkpoint:
                    assert num_updates != 0
                    self.save_jobs[-num_updates] = upload_models(cache + 's', filename + 's')
            if self._prev_best and self.args.save_best_only and num_updates == self.best_step:
                self.remove_checkpoint(self._prev_best)
            if self._last_save is not None and self.args.save_last_only and \
                    not (self.args.save_best_only and self._last_save == self.best_step):
                self.remove_checkpoint(self._last_save)
            self._last_save = num_updates

        # save deepspeed model and optimizer
        if self.args.save_full_checkpoint:
            deepspeed_save_dir = self.args.deepspeed_save_dir
            if not io.exists(deepspeed_save_dir):
                io.makedirs(deepspeed_save_dir)
            self._model.save_checkpoint(deepspeed_save_dir,num_updates)

    def remove_checkpoint(self, num_updates):
        filename = os.path.join(self.args.save_dir, self.save_pattern(num_updates))

        def _remove_checkpoint(path, _id):
            if _id in self.save_jobs and self.save_jobs[_id].is_alive():
                self.save_jobs[_id].terminate()
            elif io.exists(path):
                io.remove(path)

        _remove_checkpoint(filename, num_updates)
        # remove full states as well
        _remove_checkpoint(filename + 's', -num_updates)
    
    def cleanup(self, interrupted):
        for job in self.save_jobs.values():
            if job.is_alive():
                print('| Model still saving...Wait for completion.')
                job.join()
        if self.args.cuda:
            from packaging import version
            if version.parse(torch.__version__) >= version.parse('1.1'):
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.empty_cache()
            if self.args.distributed_world_size > 1:
                torch.distributed.destroy_process_group()
                if 'VISIBLE_DEVICE_LIST' in os.environ:
                    # make sure all the workers have exited before starting a new one
                    # https://work.aone.alibaba-inc.com/issue/30314064
                    time.sleep(5)
        delayed_flush(0)
        distributed_utils.restore_output()
        if distributed_utils.is_master(self.args):
            if self.args.save and not interrupted and (
                self.args.save_above_score is None or
                (self.best_score > self.args.save_above_score) == self.args.ascending_metric
            ):
                if not io.listdir(self.args.save_dir, contains=self.save_prefix):
                    raise RuntimeError('Fail to save checkpoint.')
    
    def _aggregate_logging_outputs(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        if self.get_loss().logging_outputs_can_be_summed():
            logging_outputs, extra_stats_to_sum = self._fast_stat_sync_sum(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )
        else:
            logging_outputs, extra_stats_to_sum = self._all_gather_list_sync(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )
        return convert_to_native(logging_outputs), convert_to_native(extra_stats_to_sum)
    
    def _sync_stats(self):
        # Return True if it's using multiple GPUs and DDP or multiple GPUs with
        # BMUF and it's a bmuf sync with warmup iterations completed before.
        return self.args.distributed_world_size > 1 and (
            (not self.args.use_bmuf)
            or (
                self.args.use_bmuf
                and (self.get_num_updates() + 1) % self.args.global_sync_iter == 0
                and (self.get_num_updates() + 1) > self.args.warmup_iterations
            )
        )
    
    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suitable when logging outputs are complex types.
        """
        if ignore:
            logging_outputs = []
        results = list(zip(
            *distributed_utils.all_gather_list(
                [logging_outputs] + list(extra_stats_to_sum),
                max_size=getattr(self.args, 'all_gather_list_size', 16384 * 16),
                group=self.data_parallel_process_group,
            )
        ))
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return logging_outputs, extra_stats_to_sum

    def _fast_stat_sync_sum(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. fast_stat_sync_sum is
        faster than all_gather_list_sync, but is only suitable when
        logging outputs are scalars and can be summed. Note that
        *logging_outputs* cannot contain any nested dicts/lists.
        """
        data = {}
        for i, stat in enumerate(extra_stats_to_sum):
            data['extra_stats_' + str(i)] = stat
        if len(logging_outputs) > 0:
            log_keys = list(logging_outputs[0].keys())
            for k in log_keys:
                if not ignore:
                    v = sum(log[k] for log in logging_outputs if k in log)
                else:
                    v = logging_outputs[0][k]
                    v = torch.zeros_like(v) if torch.is_tensor(v) else 0
                data['logging_outputs_' + k] = v
        else:
            log_keys = None

        data = distributed_utils.all_reduce_dict(
            data,
            device=self.device,
            group=self.data_parallel_process_group
        )

        extra_stats_to_sum = [
            data['extra_stats_' + str(i)] for i in range(len(extra_stats_to_sum))
        ]
        if log_keys is not None:
            logging_outputs = [{k: data['logging_outputs_' + k] for k in log_keys}]
        else:
            logging_outputs = []
        return logging_outputs, extra_stats_to_sum
    
    def _check_grad_norms(self, grad_norm):
        """Check that grad norms are consistent across workers."""
        if self._grad_norm_buf is not None and not self.args.use_bmuf:
            self._grad_norm_buf.zero_()
            self._grad_norm_buf[self.args.distributed_rank] = grad_norm
            distributed_utils.all_reduce(self._grad_norm_buf, group=self.data_parallel_process_group)
            if not (self._grad_norm_buf == self._grad_norm_buf[0]).all():
                if self.args.fp16 and (torch.isnan(self._grad_norm_buf).any() or torch.isinf(self._grad_norm_buf).any()):
                    return True
                print(
                    'Fatal error: gradients are inconsistent between workers. '
                    'Try {ddp_backend: "no_c10d", force_sync: "true"}.\n'
                    + str(self._grad_norm_buf.cpu().tolist())
                )
                torch.cuda.empty_cache()
                return False
        return True
    
    def sync_oom(self, oom=False):
        if self._oom_buf is None:
            return
        # synchronize OOM handling between workers
        self._oom_buf.zero_()
        self._oom_buf[self.args.distributed_rank] = int(oom)
        distributed_utils.all_reduce(self._oom_buf, group=self.data_parallel_process_group)
        if self._oom_buf.max().item():
            return True
        
    class HandleOOM(Exception): ...

    def _log_oom(self, exc):
        print(f"| OOM: Ran out of memory with exception: {exc}")
        if self.args.cuda and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                print(torch.cuda.memory_summary(device=device_idx))

    @property
    def data_parallel_process_group(self):
        return None

    @property
    def best_score(self):
        return self._best_score
    
    @property
    def best_step(self):
        return self._best_step
    
    @property
    def last_eval_step(self):
        return self._last_eval_step
    
    @best_score.setter
    def best_score(self, value):
        self._best_score = value
        self._prev_best, self._best_step = self._best_step, self.get_num_updates()

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        if self.args.seed:
            seed = self.args.seed + self.get_num_updates()
            torch.manual_seed(seed)
            if self.cuda:
                torch.cuda.manual_seed(seed)
    
    def _prepare_sample(self, sample):
        if self.cuda:
            sample = utils.move_to_cuda(sample)

            if self.args.fp16 and self.args.fp16_backend == 'fairseq':
                sample = utils.cast_to_half(sample)

        return sample
    
    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_loss(self):
        """Get the (non-wrapped) loss instance."""
        return self._loss

    def get_meter(self, name):
        """Get a specific meter by name."""
        if name not in self.meters:
            return None
        return self.meters[name]
    
    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()

    @property
    def loss(self):
        if self._wrapped_loss is None:
            if (
                utils.has_parameters(self._loss)
                and self.args.distributed_world_size > 1
                and not self.args.use_bmuf
            ):
                if self.args.fp16 and self.args.fp16_backend == 'apex':
                    from apex.parallel import DistributedDataParallel
                    self._wrapped_loss = DistributedDataParallel(
                        self._loss, delay_allreduce=True
                    )
                else:
                    from .models import DistributedDataParallel
                    self._wrapped_loss = DistributedDataParallel(
                        self.args, self._loss,
                        process_group=self.data_parallel_process_group,
                )
            else:
                self._wrapped_loss = self._loss
        return self._wrapped_loss

    @property
    def model(self):
        if self._wrapped_model is None:
            self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        return self._optimizer
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    @property
    def lr_scheduler(self):
        return self._lr_scheduler
    
    def get_lr(self):
        """Get the current learning rate."""
        return self.p_optimizer.get_lr()
    
    def lr_step_update(self):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(self.get_num_updates())


def upload_models(src, dst):
    assert io.islocal(src)
    assert not io.islocal(dst)
    p = Process(target=_upload_models, args=(src, dst))
    p.start()
    return p


def _upload_models(src, dst):
    from xdpx.utils import import_user_module, io
    try:
        io = import_user_module(reload=True)['oss_credentials'].io
    except KeyError:
        pass
    print(f'saving model to {dst}')
    persistent_run(io.copy, src, dst)


@lru_cache(maxsize=8)
def cached_exists(path):
    return io.exists(path)
