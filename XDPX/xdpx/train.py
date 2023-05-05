"""
Train a new model on one or across multiple GPUs.
"""

import os
import sys
import math
import time
import itertools
import psutil
import traceback
import subprocess
import hjson
import torch
from packaging import version

import xdpx
from xdpx.trainer import Trainer
from xdpx.utils import (
    io, distributed_utils, current_time, cache_file, diff_params, format_time_span,
    log_to_file, compress_dir, get_total_steps, parse_model_path, get_train_subsets
)
from xdpx.utils.versions import torch_ne_13, torch_lt_120
from xdpx.options import Options, Arg
from xdpx.tasks import tasks
from xdpx.visualize import main as visualize_cli
from xdpx.bootstrap import bootstrap
from xdpx.utils.summary import summary_string


def main(args, init_distributed=False):
    # import user module for each process
    check_prepro_version(args)

    # Initialize CUDA and distributed training
    if args.cuda:
        torch.cuda.set_device(args.device_id)
        print(f'| CUDA is set to device {args.device_id}')
    if args.seed:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    state_dict = prev_step = None
    if args.resume:
        prev_path = parse_model_path('<last>', args)
        prev_step = Trainer.get_step_from_path(prev_path)
        print(f'| resume training from {prev_path}')
        with io.open(prev_path + 's', 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')

    # Setup task
    task = tasks[args.task](args)

    if distributed_utils.is_master(args):
        # restore arguments before save_dir creation
        options = build_train_options()
        if not args.resume:
            # batch arguments are parsed at the beginning of program execution;
            # update save_dir here to avoid collision with other independent training tasks with auto_suffix
            args.save_dir = Trainer.auto_save_dir(args.save_dir, args)
        arguments = options.tree(args)
        # create save_dir
        if args.overwrite and io.exists(args.save_dir):
            print(f'| WARNING: Overwrite existed save_dir: {args.save_dir}')
            time.sleep(1)
            io.remove(args.save_dir)
        if not args.resume:
            io.makedirs(args.save_dir, exist_ok=False)

        # save training args
        args_save_name = 'args'
        starter_save_name = 'starter_config'
        should_save_args = True
        if args.resume:
            prev_args = state_dict['args']
            match = compare_args(prev_args,
                                 args.change(batch_size=max(1, args.batch_size // args.distributed_world_size)))
            if not match:
                args_save_name += f'.resume-{prev_step}'
                starter_save_name += f'.resume-{prev_step}'
            else:
                should_save_args = False
        if should_save_args:
            with io.open(os.path.join(args.save_dir, f'{args_save_name}.py'), 'w') as f:
                f.write(arguments)
            with io.open(os.path.join(args.save_dir, f'{starter_save_name}.hjson'), 'w') as f:
                hjson.dump(Options.parse_starter_config(arguments), f)

        # backup resources in data_dir
        for resource in task.processor.resources:
            src = os.path.join(args.data_dir, resource)
            if io.exists(src):
                io.copy(src, os.path.join(args.save_dir, resource))

        # backup current code for reproducibility
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        backup_path = os.path.join(args.save_dir, 'snapshot.zip')
        max_size = 100 * 1024 ** 2  # 100M
        errmsg = f'"{os.path.dirname(__file__)}" has exceeded 100M;\nPlease clean the XDPX directory to enable ' \
                 'fast & stable code backup.'
        if not args.resume:
            compress_dir(project_root, backup_path, max_size=max_size, errmsg=errmsg)
        else:
            new_backup_path = os.path.join(args.save_dir, f'snapshot.resume-{prev_step}.zip')
            cached_backup_path = cache_file(new_backup_path, dry=True)
            md5 = compress_dir(project_root, cached_backup_path, max_size=max_size, errmsg=errmsg)
            prev_md5 = io.md5(backup_path)
            if md5 != prev_md5:
                io.move(cached_backup_path, new_backup_path)
            else:
                io.remove(cached_backup_path)
        if args.log_file:
            log_to_file(os.path.join(args.save_dir, args.log_file))
        show_git_info()

    print(f'| PyTorch version: {torch.__version__}')
    if hasattr(args, '__dist_config__'):
        print('| Distributed configuration:', args.__dist_config__)
    distributed_utils.show_dist_info()
    # get the batch size for each device
    args.batch_size = max(1, args.batch_size // args.distributed_world_size)

    # Build model and loss
    model = task.build_model(args)
    loss = task.build_loss(args)
    print(f'| model {model.__class__.__name__}, loss {loss.__class__.__name__}')
    print('| num. model params: {:,} (num. trained: {:,})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, loss, state_dict)
    print('| training on ' + ((str(args.distributed_world_size) + ' GPU' +
                               ('s' if args.distributed_world_size > 1 else ''))
                              if args.cuda else 'CPUs'))
    total_steps = get_total_steps(args, runtime=True)
    if not args.train_steps:
        print(f'| training with maximum {total_steps} steps.')
    else:
        print(f'| training with {args.train_steps} steps (maximum {total_steps} steps).')
    if args.cuda:
        print(f'| GPU Memory consumed by model: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB')
        #summary_model = summary_string(model, dummy_inputs=trainer._prepare_sample(model.dummy_inputs))[0]
        #print(f'| Summary of model:')
        #print(summary_model)

    # load dataset
    task.load_dataset(args.valid_subset, is_train=False)
    if not args.lazy_load:
        task.load_dataset(get_train_subsets(args), is_train=True)

    # Train until the learning rate gets too small
    start_epoch = trainer.epoch
    max_epoch = args.max_epoch or math.inf
    exit_msg = 'done training in'
    interrupted = False
    start_time = time.time()

    try:
        for epoch in itertools.count(start_epoch):
            if epoch + 1 > max_epoch:
                break
            print(f'Epoch {epoch + 1}')
            trainer.set_epoch(epoch)
            train(args, trainer, task, start_i=trainer.train_start_i)
    except EarlyStop as e:
        print('| ' + str(e))
        if 'maximum updates' in str(e) and trainer.reach_train_steps():
            exit_msg = 'pause training after'
    except KeyboardInterrupt:
        interrupted = True
        print('| KeyboardInterrupt')
    end_time = time.time()
    print(f'| {exit_msg} {format_time_span(start_time, end_time)}')
    print(f'| best validation {args.major_metric}: {trainer.best_score} at step {trainer.best_step}')
    trainer.cleanup(interrupted)


class Progress:
    def __init__(self, args):
        self.stream = distributed_utils.single_stream()
        self.total = get_total_steps(args, runtime=True)

    def write(self, msg, i=None):
        if not msg:
            return
        if isinstance(msg, dict):
            msg = ', '.join(f'{key}={val}' for key, val in msg.items())
            msg = f'[{msg}]'
        if i is not None:
            msg = f'({i/self.total*100:.1f}%) {i}/{self.total} ' + msg
        self.stream.write(current_time() + ' ' + msg + '\n')


def train(args, trainer, task, start_i=-1):
    """Train the model for one epoch."""
    data = task.data_loaders[task.TRAIN]
    data_iter = iter(data)
    progress = Progress(args)

    i = 0
    unstable = 0
    while i < len(data):
        if i <= start_i:
            if torch_lt_120():
                next(data_iter.sample_iter)
            else:
                data_iter._next_index()
            i += 1
            continue
        if i > len(data) - args.update_freq:
            # the last batch is dropped if its data are not enough
            break

        handle_oom = False
        log_output = summary = None
        try:
            log_output, summary = trainer.train_step(LazyBatches(data_iter, args.update_freq))
        except RuntimeError as e:
            # OOM handling must be outside the try-except block otherwise the GPU memory will not be released
            handle_oom = 'out of memory' in str(e) and args.auto_ga and args.batch_size > 1
            if not handle_oom:
                raise e
            # send oom signal to other processes
            trainer.sync_oom(True)
        except trainer.HandleOOM:
            # receives oom signal from another process
            handle_oom = True

        if handle_oom:
            args.batch_size //= 2
            args.update_freq *= 2
            print(f'| Recover from OOM in forward/backward pass...'
                  f'reduce batch_size to {args.batch_size} with update_freq {args.update_freq}')
            # clear cache manually here to avoid RuntimeError: too many resources requested for launch
            torch.cuda.empty_cache()
            trainer.zero_grad()
            task.set_epoch(trainer.epoch, reload=True)  # reload train set with or without lazy load
            task.load_dataset(args.valid_subset, is_train=False, reload=True)
            assert task.data_loaders[task.TRAIN].batch_size == args.batch_size
            return train(args, trainer, task, start_i=max(i + 1 - args.update_freq, 0) * 2 - 1)

        num_updates = trainer.get_num_updates()
        if (trainer.epoch == 0 or start_i > 0) and i - start_i == args.update_freq * args.log_interval + 1:
            show_mem_stats(args.cuda)

        if num_updates % args.log_interval == 0:
            progress.write(summary, num_updates)
        if log_output and log_output['loss'] > args.max_loss:
            unstable += 1
            if unstable >= 3:
                raise EarlyStop(f'loss {log_output["loss"]} exceeds max_loss, stopping unstable training')
        else:
            unstable = 0

        if trainer.should_eval():
            summary, _ = validate(args, trainer, task, i)
            progress.write(summary)
            if trainer.reach_tolerance():
                raise EarlyStop('training reaches tolerance steps')

        if trainer.reach_last_step(i):
            # evaluate for the last time when training endings
            if trainer.last_eval_step < num_updates:
                summary, _ = validate(args, trainer, task, i)
                progress.write(summary)
            raise EarlyStop('maximum updates reached')

        if trainer.get_lr() <= args.min_lr:
            raise EarlyStop('stop training because lr drops below min_lr')

        i += args.update_freq


def validate(args, trainer, task, i):
    """Evaluate the model on the validation set(s) and return the best validation results."""
    data = task.data_loaders[task.VALID]
    trainer.ema.apply_shadow()
    summary, score = trainer.validate(data)
    trainer.save_checkpoint(i)
    visualize(args)
    trainer.ema.restore()
    return summary, score


def visualize(args):
    if distributed_utils.is_master(args):
        try:
            visualize_cli(Arg(save_dir=args.save_dir, ref_dir=args.viz_ref_dir, figext=args.figext,
                              walltime=False, label=None))
        except Exception:
            # In any time, skip visualization errors because it can be easily rerun with x-viz
            print(traceback.format_exc())
            print('Error encountered. Skip visualization.')


class EarlyStop(Exception):
    pass


class LazyBatches:
    def __init__(self, iterator, count):
        self.iterator = iterator
        self.count = count

    def __len__(self):
        return self.count

    def __iter__(self):
        for _ in range(self.count):
            yield next(self.iterator)


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    os.environ["LOCAL_RANK"] = str(i)
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def build_train_options():
    options = Options()
    Trainer.register(options)
    return options


def check_prepro_version(args):
    with io.open(os.path.join(args.data_dir, 'meta.hjson')) as f:
        cv = version.parse(xdpx.__version__)
        pv = ''
        meta = hjson.load(f)
        match = '__version__' in meta
        if match:
            pv = version.parse(meta['__version__'])
            if (pv.major, pv.minor) != (cv.major, cv.minor):
                match = False
        if not match:
            msg = f'''
            ============================= WARNING ==========================================
            Prepro version {pv} does not match the version in training {cv}
            Consider rerun preprocessing or modify meta.hjson if any conflict occurs.
            ================================================================================
            '''
            sys.stderr.write(msg + '\n')


def show_mem_stats(cuda):
    mem_used = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    print(f'| Memory consumed in training: {mem_used:.0f} MB.', flush=True)
    if cuda:
        print(f'| GPU Memory consumed in training: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB',
              flush=True)


def show_git_info():
    info = subprocess.run('git rev-parse HEAD && git status', shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8').stdout.rstrip()
    if not info and os.path.exists('.git_version'):
        with open('.git_version') as f:
            info = f.read()
    if info:
        if 'working tree clean' in info:
            info = info.split('\n')[0]
        print('| Git commit: ' + info)


def compare_args(prev_args, args):
    """compare args before & after resuming"""
    diff = diff_params(prev_args, args, exclude=Arg.__exclude__)
    print('| compare args before & after resuming:')
    if not diff:
        print('|   No difference.')
        return True
    for line in diff:
        print('|   ' + line)
    return False


def cli_main(argv=sys.argv):
    def entry(args):
        if args.cuda and args.distributed_init_method and args.distributed_init_method.startswith('env'):
            # distributed training
            device_count = torch.cuda.device_count()
            if torch_ne_13() and device_count > 1:
                # classic multi-worker + multi-GPU training
                start_rank = args.distributed_rank * device_count
                args.distributed_rank = None  # assign automatically
                torch.multiprocessing.spawn(
                    fn=distributed_main,
                    args=(args, start_rank),
                    nprocs=device_count,
                )
            else:
                if device_count > 1:
                    # multi-worker + multi-GPU training with docker fusion
                    # https://www.atatech.org/articles/171093
                    distributed_main(int(os.environ['VISIBLE_DEVICE_LIST']), args)
                else:
                    # distributed training with only one GPU on each machine
                    distributed_main(0, args)
        elif args.distributed_world_size > 1:
            # fallback for single node with multiple GPUs
            assert args.distributed_init_method.startswith('tcp')
            args.distributed_rank = None  # set based on device id
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args,),
                nprocs=args.distributed_world_size,
            )
        else:
            # single GPU or CPU training
            args.device_id = 0
            main(args)

    # build_train_options: register args through Trainer.register(options)
    # bootstrap: init args value
    bootstrap(build_train_options(), entry_func=entry, entry_file=__file__, argv=argv)


if __name__ == '__main__':
    cli_main()
