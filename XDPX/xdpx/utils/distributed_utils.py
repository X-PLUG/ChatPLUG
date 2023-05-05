import os
import sys
import pickle
import socket
import struct
import warnings
import psutil
from collections import OrderedDict
from typing import Mapping, Dict, Any
from contextlib import contextmanager
from packaging import version

import torch
import torch.distributed as dist
import builtins as __builtin__
builtin_print = __builtin__.print


def is_master(args=None):
    if args is not None:
        return hasattr(args, 'distributed_rank') and args.distributed_rank == 0
    return not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)


def distributed_init(args):
    if dist.is_initialized():
        warnings.warn('Distributed is already initialized, cannot initialize twice!')
    else:
        print('| distributed init (rank {}/{}): {}'.format(
            args.distributed_rank, args.distributed_world_size - 1, args.distributed_init_method), flush=True)

        dist.init_process_group(
            backend=args.distributed_backend,
            init_method=args.distributed_init_method,
            world_size=args.distributed_world_size,
            rank=args.distributed_rank,
        )
        print('| initialized host {} rank {}'.format(
            socket.gethostname(), args.distributed_rank), flush=True)

        # perform an all-reduce to initialize the NCCL communicator
        device_ids = torch.zeros(args.distributed_world_size).cuda()
        device_ids[args.distributed_rank] = args.device_id
        dist.all_reduce(device_ids)
        dist_config = get_dist_config(device_ids)
        args.__dist_config__ = dist_config

        suppress_output(has_exclusive_output())
    
    if args.distributed_rank != dist.get_rank():
        print('WARNING: rank mismatch', args.distributed_rank, dist.get_rank(), force=True, flush=True)
    args.distributed_rank = dist.get_rank()
    return args.distributed_rank


def suppress_output(is_master):
    """Suppress printing on the current device. Force printing with `force=True`."""
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def restore_output():
    __builtin__.print = builtin_print


def should_barrier():
    # get_world_size() should be used instead of torch.cuda.device_count() here when docker fusion is on.
    # For example, when there're 6 GPUs [4+1+1], workers with a single GPU should pass the barrier as well
    # otherwise workers on the 4-GPU machine will wait forever.
    return torch.distributed.is_available() and torch.distributed.is_initialized() and get_world_size() > 1


def is_worker_master(barrier=None):
    if barrier is None:
        barrier = should_barrier()
    return not barrier or torch.cuda.current_device() == 0


def has_exclusive_output():
    worker_master = is_worker_master()
    docker_fusion = version.parse(torch.__version__) > version.parse('1.2') and 'VISIBLE_DEVICE_LIST' in os.environ
    return worker_master or docker_fusion


@contextmanager
def worker_master_first():
    """
    This function is not considering the global master (dist_rank == 0). Instead, it's for local file cache order,
    so device_id == 0 is the criterion.
    """
    barrier = should_barrier()
    is_master = is_worker_master(barrier)
    if barrier and not is_master:
        dist.barrier()
    yield
    if barrier and is_master:
        dist.barrier()


def show_dist_info(*args, **kwargs):
    mem_info = psutil.virtual_memory()
    print(f'| Total Machine Memory: {mem_info.total / 1024**2:.0f}MB (Available: {mem_info.available/ 1034**2:.0f}MB)')
    if torch.cuda.is_available():
        from pynvml import (
            nvmlInit, nvmlSystemGetDriverVersion, nvmlDeviceGetCount, 
            nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo,
        )
        nvmlInit()
        print('| Nvidia Driver Version:', nvmlSystemGetDriverVersion().decode())
        print('| CUDA Version: ', torch.version.cuda)
        print('| CUDNN Version: ', torch.backends.cudnn.version())
        deviceCount = nvmlDeviceGetCount()
        available_devices = map(int, os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(deviceCount)))).split(','))
        for i, device_id in enumerate(available_devices):
            handle = nvmlDeviceGetHandleByIndex(device_id)
            print(f'| Device {i}:', nvmlDeviceGetName(handle).decode())
            print(f'| \tDevice Capability: {".".join(map(str, torch.cuda.get_device_capability(i)))}')
            print(f'| \tTotal Memory: {nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024}MB')


def get_dist_config(device_ids, name=None):
    if torch.is_tensor(device_ids):
        device_ids = device_ids.cpu().tolist()
    device_ids = list(map(int, device_ids))
    dist_config = []
    while device_ids:
        max_id = max(device_ids)
        dist_config.append(max_id + 1)
        for id_ in range(max_id, -1, -1):
            index = device_ids.index(id_)
            del device_ids[index]
    if not name:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        name = nvmlDeviceGetName(handle).decode()
    msg = '+'.join(map(str, dist_config))
    if len(dist_config) > 1:
        msg = f'({msg})'
    return f'{name}*{msg}'


class NullStream:
    def write(self, *args, **kwargs):
        pass
    def flush(self):
        pass


class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)


def single_stream(stream=sys.stderr):
    if has_exclusive_output():
        return Unbuffered(stream)
    else:
        return NullStream()


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def get_default_group():
    return dist.group.WORLD


def all_reduce(tensor, group=None):
    if group is None:
        group = get_default_group()
    return dist.all_reduce(tensor, group=group)


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    from . import move_to_cpu

    rank = get_rank()
    world_size = get_world_size()

    buffer_size = max_size * world_size
    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()
    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    data = move_to_cpu(data)
    enc = pickle.dumps(data)
    enc_size = len(enc)
    header_size = 4  # size of header that contains the length of the encoded data
    size = header_size + enc_size
    if size > max_size:
        raise ValueError('encoded data size ({}) exceeds max_size ({})'.format(size, max_size))

    header = struct.pack(">I", enc_size)
    cpu_buffer[:size] = torch.ByteTensor(list(header + enc))
    start = rank * max_size
    buffer[start:start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    buffer = buffer.cpu()
    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size:(i + 1) * max_size]
            enc_size, = struct.unpack(">I", bytes(out_buffer[:header_size].tolist()))
            if enc_size > 0:
                result.append(pickle.loads(bytes(out_buffer[header_size:header_size + enc_size].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data. '
            'Try rerunning with --ddp-backend=no_c10d and see if that helps.'
        )

def all_reduce_dict(
    data: Mapping[str, Any],
    device,
    group=None,
) -> Dict[str, Any]:
    """
    AllReduce a dictionary of values across workers. We separately
    reduce items that are already on the device and items on CPU for
    better performance.

    Args:
        data (Mapping[str, Any]): dictionary of data to all-reduce, but
            cannot be a nested dictionary
        device (torch.device): device for the reduction
        group (optional): group of the collective
    """
    data_keys = list(data.keys())

    # We want to separately reduce items that are already on the
    # device and items on CPU for performance reasons.
    cpu_data = OrderedDict()
    device_data = OrderedDict()
    for k in data_keys:
        t = data[k]
        if not torch.is_tensor(t):
            cpu_data[k] = torch.tensor(t, dtype=torch.double)
        elif t.device.type != device.type:
            cpu_data[k] = t.to(dtype=torch.double)
        else:
            device_data[k] = t.to(dtype=torch.double)

    def _all_reduce_dict(data: OrderedDict):
        if len(data) == 0:
            return data
        buf = torch.stack(list(data.values())).to(device=device)
        all_reduce(buf, group=group)
        return {k: buf[i] for i, k in enumerate(data)}

    cpu_data = _all_reduce_dict(cpu_data)
    device_data = _all_reduce_dict(device_data)

    def get_from_stack(key):
        if key in cpu_data:
            return cpu_data[key]
        elif key in device_data:
            return device_data[key]
        raise KeyError

    return OrderedDict([(key, get_from_stack(key)) for key in data_keys])
