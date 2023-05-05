import time
import torch
import torch.nn as nn
from contextlib import contextmanager


@contextmanager
def measure_time(module, flag):
    if tok_hook not in module._forward_hooks.values():
        yield
        return
    flag = '__' + flag
    if not hasattr(module, flag):
        setattr(module, flag, nn.Module())
    flag = getattr(module, flag)

    if not hasattr(flag, 'start_time'):
        flag.start_time = []
    if not hasattr(flag, 'elapsed_time_ms'):
        flag.elapsed_time_ms = []
    cuda = is_cuda(module)
    if cuda:
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    flag.start_time.append(start_time)
    yield
    if cuda:
        torch.cuda.synchronize()
    flag.elapsed_time_ms.append((time.perf_counter() - start_time) * 1000)
    assert len(flag.elapsed_time_ms) == len(flag.start_time)


@contextmanager
def measure_cuda_stream(callback):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    yield
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    callback(elapsed_time_ms)


@contextmanager
def measure_runtime(callback):
    start_time = time.perf_counter()
    yield
    elapsed_time_ms = (time.perf_counter() - start_time) * 1000
    callback(elapsed_time_ms)


def tik_hook(module, input):
    if not hasattr(module, 'start_time'):
        module.start_time = []
    cuda = is_cuda(module)
    if cuda:
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    module.start_time.append(start_time)


def tok_hook(module, input, output):
    if not hasattr(module, 'elapsed_time_ms'):
        module.elapsed_time_ms = []
    cuda = is_cuda(module)
    if cuda:
        torch.cuda.synchronize()
    module.elapsed_time_ms.append((time.perf_counter() - module.start_time[-1]) * 1000)
    assert len(module.elapsed_time_ms) == len(module.start_time)


@contextmanager
def profile_module(module: nn.Module):
    handles = []
    
    def register_hooks(m):
        handle1 = m.register_forward_pre_hook(tik_hook)
        handle2 = m.register_forward_hook(tok_hook)
        handles.append(handle1)
        handles.append(handle2)

    module.apply(register_hooks)

    yield Profiler(module)

    for handle in handles:
        handle.remove()


class Profiler:
    def __init__(self, module):
        self.module = module
    
    def clear(self):
        def clear_records(m):
            for name in 'start_time elapsed_time_ms'.split():
                if hasattr(m, name):
                    getattr(m, name).clear()

        self.module.apply(clear_records)
    
    def total_time(self):
        return sum(self.module.elapsed_time_ms)

    def summarize(self, module=None, ts=[0.0], prefix='', pid='Forward'):
        """
        Reference for Chrome trace event format:
        https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
        """
        if module is None:
            module = self.module
        chrome_events = []
        named_children = []
        for name, m in module.named_children():
            if isinstance(m, nn.ModuleDict):
                for name_i, m_i in m.items():
                    named_children.append((f'{name}/{name_i}', m_i))
            elif isinstance(m, nn.ModuleList):
                for i, m_i in enumerate(m):
                    named_children.append((f'{name}/{i}', m_i))
            else:
                named_children.append((name, m))
        assert len(module.start_time) == len(ts)
        parent_ts = ts
        for name, m in named_children:
            if not (hasattr(m, 'start_time') and hasattr(m, 'elapsed_time_ms') and m.start_time and m.elapsed_time_ms):
                print(f'WARNING: module {prefix}/{name} does not have time logs.')
                continue
            if len(parent_ts) < len(m.start_time):
                # this child run multiple times during a single run of its parent
                # so we repeat the parent's time logs
                ratio, reminder = divmod(len(m.start_time), len(parent_ts))
                assert reminder == 0
                ts = [ts_i for ts_i in parent_ts for _ in range(ratio)]
                start_time = [t for t in module.start_time for _ in range(ratio)]
            else:
                ts = parent_ts
                start_time = module.start_time
            assert len(ts) <= len(m.start_time)
            ts_ms = []
            for m_start_time, elapsed_time_ms, start_time, ts_i in zip(m.start_time, m.elapsed_time_ms, start_time, ts):
                ts_m = ts_i + (m_start_time - start_time) * 1000
                ts_ms.append(ts_m)
                chrome_events.append(dict(
                    name=name.lstrip('_'),
                    cat=m.__class__.__name__,  # comma separated list of categories for the event
                    ph='X',
                    ts=ts_m * 1000,
                    dur=elapsed_time_ms * 1000,
                    tid=0,
                    pid=pid,
                    args={},
                ))
            chrome_events.extend(self.summarize(m, ts=ts_ms, prefix=f'{prefix}/{name}', pid=pid))
        return chrome_events


def is_cuda(module):
    try:
        return next(module.parameters()).is_cuda
    except StopIteration:
        return False
