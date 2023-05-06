import os
import math
import time
import torch
from collections import OrderedDict, defaultdict
from xdpx.utils import io
from xdpx.utils.distributed_utils import is_master


class _LogWrapper:
    def __init__(self):
        self._logger = None
        self._warned = set()
    
    def set_logger(self, logger):
        self._logger = logger
    
    def __getattr__(self, name):
        if self._logger is None:
            if name not in self._warned:
                print(f'WARNING: logger.{name} is called before logger initialization.')
                self._warned.add(name)
            return self.void
        if hasattr(self._logger, name):
            return getattr(self._logger, name)
        return super().__getattr__(name)
    
    def void(self, *args, **kwargs):
        pass
    
    def __repr__(self):
        return f'Logger for {self._logger.save_dir}'


log = _LogWrapper()


class Logger:
    def __init__(self, args):
        self.epoch = 0
        self.cuda = args.cuda
        self.save_dir = args.save_dir
        self.interval = args.log_interval
        self.major_metric = args.major_metric
        self.ascending_metric = args.ascending_metric
        self.clip_norm = args.clip_norm
        self.best_score = math.inf * (-1. if args.ascending_metric else 1.)
        self.is_master = is_master(args)
        self.init_meters()
        log.set_logger(self)
    
    def init_meters(self):
        self.meters = OrderedDict()
        self.meters['loss'] = AverageMeter()
        self.meters['wps'] = StopwatchMeter()  # words per second
        self.meters['ups'] = StopwatchMeter()  # updates per second
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()   # % of updates clipped
        self.meters['train_wall'] = StopwatchMeter()  # train wall time in seconds
        self.extra_meters = defaultdict(lambda: AverageMeter())
        self.summaries = defaultdict(lambda: AverageMeter())
        self.accum_summaries = {}
        self.train_output = None
        self.valid_output = None
        self.train_header = None
        self.valid_header = None
        train_log_path = os.path.join(self.save_dir, 'train.log.tsv')
        valid_log_path = os.path.join(self.save_dir, 'valid.log.tsv')
        if io.exists(valid_log_path):  # resume
            import pandas as pd
            try:
                with io.open(valid_log_path) as f:
                    prev_log = pd.read_csv(f, sep='\t', header=0).iloc[-1]
                    self.best_score = prev_log['best_score']
                    self.meters['train_wall'].reset(start=prev_log['train_wall'])
                    # avoid writing the header again
                    self.train_header = self.valid_header = True
            except pd.errors.EmptyDataError:  
                # when in distributed training and the master has just created the file
                pass
        if self.is_master:
            self.train_output = io.open(train_log_path, 'a')
            self.valid_output = io.open(valid_log_path, 'a')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def tik(self):
        self.meters['ups'].start()
        self.meters['wps'].start()
        self.meters['train_wall'].start()
        
    def add_summary(self, name, value, n=1):
        if torch.is_tensor(value):
            value = value.cpu().item()
        self.summaries[name].update(value, n)
    
    def add_accumulative_summary(self, name, values, reduce_fn: callable = None, prefix=''):
        if name not in self.accum_summaries:
            assert reduce_fn is not None, 'reduce_fn should be specified in the first call of add_accumulative_summary'
            self.accum_summaries[name] = dict(
                values=[],
                reduce_fn=reduce_fn,
                prefix=prefix,
            )
        self.accum_summaries[name]['values'].append(values)

    def update(self, logging_output):
        logging_output = logging_output.copy()
        step = logging_output.pop('step')
        grad_norm = logging_output.pop('grad_norm')
        sample_size = logging_output.pop('sample_size')
        ntokens = logging_output.pop('ntokens')
        clip = 1. if grad_norm > self.clip_norm and self.clip_norm > 0 else 0.
        loss = logging_output.pop('loss', 0)
        loss_scale = logging_output.pop('loss_scale', None)

        self.meters['loss'].update(loss, sample_size)
        self.meters['gnorm'].update(grad_norm)
        self.meters['clip'].update(clip)
        self.meters['ups'].stop()
        self.meters['wps'].stop(ntokens)
        self.meters['train_wall'].stop()
        wps = ntokens / self.meters['wps'].delta
        ups = 1. / self.meters['ups'].delta

        for k, v in logging_output.items():
            self.extra_meters[k].update(v, sample_size)

        if self.is_master and self.train_output and step % self.interval == 0:
            if not self.train_header:  # lazy write header
                self.train_header = ['epoch', 'step', 'loss', 'gnorm', 'clip', 'wps', 'ups'] + sorted(self.extra_meters.keys()) + \
                    sorted(self.summaries.keys())
                if loss_scale:
                    self.train_header += ['loss_scale']
                self.train_output.write('\t'.join(self.train_header) + '\n')
            self.train_output.write('\t'.join(map(str,
                [self.epoch, step, loss, grad_norm, clip, wps, ups] +
                [self.extra_meters[key].val for key in sorted(self.extra_meters.keys())] +
                [self.summaries[key].val for key in sorted(self.summaries.keys())] +
                ([loss_scale] if loss_scale else [])
            )) + '\n')
            self.train_output.flush()
        return OrderedDict((
            ('loss', f'{loss:.5f}'),
            ('gnorm', f'{grad_norm:.1f}'),
            ('wps', f'{wps:.1f}'),
        ))
    
    def summarize(self, step, valid_stats):
        assert self.major_metric in valid_stats
        train_loss = self.meters['loss'].avg
        grad_norm = self.meters['gnorm'].avg
        clip = self.meters['clip'].avg
        ups = 1. / self.meters['ups'].avg if self.meters['ups'].avg else 0.
        wps = 1. / self.meters['wps'].avg if self.meters['wps'].avg else 0.
        train_wall = self.meters['train_wall'].sum
        valid_stats.pop('sample_size')
        valid_loss = valid_stats['loss']
        score = valid_stats[self.major_metric]
        # GPU memory allocated in MB
        memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if self.cuda else None
        if (score > self.best_score) == self.ascending_metric:
            self.best_score = score

        if self.is_master and self.valid_output:
            accum_summary_names, accum_summaries = self.summarize_accum()
            
            if not self.valid_header:
                self.valid_header = ['epoch', 'step', 'ups', 'wps', 'train_wall', 'train_loss', 'gnorm', 'clip'] + \
                    (['memory'] if self.cuda else []) + \
                    ['train_' + key for key in sorted(self.extra_meters.keys())] + \
                    ['valid_' + key for key in sorted(valid_stats.keys())] + \
                    accum_summary_names + \
                    ['best_score']
                self.valid_output.write('\t'.join(self.valid_header) + '\n')

            self.valid_output.write('\t'.join(map(str,
                [self.epoch, step, ups, wps, train_wall, train_loss, grad_norm, clip] +
                ([memory] if self.cuda else []) + 
                [self.extra_meters[key].avg for key in sorted(self.extra_meters.keys())] +
                [valid_stats[key] for key in sorted(valid_stats.keys())] +
                accum_summaries + 
                [self.best_score]
            )) + '\n')
            self.valid_output.flush()
        
        self.meters['loss'].reset()
        self.meters['gnorm'].reset()
        self.meters['clip'].reset()
        self.meters['ups'].reset()
        self.meters['wps'].reset()
        if self.cuda and torch.__version__ >= '1.2':
            torch.cuda.reset_max_memory_allocated()
        for collection in (self.extra_meters, self.summaries):
            for meter in collection.values():
                meter.reset()
        for records in self.accum_summaries.values():
            records['values'].clear()
        return f'train_loss={train_loss:.2f}, valid_loss={valid_loss:.2f}, clip={clip*100:.0f}%, ' \
            f'wps={wps:.1f}, score={score:.3f}, best={self.best_score:.3f}', score

    def summarize_accum(self):
        accum_summary_names = []
        accum_summaries = []
        for key in sorted(self.accum_summaries.keys()):
            stats = self.accum_summaries[key]
            fn = stats['reduce_fn']
            values = stats['values']
            prefix = stats['prefix']
            with torch.no_grad():
                results = fn(values)
            names = sorted(results.keys())
            for name in names:
                accum_summary_names.append(prefix + name)
                accum_summaries.append(results[name])
        return accum_summary_names, accum_summaries
        
    def get_meter(self, name):
        return self.meters.get(name, None)


class SummaryLogger:
    def __init__(self):
        self.accum_summaries = {}
        log.set_logger(self)

    def add_summary(self, *args, **kwargs):
        pass

    def add_accumulative_summary(self, *args, **kwargs):
        Logger.add_accumulative_summary(self, *args, **kwargs)

    def summarize(self):
        accum_summary_names, accum_summaries = Logger.summarize_accum(self)
        for records in self.accum_summaries.values():
            records['values'].clear()
        return accum_summary_names, list(map(str, accum_summaries))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""
    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            self.delta = time.time() - self.start_time
            self.sum += self.delta
            self.n += n
            self.start_time = None

    def reset(self, start=0, n=0):
        self.sum = start
        self.n = n
        self.start_time = None
        self.delta = None

    @property
    def avg(self):
        return self.sum / self.n if self.n > 0 else 0
