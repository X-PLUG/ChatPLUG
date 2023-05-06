import torch
import unittest
from torch.utils.data import DataLoader
from xdpx.options import Arg
from xdpx.modules import LinearProjection
from xdpx.optimizers import optimizers
from xdpx.optimizers.lr_schedulers import lr_schedulers


class TestLinearDecayScheduler(unittest.TestCase):
    def test_single_data(self):
        args = Arg()
        args.max_epoch = 3
        args.max_update = None
        args.learning_rate = 1.0
        args.batch_size = 64
        args.warmup_steps = 500
        args.train_subset = ['train']
        args.adam_betas = [0.9, 0.999]
        args.adam_eps = 1e-6
        args.weight_decay = 0.0
        args.distributed_world_size = 1
        args.anneal_strategy = 'linear'
        args.div_factor = 25
        args.update_freq = 1
        args.lazy_load = False
        args.cycle_momentum = False
        args.data_size = {
            'train': 100_000,
        }
        model = LinearProjection(10, 20)  # just a random model
        params = model.parameters()
        optimizer = optimizers['adam'](args, params)
        scheduler = lr_schedulers['one_cycle'](args, optimizer)
        scheduler.step_update(0)

        prev_lr = optimizer.get_lr()
        self.assertAlmostEqual(prev_lr, args.learning_rate / args.div_factor)
        
        step = 0
        data = [0] * 100_000

        def collate(samples):
            return torch.tensor(samples)

        data = DataLoader(
            data, batch_size=args.batch_size * args.update_freq, drop_last=True, shuffle=True, collate_fn=collate,
        )
        for _ in range(args.max_epoch):
            for _ in data:
                step += 1
                scheduler.step_update(step)
                new_lr = optimizer.get_lr()
                if step < args.warmup_steps:
                    self.assertLess(prev_lr, new_lr)
                elif step == args.warmup_steps:
                    self.assertAlmostEqual(args.learning_rate, new_lr)
                else:
                    self.assertLess(new_lr, prev_lr)
                prev_lr = new_lr
        self.assertAlmostEqual(optimizer.get_lr(), args.learning_rate / args.div_factor)

    def test_multi_data(self):
        args = Arg()
        args.max_epoch = 3
        args.max_update = None
        args.learning_rate = 1.0
        args.batch_size = 64
        args.warmup_steps = 500
        args.train_subset = ['train0', 'train1']
        args.adam_betas = [0.9, 0.999]
        args.adam_eps = 1e-6
        args.weight_decay = 0.0
        args.distributed_world_size = 1
        args.anneal_strategy = 'linear'
        args.div_factor = 25
        args.update_freq = 1
        args.lazy_load = False
        args.cycle_momentum = False
        args.data_size = {
            'train0': 50_000,
            'train1': 30_000,

        }
        model = LinearProjection(10, 20)  # just a random model
        params = model.parameters()
        optimizer = optimizers['adam'](args, params)
        scheduler = lr_schedulers['one_cycle'](args, optimizer)
        scheduler.step_update(0)

        prev_lr = optimizer.get_lr()
        self.assertAlmostEqual(prev_lr, args.learning_rate / args.div_factor)

        step = 0
        data0 = [0] * 50_000
        data1 = [0] * 30_000

        def collate(samples):
            return torch.tensor(samples)

        data = DataLoader(
            data0 + data1, batch_size=args.batch_size * args.update_freq, drop_last=True, shuffle=True,
            collate_fn=collate,
        )
        for _ in range(args.max_epoch):
            for _ in data:
                step += 1
                scheduler.step_update(step)
                new_lr = optimizer.get_lr()
                if step < args.warmup_steps:
                    self.assertLess(prev_lr, new_lr)
                elif step == args.warmup_steps:
                    self.assertAlmostEqual(args.learning_rate, new_lr)
                else:
                    self.assertLess(new_lr, prev_lr)
                prev_lr = new_lr
        self.assertAlmostEqual(optimizer.get_lr(), args.learning_rate / args.div_factor)

    def test_lazy_load(self):
        args = Arg()
        args.max_epoch = 3
        args.max_update = None
        args.learning_rate = 1.0
        args.batch_size = 64
        args.warmup_steps = 500
        args.train_subset = ['train0', 'train1']
        args.adam_betas = [0.9, 0.999]
        args.adam_eps = 1e-6
        args.weight_decay = 0.0
        args.distributed_world_size = 1
        args.anneal_strategy = 'linear'
        args.div_factor = 25
        args.update_freq = 1
        args.lazy_load = True
        args.cycle_momentum = False
        args.data_size = {
            'train0': 50_000,
            'train1': 30_000,

        }
        model = LinearProjection(10, 20)  # just a random model
        params = model.parameters()
        optimizer = optimizers['adam'](args, params)
        scheduler = lr_schedulers['one_cycle'](args, optimizer)
        scheduler.step_update(0)

        prev_lr = optimizer.get_lr()
        self.assertAlmostEqual(prev_lr, args.learning_rate / args.div_factor)

        step = 0
        data0 = [0] * 50_000
        data1 = [0] * 30_000

        def collate(samples):
            return torch.tensor(samples)

        data = [DataLoader(
            data_i, batch_size=args.batch_size * args.update_freq, drop_last=True, shuffle=True,
            collate_fn=collate,
        )for data_i in [data0, data1]]
        for i in range(args.max_epoch):
            for _ in data[i % len(data)]:
                step += 1
                scheduler.step_update(step)
                new_lr = optimizer.get_lr()
                if step < args.warmup_steps:
                    self.assertLess(prev_lr, new_lr)
                elif step == args.warmup_steps:
                    self.assertAlmostEqual(args.learning_rate, new_lr)
                else:
                    self.assertLess(new_lr, prev_lr)
                prev_lr = new_lr
        self.assertAlmostEqual(optimizer.get_lr(), args.learning_rate / args.div_factor)


if __name__ == '__main__':
    unittest.main()
