import os
import sys
import traceback
import torch
from xdpx.options import Arg, Options
from xdpx.tasks import tasks
from xdpx.loaders import loaders
from xdpx.loaders.parsers import parsers
from xdpx.utils import io, move_to_cuda, parse_model_path


class Model:
    def __init__(self, save_dir, strict=True, notnull=False, checkpoint=None):
        self.cuda = torch.cuda.is_available()
        self.strict = strict
        self.notnull = notnull
        with io.open(os.path.join(save_dir, 'args.py')) as f:
            args = Options.parse_tree(eval(f.read()))
        try:
            with io.open(os.path.join(args.data_dir, 'args.py')) as f:
                args = Arg().update(Options.parse_tree(eval(f.read()))).update(args)
        except IOError:
            pass
        args.__cmd__ = 'serve'
        args.save_dir = save_dir
        args.strict_size = True
        # build the task
        task = tasks[args.task](args)
        model = task.build_model(args)
        loss = task.build_loss(args)
        if checkpoint:
            model_path = checkpoint
        else:
            model_path = parse_model_path('<best>', args)
        model.load(model_path)

        if self.cuda:
            model = model.cuda()
            loss = loss.cuda()

        self.task = task
        self.processor = task.processor
        self.loader = loaders[args.loader](args)
        self.parser = parsers[args.parser](args)
        self.model = model
        self.loss = loss

        # warmup the processor to load lazy-loaded resources in advance
        self.predict(self.placeholder)

    @property
    def placeholder(self):
        if 'pair' in self.processor.__class__.__name__.lower():
            return [['XDPX', 'XDPX']]
        elif 'chat' in self.processor.__class__.__name__.lower():
            return [['XDPX','XDPX','XDPX']] # context,response,passages
        return [['XDPX']]
    
    def predict(self, batch):
        if not isinstance(batch, list):
            batch = [batch]
        try:
            batch = [self.processor.encode(self.loader, sample) for sample in batch]
            batch = self.processor.collate(batch)
            if self.cuda:
                batch = move_to_cuda(batch)
            return self.task.inference_step(batch, self.model, self.loss)
        except Exception as e:
            if self.strict:
                raise e
            traceback.print_exc()
            if self.notnull:
                return self.predict(self.placeholder)


def cli_main(argv):
    """for interactive testing of model behaviour"""
    assert len(argv) == 2
    save_dir = argv[1]
    checkpoint = None
    if io.isfile(save_dir):
        checkpoint = save_dir
        save_dir = os.path.dirname(checkpoint)
    model = Model(save_dir, strict=False, notnull=False, checkpoint=checkpoint)
    _id = 0
    while True:
        try:
            inputs = input('INPUT:  ')
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            continue
        except Exception as e :
            print(f'input error: {e}')
            continue
        try:
            inputs = model.parser.parse_line(inputs)
        except Exception as e:
            print(f'parser error: {e}')
            continue
        if isinstance(inputs, str):
            inputs = [inputs]
        # print('ENCODE ', model.processor.encode(model.loader, inputs, _id))
        print('PREDICT', model.predict([inputs]))
        print()
        _id += 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python xdpx/serve.py $save_dir')
        exit()
    cli_main(sys.argv)
