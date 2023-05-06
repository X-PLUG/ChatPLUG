import os
import sys
import traceback
import torch
from prettytable import PrettyTable
from xdpx.tasks import tasks
from xdpx.loaders import loaders
from xdpx.utils import io, move_to_cuda, parse_model_path
from xdpx.bootstrap import bootstrap
from xdpx.options import Options, Argument, Arg
import logging
from typing import List
import json

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


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
        self.model = model
        self.loss = loss

        # warmup the processor to load lazy-loaded resources in advance
        self.predict(self.placeholder)

    @property
    def placeholder(self):
        if 'pair' in self.processor.__class__.__name__.lower():
            return [['XDPX', 'XDPX']]
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


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('simcse_senteval', required=True),
        Argument('save_dir', required=True, validate=lambda val: io.exists(
            os.path.join(val, 'args.py')) or io.exists(os.path.join(os.path.dirname(val), 'args.py'))),
        Argument('mode',
                 doc='What evaluation mode to use (dev: fast mode, dev results; '
                     'test: full mode, test results); fasttest: fast mode, test results',
                 default="test",
                 validate=lambda value: (value in ['test', 'dev', 'fasttest'], f'Unknown mode {value}'),
                 ),
        Argument('task_set',
                 doc='What set of tasks to evaluate on.',
                 default="na",
                 validate=lambda value: (value in ['sts', 'transfer', 'full', 'na'], f'Unknown taskset {value}'),
                 ),
        Argument('tasks',
                 doc="Tasks to evaluate on. If 'task_set' is specified, this will be overridden",
                 default=['STSBenchmark'], type=List[str]
                 ),
    )
    bootstrap(options, main, __file__, argv)


def main(args: Arg):
    # Set path to SentEval
    print(os.path.dirname(__file__))
    PATH_TO_DATA = os.path.join(args.simcse_senteval, "data")
    # Import SentEval
    sys.path.insert(0, args.simcse_senteval)
    import senteval

    checkpoint = None
    if io.isfile(args.save_dir):
        checkpoint = args.save_dir
        save_dir = os.path.dirname(checkpoint)
    else:
        save_dir = args.save_dir
    model = Model(save_dir, strict=False, notnull=False, checkpoint=checkpoint)

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        sentences = [[' '.join(s)] for s in batch]
        embeddings = model.predict(sentences)[1]
        return embeddings

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    print("| tasks: {}".format(args.tasks))
    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        print(json.dumps(result, indent=4))
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


if __name__ == '__main__':
    cli_main(sys.argv)
