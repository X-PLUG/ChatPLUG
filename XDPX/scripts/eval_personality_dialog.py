import os
import sys
import traceback
import torch
from xdpx.options import Arg, Options
from xdpx.tasks import tasks
from xdpx.loaders import loaders
from xdpx.loaders.parsers import parsers
from xdpx.utils import io, move_to_cuda, parse_model_path
import json
import numpy as np
from tqdm import tqdm

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
            return [['XDPX', 'XDPX', 'XDPX']]  # context,response,passages
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
    assert len(argv) == 3, 'usage: x-script eval_personality_dialog <style_classifier_save_dir>  <test_file> '
    save_dir = argv[1]  # oss://xdp-expriment/gaoxing.gx/chat/personality/model/zh/0.0001/
    checkpoint = None
    if io.isfile(save_dir):
        checkpoint = save_dir
        save_dir = os.path.dirname(checkpoint)
    model = Model(save_dir, strict=False, notnull=False, checkpoint=checkpoint)

    labels = [l.strip().lower().replace(' ', '') for l in
              io.open('oss://xdp-expriment/gaoxing.gx/chat/personality/data/zh/target_map.txt')]

    test_file = argv[2]  # oss://xdp-expriment/gaoxing.gx/chat/benchmark/results/test_file.logs.1667284959.9586577.json

    with io.open(test_file + '.eval.output.json', 'w') as nf:
        lines = io.open(test_file).readlines()[1:]
        print(f'total size:{len(lines)}')
        topn = 30
        hit_cnt = 0
        for l in tqdm(lines):
            d = json.loads(l)
            response = d['generated_response']
            try:
                inputs = model.parser.parse_line(response)
            except Exception as e:
                print(f'parser error: {e}')
                continue
            if isinstance(inputs, str):
                inputs = [inputs]
            result = model.predict([inputs])

            prob = [(t, i) for i, t in enumerate(result[1][0])]
            prob = sorted(prob, key=lambda x: x[0], reverse=True)
            topn_labels = [labels[t[1]] for t in prob[:topn]]
            d['generated_response_labels'] = '、'.join(topn_labels)
            bot_profile = d['bot_profile']
            topn_labels_in_bot_profile = any([int(t in bot_profile) for t in topn_labels])
            hit_cnt += int(topn_labels_in_bot_profile)
            nl = json.dumps(d, ensure_ascii=False) + '\n'
            nf.write(nl)
        print(f'hit_cnt: {hit_cnt}, ratio={hit_cnt/float(len(lines))}')


if __name__ == '__main__':
    cli_main(sys.argv)
