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
    assert len(argv) >= 4, 'usage: x-script chat_add_style <style_classifier_save_dir>  <input_dir> <output_dir> '
    save_dir = argv[1]  # oss://xdp-expriment/gaoxing.gx/chat/personality/model/zh/0.0001/
    checkpoint = None
    if io.isfile(save_dir):
        checkpoint = save_dir
        save_dir = os.path.dirname(checkpoint)
    model = Model(save_dir, strict=False, notnull=False, checkpoint=checkpoint)

    labels = [l.strip().lower().replace(' ', '') for l in
              io.open(f'{save_dir}/target_map.txt')]

    data_dir = argv[
        2]  # 'oss://xdp-expriment/gaoxing.gx/chat/data/personal/  oss://xdp-expriment/jiayi.qm/0_digital_human/data/v1.1.6/
    new_data_dir = argv[3]  # '/tmp/root/filecache/xdp-expriment/gaoxing.gx/chat/data/personal_style/'
    topn = int(argv[4]) if len(argv) > 4 else 3

    label_str_map = {
        'a': '活泼可爱的、有青春活力、甜美的、乐观的、快乐的、诙谐风趣、富有趣味的、爱玩的、幽默的、温柔体贴的、善解人意的、充满柔情的、富有同情心',
        'b': '强势的、冷酷的、善变的、浮夸的、野蛮的、爱瞎想的、势利爱炫耀的、金钱至上的、自恋的、脾气坏的、古怪的、爱问问题的、讽刺的、挑衅的、懒散的',
        'c':'理性的、平和的、博学的、有爱心的、恭敬有礼貌的、浪漫的、自然的、老练的、温暖的、谦逊的、诚实的、友善的、绅士有礼貌的、有口才善于表达',
        'd':'消极的、偏执的、呆板的、忧郁的、悲观沮丧的、不诚实的、焦虑的、困惑的、羞怯的、空虚的、迷糊的、糊涂的、懦弱的、相信宿命论的、害怕的、可悲的'
    }
    for file in io.listdir(data_dir):
        with io.open(os.path.join(new_data_dir, file), 'w') as nf:
            with io.open(os.path.join(data_dir, file)) as f:
                print(file)
                lcnt = 0
                for l in f:
                    lcnt += 1
                    if lcnt % 1000 == 0:
                        print(lcnt)
                    try:
                        data = json.loads(l)
                        response = data['response'].strip().replace('</s>', '')
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

                        topn_labels = [label_str_map[t] if t in label_str_map else t for t in topn_labels ]
                        extra_bot_profile = ';'.join(['我是个{}人'.format(l) for l in topn_labels])
                        passages = data['passages'].split(';;;')
                        append = False
                        for i, p in enumerate(passages):
                            if p.strip().startswith('bot_profile:'):
                                passages[i] += ';' + extra_bot_profile
                                append = True
                        if not append:
                            passages.append('bot_profile:{}'.format(extra_bot_profile))
                        passages = ';;;'.join(passages)
                        data['passages'] = passages
                        nl = json.dumps(data, ensure_ascii=False) + '\n'
                        nf.write(nl)
                    except Exception as e:
                        pass


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python xdpx/chat_add_style.py $save_dir')
        exit()
    cli_main(sys.argv)
