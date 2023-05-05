import os
import sys
import traceback
import torch
from xdpx.tasks import tasks
from xdpx.loaders import loaders
from xdpx.utils import io, move_to_cuda, parse_model_path
from xdpx.bootstrap import bootstrap
from xdpx.options import Options, Argument, Arg
import logging
import json
from tqdm import tqdm

import numpy as np
import random
from xdpx.models.fewshot.mgimn import BertMatchingNetBase
import torch.nn.functional as F

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


class Model:
    def __init__(self, save_dir, strict=True, notnull=False, checkpoint=None):
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
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
        self.args = args

    def predict(self, batch, support_input_ids, seq_support_emb, pooled_support_emb,
                support_labels):  # batch: List[str]
        if not isinstance(batch, list):
            batch = [batch]
        try:
            inputs = {'id': [0], 'support': [], 'mode': 'retrieval', 'domain': 'default',
                      'query': [self.loader.tokenizer.encode(item) for item in batch], 'support_labels': support_labels}
            episode = [self.processor.numerize(inputs)]
            episode = self.processor.collate(episode)

            if self.cuda:
                episode = move_to_cuda(episode)
            self.model.eval()
            with torch.no_grad():
                episode['net_input']['support_input_ids'] = support_input_ids
                if isinstance(self.model, BertMatchingNetBase):
                    episode['net_input']['support_emb'] = seq_support_emb
                else:
                    episode['net_input']['support_emb'] = pooled_support_emb

                logits, support_emb, query_emb, proto_logits, prompt_logits, prompt_masked_logits, masked_logits = self.model(**episode['net_input'])  # K' * N
                probs = F.softmax(logits, dim=-1)

            return probs
        except Exception as e:
            if self.strict:
                raise e
            traceback.print_exc()

    def pre_compute_support_emb(self, support_file):
        support_samples = json.loads(io.open(support_file).read())
        dic = {}
        for sample in support_samples:
            text = sample['text']
            label = sample['label']
            if label not in dic:
                dic[label] = []
            dic[label].append(text)

        support_labels = list(dic.keys())
        try:
            support_set = [[self.loader.tokenizer.encode(text) for text in dic[label]] for label in support_labels]
            inputs = {'id': [0], 'support': support_set, 'domain': 'default',
                      'query': [], 'mode': 'retrieval'}
            episode = [self.processor.numerize(inputs)]
            episode = self.processor.collate(episode)  # 1 * K * max_seq
            if self.cuda:
                episode = move_to_cuda(episode)

            self.model.eval()
            with torch.no_grad():
                support_input_ids = episode['net_input']['support_input_ids']
                N, K, max_support_len = support_input_ids.shape
                support_input_ids = support_input_ids.reshape(-1, max_support_len)

                seq_support_emb_list = []
                pooled_support_emb_list = []
                for i in tqdm(range(0, N * K, 100)):
                    net_input = {'input_ids': support_input_ids[i:i + 100]}
                    seq_output, pooled_output = self.model.bert_forward(**net_input)[:2]  # NK * D
                    dim = seq_output.shape[-1]
                    if isinstance(self.model, BertMatchingNetBase):
                        seq_support_emb_list.append(seq_output)
                    else:
                        pooled_support_emb_list.append(pooled_output)

                if isinstance(self.model, BertMatchingNetBase):
                    seq_output = torch.cat(seq_support_emb_list, dim=0)
                    seq_support_emb = seq_output.view(N, K, max_support_len, dim)
                    return support_input_ids, seq_support_emb, None, support_labels
                else:
                    pooled_output = torch.cat(pooled_support_emb_list, dim=0)
                    pooled_support_emb = pooled_output.view(N, K, dim)
                    return support_input_ids, None, pooled_support_emb, support_labels

        except Exception as e:
            if self.strict:
                raise e
            traceback.print_exc()


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('save_dir', required=True, validate=lambda val: io.exists(
            os.path.join(val, 'args.py')) or io.exists(os.path.join(os.path.dirname(val), 'args.py'))),
        Argument('support_file', required=True, type=str),
        Argument('test_file', required=True, type=str),
        Argument('save_prefix', type=str, default="save_prefix")

    )
    bootstrap(options, main, __file__, argv)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def main(args: Arg):
    checkpoint = None
    if io.isfile(args.save_dir):
        checkpoint = args.save_dir
        save_dir = os.path.dirname(checkpoint)
    else:
        save_dir = args.save_dir
    model = Model(save_dir, strict=True, notnull=False, checkpoint=checkpoint)

    model_name = model.args.model
    print('current few-shot model: {}'.format(model_name))

    support_input_ids, seq_support_emb, pooled_support_emb, support_labels \
        = model.pre_compute_support_emb(args.support_file)

    test_samples = json.loads(io.open(args.test_file).read())
    test_labels = [sample['label'] if 'label' in sample else 'UNK' for sample in test_samples ]
    test_texts = [sample['text'] for sample in test_samples]

    predict_labels = []
    predict_probs = []
    for i in tqdm(range(0, len(test_texts), 10)):
        logits = model.predict(test_texts[i:i + 10], support_input_ids,
                               seq_support_emb, pooled_support_emb, support_labels)
        logits = logits.detach().cpu().tolist()
        pindex = np.argmax(logits, 1)
        predict_labels.extend([support_labels[p] for p in pindex])
        predict_probs.extend(np.max(logits, 1).tolist())
    acc = sum(p == t for p, t in zip(predict_labels, test_labels)) / float(len(test_labels))
    print('acc:{}'.format(acc))

    with io.open(os.path.join(save_dir, args.save_prefix, 'badcases.csv'), 'w') as nf:
        nf.write('acc:{}\n'.format(acc))
        nf.write('{}\t{}\t{}\n'.format("text", "ground_truth", "predict_label"))
        for idx, (p, t, s) in enumerate(zip(predict_labels, test_labels, predict_probs)):
            if p != t:
                nf.write('{}\t{}\t{}\t{}\n'.format(test_texts[idx], t, p, s))
    with io.open(os.path.join(save_dir, args.save_prefix, 'predict_labels.txt'), 'w') as nf:
        for text, pred_label in zip(test_texts, predict_labels):
            nf.write('{}\t{}\n'.format(text, pred_label))


if __name__ == '__main__':
    cli_main(sys.argv)
