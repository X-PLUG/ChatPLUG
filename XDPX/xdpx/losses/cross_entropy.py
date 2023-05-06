import math
import torch
import torch.nn.functional as F
from itertools import chain
from collections import defaultdict
from typing import List
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

from . import Loss, register
from xdpx.options import Argument
import torch.nn as nn


class BaseCrossEntropy(Loss):
    @staticmethod
    def register(options):
        options.register(
            Argument('negative_classes_in_f1', default=[], type=List[int],
                     doc='negative classes for micro F1 score; if emtpy, no F1 score will be computed.'),
            Argument('predict_threshold', type=float),
            Argument('query_level_f1', default=False, doc='should be used with loader=="rank"'),
            Argument('f1_average', default="micro",
                     validate=lambda val: val in ('micro', 'weighted', 'macro', 'samples', 'binary')),
            Argument('use_r_dropout', default=False, doc='use r-dropout'),
            Argument('kl_alpha', default=0.5, doc='when r-dropout=true ')
        )
        options.add_global_constraint(
            lambda args: not args.negative_classes_in_f1 or max(args.negative_classes_in_f1) < args.num_classes,
        )

    def _custom_loss_fn(self, logits, target):
        raise NotImplementedError

    def forward(self, model, sample, logits=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if logits is None:
            logits = model(**sample['net_input'])

        target = sample['target']
        loss = self._custom_loss_fn(logits, target)
        lprobs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(lprobs)
        max_p, indices = torch.max(probs, dim=1)

        logging_output = {}
        if self.args.use_r_dropout:
            logits2 = model(**sample['net_input'])
            p_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits2, dim=-1), reduction='none')
            q_loss = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits, dim=-1), reduction='none')
            p_loss = p_loss.sum(-1)
            q_loss = q_loss.sum(-1)
            kl_loss = (p_loss + q_loss) / 2
            ce_loss = 0.5 * (self._custom_loss_fn(logits, target) + self._custom_loss_fn(logits2, target))

            logging_output.update(
                {'ce_loss': ce_loss.detach().cpu().tolist(), 'kl_loss': kl_loss.detach().cpu().tolist()})
            loss = ce_loss + self.args.kl_alpha * kl_loss

        threshold = getattr(self.args, 'predict_threshold', None)
        if threshold is not None:
            indices = indices.where(max_p > threshold,
                                    torch.tensor(self.args.negative_classes_in_f1[0]).to(indices))
        sample_size = target.numel()

        logging_output.update({
            'loss': loss.detach().cpu().tolist(),
            'sample_size': sample_size,
            # for computing F1
            'target': target.tolist(),
            'pred': indices.tolist(),
        })
        if getattr(self.args, 'query_level_f1', False) and not model.training:
            logging_output.update({
                'query_id': [_id.split('-')[0] for _id in sample['id']],
            })

        if logits.size(1) == 2:
            logging_output.update({
                'prob': probs[:, 1].tolist(),
            })
        if reduce:
            loss = loss.sum()
        return loss, sample_size, logging_output

    def inference(self, model, sample):
        logits = model(**sample['net_input'])
        probs = self.get_prob(logits)
        _, indices = torch.max(probs, dim=1)
        return indices.tolist(), probs.tolist()

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        """Aggregate logging outputs from data parallel training."""

        agg_output = {
            'sample_size': sample_size,
        }
        if 'ntokens' in logging_outputs[0]:
            ntokens = sum(log['ntokens'] for log in logging_outputs)
            agg_output['ntokens'] = ntokens
        if max_count:
            sample_size = max(min(max_count, sample_size), 1)
        loss = list(chain.from_iterable(log['loss'] for log in logging_outputs))[:sample_size]
        target = list(chain.from_iterable(log['target'] for log in logging_outputs))[:sample_size]
        pred = list(chain.from_iterable(log['pred'] for log in logging_outputs))[:sample_size]

        agg_output['loss'] = sum(loss) / sample_size / math.log(2) if sample_size > 0 else 0.
        accuracy = accuracy_score(target, pred)
        agg_output['acc'] = accuracy

        if 'ce_loss' in logging_outputs[0]:
            ce_loss = list(chain.from_iterable(log['ce_loss'] for log in logging_outputs))
            agg_output['ce_loss'] = sum(ce_loss) / len(ce_loss) / math.log(2) if sample_size > 0 else 0.

        if 'kl_loss' in logging_outputs[0]:
            kl_loss = list(chain.from_iterable(log['kl_loss'] for log in logging_outputs))
            agg_output['kl_loss'] = sum(kl_loss) / len(kl_loss) / math.log(2) if sample_size > 0 else 0.

        if getattr(self.args, 'query_level_f1', False):
            if 'query_id' in logging_outputs[0]:
                query_ids = list(chain.from_iterable(log['query_id'] for log in logging_outputs))[:sample_size]
                prob = list(chain.from_iterable(log['prob'] for log in logging_outputs))[:sample_size]
                precision, recall, f1 = query_level_f1(query_ids, target, pred, prob)
                agg_output['precision'] = precision
                agg_output['recall'] = recall
                agg_output['f1'] = f1
        elif self.args.negative_classes_in_f1:
            labels = [i for i in range(self.args.num_classes) if i not in self.args.negative_classes_in_f1]
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(target, pred, labels=labels, average='micro')
            except ValueError:
                precision = recall = f1 = 0.0
            agg_output['precision'] = precision
            agg_output['recall'] = recall
            agg_output['f1'] = f1
        else:
            labels = [i for i in range(self.args.num_classes)]
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(target, pred, labels=labels,
                                                                           average=self.args.f1_average)
            except ValueError:
                precision = recall = f1 = 0.0
            agg_output['precision'] = precision
            agg_output['recall'] = recall
            agg_output['f1'] = f1

        if 'prob' in logging_outputs[0]:
            prob = list(chain.from_iterable(log['prob'] for log in logging_outputs))[:sample_size]
            try:
                auc = roc_auc_score(target, prob)
            except ValueError:  # when batch size is small and only one class presented in targets
                auc = 0.0
            agg_output['auc'] = auc

        return agg_output


def query_level_f1(query_ids, targets, predictions, probabilities):
    queries = defaultdict(lambda: ([], [], []))
    for i, t, pd, pb in zip(query_ids, targets, predictions, probabilities):
        queries[i][0].append(t)
        queries[i][1].append(pd)
        queries[i][2].append(pb)
    tp = 0
    fp = 0
    fn = 0
    for t, pd, pb in queries.values():
        index = max(range(len(pb)), key=pb.__getitem__)
        predict = pd[index]
        target = t[index]
        if predict:
            if target == predict:
                tp += 1
            else:
                fp += 1
        if any(t) and (target != predict or not predict):
            fn += 1
    if tp == 0:
        return 0., 0., 0.
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


@register('cross_entropy')
class CrossEntropy(BaseCrossEntropy):
    def _custom_loss_fn(self, logits, target):
        return F.nll_loss(F.log_softmax(logits, dim=-1), target, reduction='none')


@register('label_smooth_ce')
class LabelSmoothCrossEntropyLoss(BaseCrossEntropy):
    @staticmethod
    def register(options):
        BaseCrossEntropy.register(options)
        options.register(
            Argument('smoothing', default=0.1)
        )

    def _custom_loss_fn(self, logits, target):
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = (1 - self.args.smoothing) * nll_loss + self.args.smoothing * smooth_loss
        return loss


@register('taylor_ce')
class TaylorCrossEntropyLoss(BaseCrossEntropy):
    @staticmethod
    def register(options):
        BaseCrossEntropy.register(options)
        options.register(
            Argument('taylor_n', default=2)
        )

    def __init__(self, args):
        super().__init__(args)

    def _custom_loss_fn(self, logits, target):
        log_probs = self.taylor_softmax(logits, self.args.taylor_n, dim=-1).log()
        loss = F.nll_loss(log_probs, target, reduction='none')
        return loss

    # implementations reference - https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
    # paper - https://www.ijcai.org/Proceedings/2020/0305.pdf

    def taylor_softmax(self, x, n, dim=-1):
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, n + 1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / torch.sum(fn, dim=dim, keepdim=True)
        return out
