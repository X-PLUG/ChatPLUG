import math
import torch
import torch.nn.functional as F
from itertools import chain
from collections import defaultdict
from typing import List
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

from . import Loss, register
from xdpx.options import Argument
from .cross_entropy import query_level_f1


@register('audio_flex_cross_entropy')
class AudioCrossEntropy(Loss):
    @staticmethod
    def register(options):
        options.register(
            Argument('negative_classes_in_f1', default=[], type=List[int],
                     doc='negative classes for micro F1 score; if emtpy, no F1 score will be computed.'),
            Argument('predict_threshold', type=float),
            Argument('query_level_f1', default=False, doc='should be used with loader=="rank"'),
            Argument('f1_average', default="micro",
                     validate=lambda val: val in ('micro', 'weighted', 'macro', 'samples', 'binary')),
            Argument('kl_alpha', default=0.5),
        )
        options.add_global_constraint(
            lambda args: not args.negative_classes_in_f1 or max(args.negative_classes_in_f1) < args.num_classes,
        )

    def forward(self, model, sample, logits=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        logits1, logits2 = None, None
        if logits is None:
            logits, logits1, logits2 = model(**sample['net_input'])

        lprobs = F.log_softmax(logits, dim=-1)
        target = sample['target']
        probs = torch.exp(lprobs)
        max_p, indices = torch.max(probs, dim=1)

        logging_output = {}
        if logits1 is not None and logits2 is not None:
            lprobs1 = F.log_softmax(logits1, dim=-1)
            lprobs2 = F.log_softmax(logits2, dim=-1)
            p_loss = F.kl_div(lprobs1, F.softmax(logits2, dim=-1), reduction='none')
            q_loss = F.kl_div(lprobs2, F.softmax(logits1, dim=-1), reduction='none')
            p_loss = p_loss.sum(-1)
            q_loss = q_loss.sum(-1)
            kl_loss = (p_loss + q_loss) / 2
            loss_both = F.nll_loss(lprobs, target, reduction='none')
            loss_text = F.nll_loss(lprobs1, target, reduction='none')
            loss_audio = F.nll_loss(lprobs2, target, reduction='none')
            ce_loss = (loss_both + loss_text + loss_audio) / 3.0

            _, pred_text = torch.max(logits1, dim=1)
            _, pred_audio = torch.max(logits2, dim=1)

            logging_output.update(
                {'ce_loss': ce_loss.detach().cpu().tolist(),
                 'loss_both': loss_both.detach().cpu().tolist(),
                 'loss_text': loss_text.detach().cpu().tolist(),
                 'loss_audio': loss_audio.detach().cpu().tolist(),
                 'pred_text': pred_text.tolist(),
                 'pred_audio': pred_audio.tolist(),
                 'kl_loss': kl_loss.detach().cpu().tolist()})
            loss = ce_loss + self.args.kl_alpha * kl_loss
        else:
            loss = F.nll_loss(lprobs, target, reduction='none')

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
        logits, _, _ = model(**sample['net_input'])
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

        if 'loss_both' in logging_outputs[0]:
            loss_both = list(chain.from_iterable(log['loss_both'] for log in logging_outputs))
            agg_output['loss_both'] = sum(loss_both) / len(loss_both) / math.log(2) if sample_size > 0 else 0.
        if 'loss_text' in logging_outputs[0]:
            loss_text = list(chain.from_iterable(log['loss_text'] for log in logging_outputs))
            agg_output['loss_text'] = sum(loss_text) / len(loss_text) / math.log(2) if sample_size > 0 else 0.
            pred_text = list(chain.from_iterable(log['pred_text'] for log in logging_outputs))[:sample_size]
            agg_output['acc_text'] = accuracy_score(target, pred_text)

        if 'loss_audio' in logging_outputs[0]:
            loss_audio = list(chain.from_iterable(log['loss_audio'] for log in logging_outputs))
            agg_output['loss_audio'] = sum(loss_audio) / len(loss_audio) / math.log(2) if sample_size > 0 else 0.
            pred_audio = list(chain.from_iterable(log['pred_audio'] for log in logging_outputs))[:sample_size]
            agg_output['acc_audio'] = accuracy_score(target, pred_audio)

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
