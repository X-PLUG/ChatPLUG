import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from typing import List
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

from . import Loss, register
from xdpx.options import Argument


@register('bce')
class BinaryCrossEntropy(Loss):
    @staticmethod
    def register(options):
        options.register(
            Argument('use_r_dropout', default=False, doc='use r-dropout'),
            Argument('kl_alpha', default=0.5, doc='when r-dropout=true '),
            Argument('predict_threshold', type=List[float],
                     validate=lambda val: val is None or all(x > 0 for x in val)),
        )
        options.add_global_constraint(
            lambda args: args.predict_threshold is None or len(args.predict_threshold) == args.num_classes - 1
        )

    def __init__(self, args):
        super().__init__(args)
        if args.predict_threshold:
            threshold = torch.tensor([0.5] + self.args.predict_threshold).unsqueeze(0)
        else:
            threshold = torch.ones(1, args.num_classes) * 0.5
        self.threshold = nn.Parameter(threshold, requires_grad=False)

    def forward(self, model, sample, logits=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if logits is None:
            logits = model(**sample['net_input'])
        probs = torch.sigmoid(logits)
        target = sample['target']
        target_oh = torch.zeros(target.size(0), self.args.num_classes).to(target)
        target_oh.scatter_(1, target.view(-1, 1), 1)
        loss = F.binary_cross_entropy(probs[:, 1:], target_oh[:, 1:].float(), reduction='none').mean(dim=1)
        indices = self.bce_inference(probs)
        sample_size = target.numel()

        logging_output = {
            'loss': loss.detach().cpu().tolist(),
            # for computing F1
            'target': target.tolist(),
            'pred': indices.tolist(),
        }
        if logits.size(1) == 2:
            logging_output.update({
                'prob': probs[:, 1].tolist(),
            })
        if reduce:
            loss = loss.sum()
        return loss, sample_size, logging_output

    def bce_inference(self, probs):
        probs = probs.masked_fill(probs < self.threshold, -1.0)
        probs[:, 0] = 0.0
        max_p, indices = torch.max(probs, dim=1)
        return indices

    def get_prob(self, logits):
        return torch.sigmoid(logits)

    def inference(self, model, sample):
        logits = model(**sample['net_input'])
        probs = self.get_prob(logits)
        indices = self.bce_inference(probs)
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


        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                target, pred, labels=list(range(1, self.args.num_classes)), average='micro')
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


@register('ghmc')
class GHMCrossEntropy(BinaryCrossEntropy):
    @staticmethod
    def register(options):
        BinaryCrossEntropy.register(options)
        options.register(
            Argument('bins', default=10),
            Argument('momentum', default=0.75),

        )

    def __init__(self, args):
        super().__init__(args)
        self.loss_fn = GHMC_Loss(args.bins, args.momentum)

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
        target_oh = torch.zeros(target.size(0), self.args.num_classes).to(target)
        target_oh.scatter_(1, target.view(-1, 1), 1)

        logging_output = {}
        if self.args.use_r_dropout:
            logits2 = model(**sample['net_input'])
            p_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits2, dim=-1), reduction='none')
            q_loss = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits, dim=-1), reduction='none')
            p_loss = p_loss.sum(-1)
            q_loss = q_loss.sum(-1)
            kl_loss = (p_loss + q_loss) / 2
            ce_loss = 0.5 * (self.loss_fn(logits, target_oh) + self.loss_fn(logits2, target_oh))

            logging_output.update(
                {'ce_loss': ce_loss.detach().cpu().tolist(), 'kl_loss': kl_loss.detach().cpu().tolist()})
            loss = ce_loss + self.args.kl_alpha * kl_loss
        else:
            loss = self.loss_fn(logits, target_oh)

        probs = torch.sigmoid(logits)
        indices = self.bce_inference(probs)
        sample_size = target.numel()

        logging_output = {
            'loss': loss.detach().cpu().tolist(),
            # for computing F1
            'target': target.tolist(),
            'pred': indices.tolist(),
        }
        if logits.size(1) == 2:
            logging_output.update({
                'prob': probs[:, 1].tolist(),
            })
        if reduce:
            loss = loss.sum()
        return loss, sample_size, logging_output


class GHMC_Loss(nn.Module):
    def __init__(self, bins, momentum):
        super(GHMC_Loss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.last_bin_count = None

    def forward(self, logits, target):
        # gradient length
        g = torch.abs(torch.sigmoid(logits).detach() - target)
        bin_idx = torch.floor(g * (self.bins - 0.0001)).long()

        bin_count = torch.zeros(self.bins, device=logits.device)
        for i in range(self.bins):
            bin_count[i] = (bin_idx == i).sum().item()

        if self.last_bin_count is None:
            self.last_bin_count = bin_count
        else:
            bin_count = self.momentum * self.last_bin_count + (1 - self.momentum) * bin_count
            self.last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()  # n valid bins

        tot = logits.numel()
        beta = tot / torch.clamp(bin_count * nonempty_bins, min=0.0001)
        weight = beta[bin_idx]
        loss = F.binary_cross_entropy_with_logits(logits, target.float(), weight=weight, reduction='none')
        return loss.mean(dim=1)

