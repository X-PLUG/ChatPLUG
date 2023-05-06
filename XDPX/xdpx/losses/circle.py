import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from sklearn.metrics import roc_auc_score

from . import Loss, register
from xdpx.options import Argument
from xdpx.utils.versions import torch_lt_120


@register('circle')
class CircleLoss(Loss):
    """
    Circle Loss: A Unified Perspective of Pair Similarity Optimization
    https://arxiv.org/pdf/2002.10857.pdf
    """
    @staticmethod
    def register(options):
        options.register(
            Argument('scale', default=32),
            Argument('margin', default=0.25),
            Argument('predict_threshold', default=0.5),
            Argument('ce_weight', default=0.),
        )
        options.add_global_constraint(lambda args: args.num_classes == 2)

    def __init__(self, args):
        super().__init__(args)
        self.soft_plus = nn.Softplus()

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
        if torch_lt_120():
            target_mask = target.byte()
        else:
            target_mask = target.bool()
        similarity = torch.sigmoid(logits[:, 1])  # TODO: whether to add sigmoid

        m = self.args.margin
        if target_mask.any():
            sp = similarity.masked_select(target_mask)
            ap = torch.clamp_min(-sp.detach() + 1 + m, min=0.)
            logits_p = -ap * (sp - 1 + m) * self.args.scale
            loss_p = torch.logsumexp(logits_p, dim=0) / self.args.scale
        else:
            loss_p = 0.
        if not target_mask.all():
            sn = similarity.masked_select(~target_mask)
            an = torch.clamp_min(sn.detach() + m, min=0.)
            logits_n = an * (sn - m) * self.args.scale
            loss_n = torch.logsumexp(logits_n, dim=0) / self.args.scale
        else:
            loss_n = 0.
        # note that these (aggregated) losses will change for the same dev set when batch size changes,
        # which is expected
        logging_output = {
            'loss_p': loss_p.item() if loss_p > 0. else 0.,
            'loss_n': loss_n.item() if loss_n > 0. else 0.,
            'target': target.tolist(),
            'prob': similarity.tolist(),
        }
        loss = self.soft_plus(loss_n + loss_p)
        if self.args.ce_weight > 0:
            ce_loss = F.binary_cross_entropy(similarity, target.float())
            logging_output.update({
                'circle_loss': loss.item(),
                'ce_loss': ce_loss.item(),
            })
            loss = (loss + ce_loss * self.args.ce_weight) / (1. + self.args.ce_weight)
        logging_output['loss'] = loss.item()
        return loss, 1, logging_output

    def get_prob(self, logits):
        return torch.sigmoid(logits[:, 1])

    def inference(self, model, sample):
        logits = model(**sample['net_input'])
        probs = self.get_prob(logits)
        indices = probs.ge(self.args.predict_threshold).int()
        return indices.tolist(), torch.stack([1. - probs, probs], dim=1).tolist()

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        """Aggregate logging outputs from data parallel training."""

        agg_output = {
            'sample_size': sample_size,
        }
        if 'ntokens' in logging_outputs[0]:
            ntokens = sum(log['ntokens'] for log in logging_outputs)
            agg_output['ntokens'] = ntokens
        agg_output['loss'] = sum(log['loss'] for log in logging_outputs) / sample_size
        agg_output['loss_p'] = sum(log['loss_p'] for log in logging_outputs) / sample_size
        agg_output['loss_n'] = sum(log['loss_n'] for log in logging_outputs) / sample_size
        target = list(chain.from_iterable(log['target'] for log in logging_outputs))[:max_count]
        prob = list(chain.from_iterable(log['prob'] for log in logging_outputs))[:max_count]
        try:
            auc = roc_auc_score(target, prob)
        except ValueError:  # when batch size is small and only one class presented in targets
            auc = 0.0
        agg_output['auc'] = auc
        f1, p, r, threshold = self.auto_threshold_f1(target, prob)
        agg_output.update(dict(
            f1=f1, precision=p, recall=r, threshold=threshold,
        ))
        return agg_output

    @staticmethod
    def auto_threshold_f1(target, prob):
        data = list(zip(target, prob))
        data.sort(key=lambda x: x[1], reverse=True)
        tp = 0
        fp = 0
        fn = sum(target)
        best_f1 = -1
        best_p = -1
        best_r = -1
        threshold = None
        for i, (t, p) in enumerate(data):
            if t:
                tp += 1
                fn -= 1
            else:
                fp += 1
            assert fn >= 0
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best_f1 = f1
                best_p = tp / max(tp + fp, 1)
                best_r = tp / max(tp + fn, 1)
                if i < len(data) - 1:
                    threshold = (p + data[i + 1][1]) / 2
                else:
                    threshold = p
        assert threshold is not None
        return best_f1, best_p, best_r, threshold


@register('circle_siamese')
class CircleLossSiamese(CircleLoss):
    def forward(self, model, sample, logits=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_input = sample['net_input']
        feat1 = model.bert_forward(net_input['input_ids_1'])[1]
        feat2 = model.bert_forward(net_input['input_ids_2'])[1]
        similarity = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-5)

        target = sample['target']
        if torch_lt_120():
            target_mask = target.byte()
        else:
            target_mask = target.bool()

        m = self.args.margin
        if target_mask.any():
            sp = similarity.masked_select(target_mask)
            ap = torch.clamp_min(-sp.detach() + 1 + m, min=0.)
            logits_p = -ap * (sp - 1 + m) * self.args.scale
            loss_p = torch.logsumexp(logits_p, dim=0) / self.args.scale
        else:
            loss_p = 0.
        if not target_mask.all():
            sn = similarity.masked_select(~target_mask)
            an = torch.clamp_min(sn.detach() + m, min=0.)
            logits_n = an * (sn - m) * self.args.scale
            loss_n = torch.logsumexp(logits_n, dim=0) / self.args.scale
        else:
            loss_n = 0.
        logging_output = {
            'loss_p': loss_p.item() if loss_p > 0. else 0.,
            'loss_n': loss_n.item() if loss_n > 0. else 0.,
            'target': target.tolist(),
            'prob': similarity.tolist(),
        }
        loss = self.soft_plus(loss_n + loss_p)
        if self.args.ce_weight > 0:
            ce_loss = F.binary_cross_entropy(similarity, target.float())
            logging_output.update({
                'circle_loss': loss.item(),
                'ce_loss': ce_loss.item(),
            })
            loss = (loss + ce_loss * self.args.ce_weight) / (1. + self.args.ce_weight)
        logging_output['loss'] = loss.item()
        return loss, 1, logging_output
