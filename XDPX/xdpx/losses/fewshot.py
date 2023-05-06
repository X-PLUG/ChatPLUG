import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from collections import defaultdict
from typing import List
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

from . import Loss, register
from xdpx.options import Argument
from xdpx.modules import cross_entropy


@register('fewshot')
class FewshotCrossEntropyLoss(Loss):
    @staticmethod
    def register(options):
        options.register(
            Argument('negative_classes_in_f1', default=[0], type=List[int],
                     doc='negative classes for micro F1 score; if emtpy, no F1 score will be computed.'),
            Argument('predict_threshold', type=float),
            Argument('query_level_f1', default=False, doc='should be used with loader=="rank"'),
            Argument('use_r_dropout', default=False, doc='use r-dropout'),
            Argument('kl_alpha', default=0.5, doc='when r-dropout=true '),
            Argument('lm_weight', default=1.0, doc='mlm loss weight'),
            Argument('cl_alpha', default=0.0, doc='contrastive loss weight'),
            Argument('cl_loss_type', doc='contrastive loss type', default='infonce',
                     validate=lambda val: val in ['infonce', 'dcl'])
        )
        options.add_global_constraint(
            lambda args: max(args.negative_classes_in_f1) < args.num_classes,
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        """
        assert reduce is True
        query_input_ids = sample['net_input']['query_input_ids']
        support_input_ids = sample['net_input']['support_input_ids']
        N, K = support_input_ids.shape[:2]
        support_shot_masks = torch.any(support_input_ids.ne(self.args.pad_index), dim=-1)

        temp = self.args.temperature if hasattr(self.args, 'temperature') else 0

        mlm_targets = sample['mlm_targets'] if 'mlm_targets' in sample else None
        if mlm_targets is not None and self.args.lm_weight > 0:
            masked_tokens = mlm_targets.ne(self.args.pad_index)
            mlm_sample_size = masked_tokens.int().sum().item()  # number of predicted tokens
            if mlm_sample_size == 0:
                masked_tokens = None
        else:
            masked_tokens = None
            mlm_sample_size = 0

        logits, support_emb, query_emb, \
            proto_logits, prompt_logits, prompt_masked_logits, masked_logits = model(**sample['net_input'],
                                                                         masked_tokens=masked_tokens)

        logging_output = {}

        # compute MLM loss
        if mlm_sample_size != 0:
            mlm_targets = mlm_targets[masked_tokens]
            seq_logits = masked_logits.view(-1, masked_logits.size(-1))
            assert mlm_targets.ndimension() == 1
            seq_loss = cross_entropy(
                seq_logits,
                mlm_targets,
                ignore_index=self.args.pad_index,
            )
            logging_output['lm_loss'] = seq_loss.item()
        else:
            seq_loss = 0.0
            logging_output['lm_loss'] = seq_loss

        prompt_targets = sample['support_prompt_targets'] if 'support_prompt_targets' in sample else None
        if prompt_masked_logits is not None and prompt_targets is not None:
                masked_tokens = prompt_targets.ne(self.args.pad_index)
                prompt_targets = prompt_targets[masked_tokens]
                prompt_masked_logits = prompt_masked_logits.view(-1, prompt_masked_logits.size(-1))
                assert prompt_targets.ndimension() == 1
                prompt_loss = cross_entropy(
                    prompt_masked_logits,
                    prompt_targets,
                    ignore_index=self.args.pad_index,
                )
                logging_output['prompt_loss'] = prompt_loss.item()
        else:
            prompt_loss = 0.0
            logging_output['prompt_loss'] = 0.0

        topk = (1, 5)
        maxk = max(topk)
        if mlm_sample_size != 0:
            _, pred = masked_logits.topk(maxk, 1, True, True)  # B x k
            pred = pred.t()  # k x B
            correct = pred.eq(mlm_targets.view(1, -1).expand_as(pred))  # B x 1 -> 1 x B -> k x B
            for k in topk:
                if k == 1:
                    name = 'mlm_acc'
                else:
                    name = f'top{k}_mlm_acc'
                logging_output[name] = correct[:k].float().sum().item() / max(mlm_sample_size, 1)
        else:
            for k in topk:
                if k == 1:
                    name = 'mlm_acc'
                else:
                    name = f'top{k}_mlm_acc'
                logging_output[name] = 0.0

        if query_input_ids is not None:
            target = sample['query_targets']
            shot_masks = None
        else:
            batch_size = logits.shape[0]  # NK'
            query_shot = batch_size // N
            target = torch.arange(N).long().to(logits.device).repeat(query_shot, 1).permute(1, 0).reshape(
                N * query_shot)  # NK
            shot_masks = support_shot_masks[:, :query_shot].flatten() # NK'

        loss_fct = nn.CrossEntropyLoss(reduction='none')

        if self.args.use_r_dropout:
            logits2 = model(**sample['net_input'])[0]
            p_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits2, dim=-1), reduction='none')
            q_loss = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits, dim=-1), reduction='none')
            p_loss = p_loss.sum(-1)
            q_loss = q_loss.sum(-1)
            kl_loss = (p_loss + q_loss) / 2
            ce_loss = 0.5 * (loss_fct(logits, target)
                             + loss_fct(logits2, target))

            logging_output.update(
                {'ce_loss': ce_loss.detach().cpu().tolist(), 'kl_loss': kl_loss.detach().cpu().tolist()})
            loss = ce_loss + self.args.kl_alpha * kl_loss

        else:
            loss = loss_fct(logits, target)  # prototype contrastive loss , NK

        if self.args.cl_alpha > 0.0 and temp > 0.0:
            cl_loss = self._cl_loss(support_emb, N, K, support_shot_masks, temp)
            loss += self.args.cl_alpha * cl_loss
            logging_output.update(
                {'cl_loss': cl_loss.detach().cpu().tolist()})

        max_p, indices = torch.max(logits, dim=1)
        if shot_masks is not None:
            target = torch.masked_select(target, shot_masks)
            loss = torch.masked_select(loss, shot_masks)
            indices = torch.masked_select(indices, shot_masks)
            max_p = torch.masked_select(max_p, shot_masks)

        threshold = getattr(self.args, 'predict_threshold', None)
        if threshold is not None:
            indices = indices.where(max_p > threshold,
                                    torch.tensor(self.args.negative_classes_in_f1[0]).to(indices))
        sample_size = target.numel()

        loss = loss.mean()
        #loss = loss + self.args.lm_weight * seq_loss
        loss = loss + prompt_loss

        logging_output.update({
            'loss': loss.item(),
            'sample_size': sample_size,
            'target': target.tolist(),
            'pred': indices.tolist(),
        })

        # loss and other stats are already averaged, so 1 is used as the sample_size
        return loss, 1, logging_output

    def _cl_loss(self, support_emb, N, K, support_shot_masks, temp=0.05):
        dim = support_emb.shape[-1]
        support_emb = support_emb.reshape(-1, dim)  # NK * D
        z1 = support_emb.unsqueeze(1).repeat(1, N * K, 1)  # NK * NK * D
        z2 = support_emb.unsqueeze(0).repeat(N * K, 1, 1)  # NK * NK * D

        cos_sim = F.cosine_similarity(z1, z2, dim=-1, eps=1e-4) / temp  # NK * NK , eps should not be too small for fp16
        numerator_masks = torch.eye(N, device=z1.device). \
                              repeat(K, K, 1, 1).permute(2, 0, 3, 1).reshape(
            N * K, N * K) - torch.eye(
            N * K, device=z1.device)  # NK * NK

        denominator_masks = 1 - torch.eye(N * K, device=z1.device)
        negative_masks = 1 - numerator_masks - torch.eye(N * K, device=z1.device)

        shot_masks = support_shot_masks.flatten()
        shot_masks_expand = shot_masks.unsqueeze(1) * shot_masks.unsqueeze(0)  # NK * NK

        numerator_masks = numerator_masks * shot_masks_expand
        denominator_masks = denominator_masks * shot_masks_expand
        negative_masks = negative_masks * shot_masks_expand

        SMALL_NUM = -1e2
        if self.args.cl_loss_type == 'infonce':
            positive_loss = - torch.logsumexp(cos_sim + (1 - numerator_masks) * SMALL_NUM, 1)  # NK
            negative_loss = torch.logsumexp(cos_sim + (1 - denominator_masks) * SMALL_NUM, 1)  # NK
            loss = positive_loss + negative_loss
        elif self.args.cl_loss_type == 'dcl':
            #  Decoupled Contrastive Learning  https://arxiv.org/abs/2110.06848
            # https://github.com/raminnakhli/Decoupled-Contrastive-Learning/blob/main/loss/dcl.py
            positive_loss = - torch.logsumexp(cos_sim + (1 - numerator_masks) * SMALL_NUM, 1)  # NK
            negative_loss = torch.logsumexp(cos_sim + (1 - negative_masks) * SMALL_NUM, 1)  # NK
            loss = positive_loss + negative_loss
        return loss

    def inference(self, model, sample):
        query_input_ids = sample['net_input']['query_input_ids']
        support_input_ids = sample['net_input']['support_input_ids']
        support_shot_masks = torch.any(support_input_ids.ne(self.args.pad_index), dim=-1)

        logits = model(**sample['net_input'])[0]
        logits = F.log_softmax(logits, dim=-1)
        max_p, indices = torch.max(logits, dim=1)

        if not query_input_ids:
            shot_masks = support_shot_masks.flatten()  # NK
            indices = torch.masked_select(indices, shot_masks)
            max_p = torch.masked_select(max_p, shot_masks)
        return indices.tolist(), max_p.tolist()

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

        target = list(chain.from_iterable(log['target'] for log in logging_outputs))
        pred = list(chain.from_iterable(log['pred'] for log in logging_outputs))
        accuracy = accuracy_score(target, pred)
        agg_output['acc'] = accuracy

        loss = sum(log.get('loss', 0) for log in logging_outputs)
        lm_loss = sum(log.get('lm_loss', 0) for log in logging_outputs)
        prompt_loss = sum(log.get('prompt_loss', 0) for log in logging_outputs)
        agg_output.update({
            'loss': loss / max(sample_size, 1) / math.log(2),
            'lm_loss': lm_loss / max(sample_size, 1) / math.log(2),
            'prompt_loss': prompt_loss / max(sample_size, 1) / math.log(2)
        })

        for key in logging_outputs[0].keys():
            if 'mlm_acc' in key:
                agg_output[key] = sum(log.get(key, 0) for log in logging_outputs) / max(sample_size, 1)

        if 'ce_loss' in logging_outputs[0]:
            ce_loss = list(chain.from_iterable(log['ce_loss'] for log in logging_outputs))
            agg_output['ce_loss'] = sum(ce_loss) / len(ce_loss) / math.log(2) if sample_size > 0 else 0.

        if 'kl_loss' in logging_outputs[0]:
            kl_loss = list(chain.from_iterable(log['kl_loss'] for log in logging_outputs))
            agg_output['kl_loss'] = sum(kl_loss) / len(kl_loss) / math.log(2) if sample_size > 0 else 0.

        if 'cl_loss' in logging_outputs[0]:
            cl_loss = list(chain.from_iterable(log['cl_loss'] for log in logging_outputs))
            agg_output['cl_loss'] = sum(cl_loss) / len(cl_loss) / math.log(2) if sample_size > 0 else 0.
        return agg_output


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss
