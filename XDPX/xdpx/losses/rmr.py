import torch
import torch.nn.functional as F
from xdpx.modules import cross_entropy
from . import register
from .bert import BertLoss
from xdpx.tasks import Task, register as register_task


@register('rmr')
class RMRLoss(BertLoss):
    """
    Random Message Replacement
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert reduce is True
        logging_output = {}
        targets = sample['target']
        cls_targets = sample['cls_target']
        # compute MLM loss
        masked_tokens = targets.ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()  # number of predicted tokens

        # (Rare case) When all tokens in a batch are not masked, the model results in empty tensor and gives error.
        # It happens when the sequence is short and the batch size is small.
        if sample_size == 0:
            masked_tokens = None

        seq_logits, cls_logits = model(**sample['net_input'], masked_tokens=masked_tokens)

        batch_size = len(cls_targets)
        sender_mask = sample['sender_mask']
        session_sizes = sample['session_size']
        max_size = session_sizes.max()
        session_mask = torch.arange(max_size)[None, :].to(session_sizes) < session_sizes[:, None]
        session = torch.zeros(batch_size, max_size, cls_logits.size(1)).to(cls_logits)
        session[session_mask] = cls_logits

        cls_logits_top = model.top_forward(session, session_mask).squeeze(2)  # (batch_size, max_size)
        cls_logits_top.masked_fill_(~sender_mask, -1e4)
        logging_output['cls_acc'] = cls_logits_top.max(1)[1].eq(cls_targets).float().mean().item()
        cls_targets_oh = torch.zeros_like(cls_logits_top)
        cls_targets_oh[torch.arange(batch_size, device=session.device), cls_targets] = 1
        cls_logits_top = cls_logits_top.masked_select(session_mask)
        cls_targets_oh = cls_targets_oh.masked_select(session_mask)
        sender_mask_top = cls_logits_top.gt(-1e4 + 1)  # in case of float point error
        cls_logits_top = cls_logits_top[sender_mask_top]
        cls_targets_oh = cls_targets_oh[sender_mask_top]

        if sample_size != 0:
            targets = targets[masked_tokens]
            seq_logits = seq_logits.view(-1, seq_logits.size(-1))
            assert targets.ndimension() == 1
            seq_loss = cross_entropy(
                seq_logits,
                targets,
                ignore_index=self.padding_idx,
            )
            logging_output['lm_loss'] = seq_loss.item()
        else:
            seq_loss = 0.0
            logging_output['lm_loss'] = seq_loss
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits_top, cls_targets_oh)
        loss = self.lm_weight * seq_loss + cls_loss
        logging_output.update({
            'loss': loss.item(),
            'cls_loss': cls_loss.item(),
        })
        topk = (1, 5)
        maxk = max(topk)
        if sample_size != 0:
            _, pred = seq_logits.topk(maxk, 1, True, True)  # B x k
            pred = pred.t()  # k x B
            correct = pred.eq(targets.view(1, -1).expand_as(pred))  # B x 1 -> 1 x B -> k x B
            for k in topk:
                if k == 1:
                    name = 'acc'
                else:
                    name = f'top{k}_acc'
                logging_output[name] = correct[:k].float().sum().item() / max(sample_size, 1)
        else:
            for k in topk:
                if k == 1:
                    name = 'acc'
                else:
                    name = f'top{k}_acc'
                logging_output[name] = 0.0
        prediction = torch.sigmoid(cls_logits_top).gt(1. / self.args.max_messages)
        cls_targets_oh = cls_targets_oh.to(prediction)
        tp = (prediction & cls_targets_oh).sum(dtype=torch.int)
        fp = (prediction & ~cls_targets_oh).sum(dtype=torch.int)
        fn = (~prediction & cls_targets_oh).sum(dtype=torch.int)

        logging_output.update(dict(
            tp=tp.item(),
            fp=fp.item(),
            fn=fn.item(),
        ))
        # loss and other stats are already averaged, so 1 is used as the sample_size
        return loss, 1, logging_output

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        agg_output = super().aggregate_logging_outputs(logging_outputs, sample_size, max_count)
        tp = sum(log['tp'] for log in logging_outputs)
        fp = sum(log['fp'] for log in logging_outputs)
        fn = sum(log['fn'] for log in logging_outputs)
        if tp > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2. * precision * recall / (precision + recall)
        else:
            precision = recall = f1 = 0.0
        agg_output['cls_precision'] = precision
        agg_output['cls_recall'] = recall
        agg_output['cls_f1'] = f1
        agg_output['cls_acc'] = sum(log['cls_acc'] for log in logging_outputs) / max(sample_size, 1)
        return agg_output

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True

    def inference(self, model, sample):
        """inference for pretrain diagnosis"""
        targets = sample['target']
        cls_targets = sample['cls_target']
        # compute MLM loss
        masked_tokens = targets.ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()  # number of predicted tokens
        if sample_size == 0:
            masked_tokens = None
        seq_logits, cls_logits = model(**sample['net_input'], masked_tokens=masked_tokens)

        seq_logits = seq_logits.view(-1, seq_logits.size(-1))
        seq_probs = F.softmax(seq_logits, dim=1)
        seq_prob, seq_pred = seq_probs.topk(5, 1)  # B x k
        mask = seq_prob.cumsum(1).lt(0.95)
        mask.scatter_(1, mask.sum(1, keepdim=True).clamp(0, mask.size(1) - 1), 1)
        seq_pred *= mask.to(seq_pred)

        # compute CLS loss
        batch_size = len(cls_targets)
        session_sizes = sample['session_size']
        max_size = session_sizes.max()
        session_mask = torch.arange(max_size)[None, :].to(session_sizes) < session_sizes[:, None]
        session = torch.zeros(batch_size, max_size, cls_logits.size(1)).to(cls_logits)
        session[session_mask] = cls_logits
        cls_logits_top = model.top_forward(session, session_mask).squeeze(2)  # (batch_size, max_size)
        cls_logits_top = cls_logits_top.masked_select(session_mask)
        cls_prob = torch.sigmoid(cls_logits_top)
        return seq_pred.tolist(), cls_prob.tolist()


@register_task('rmr')
class RMRDiagnosisTask(Task):
    def inference_step(self, sample, model, loss):
        # original text
        import re
        text = [self.processor.decode(sample_i)
                for sample_i in sample['net_input']['input_ids'].tolist()]
        target_ids = sample['target'].tolist()
        for text_i, target in zip(text, target_ids):
            for i, t in enumerate(target):
                if t > 0:
                    text_i[i] = '_'
        text = [' '.join(text_i).replace('[SEP]', '').replace('[CLS]', '').replace('[PAD]', '').strip()
                for text_i in text]
        target_text = [re.sub(r'\s+', ' ', ' '.join(self.processor.decode([tid for tid in target_id if tid > 0])))
                       for target_id in target_ids]
        session_size = sample['session_size'].tolist()
        session_text = []
        session_mask_target = []
        pos = 0
        for size in session_size:
            session_text.append(self.newline_concat(text[pos: pos + size]))
            session_mask_target.append(self.newline_concat(target_text[pos: pos + size]))
            pos += size
        target_counts = sample['target'].ne(self.args.pad_index).int().sum(1).tolist()

        # model prediction
        model.eval()
        with torch.no_grad():
           seq_pred, cls_prob = loss.inference(model, sample)
        mask_pred = []
        i = 0
        for n in target_counts:
            mask_pred.append(
                '/'.join(' '.join(self.processor.decode([p for p in topk if p > 0])) for topk in seq_pred[i: i + n]))
            i += n
        session_mask_pred = []
        pos = 0
        for size in session_size:
            session_mask_pred.append(self.newline_concat(mask_pred[pos: pos+size]))
            pos += size
        session_pred = []
        session_target = []
        session_prob = []
        pos = 0
        for size, t_idx, mask in zip(session_size, sample['cls_target'].tolist(), sample['sender_mask'].tolist()):
            p = cls_prob[pos: pos + size]
            for i in range(len(p)):
                if not mask[i]:
                    p[i] = -1
            idx = max(list(range(len(p))), key=p.__getitem__)
            session_pred.append(text[pos + idx])
            session_target.append(text[pos + t_idx])
            session_prob.append(self.newline_concat(f'{pi:.2f}' for pi in p))
            pos += size

        return session_text, session_mask_target, session_mask_pred, session_prob, session_target, session_pred

    @property
    def inference_header(self):
        return 'session_text session_mask_target session_mask_pred session_prob session_target session_pred'.split()

    @staticmethod
    def newline_concat(texts):
        # use double quotes to escape a single quote in csv
        return '"' + ('\n'.join(texts)).replace('"', '""') + '"'
