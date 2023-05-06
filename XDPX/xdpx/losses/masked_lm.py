import re
import math
import torch
import torch.nn.functional as F

from . import register, Loss


@register('masked_lm')
class MaskedLMLoss(Loss):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args):
        super().__init__(args)
        self.padding_idx = args.pad_index

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        targets = sample['target']
        # compute MLM loss
        masked_tokens = targets.ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()  # number of predicted tokens

        # (Rare case) When all tokens in a batch are not masked, the model results in empty tensor and gives error.
        # It happens when the sequence is short and the batch size is small.
        if sample_size == 0:
            masked_tokens = None

        logits = model(**sample['net_input'], masked_tokens=masked_tokens)
        logits = logits.view(-1, logits.size(-1))
        if sample_size != 0:
            targets = targets[masked_tokens]
        else:
            targets = targets.view(-1)
        assert targets.ndimension() == 1
        loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            targets,
            reduction='sum',  # loss is finally normalized in the distributed trainer by aggregated sample sizes
            ignore_index=self.padding_idx,  # ignore masked (padding) tokens
        )
        logging_output = {
            'loss': loss.item() if reduce else loss.detach(),
        }
        topk = (1, 5)
        maxk = max(topk)
        _, pred = logits.topk(maxk, 1, True, True)  # B x k
        pred = pred.t()  # k x B
        correct = pred.eq(targets.view(1, -1).expand_as(pred))  # B x 1 -> 1 x B -> k x B
        for k in topk:
            logging_output[f'correct_{k}'] = correct[:k].float().sum().item()
        return loss, sample_size, logging_output
    
    def inference(self, model, sample):
        targets = sample['target']
        masked_tokens = targets.ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()
        if sample_size == 0:
            return []
        logits = model(**sample['net_input'], masked_tokens=masked_tokens)
        prediction = torch.argmax(logits, dim=1)
        targets = targets[masked_tokens]
        return prediction.detach(), targets.detach()

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / max(sample_size, 1) / math.log(2),
            'sample_size': sample_size,
        }
        if 'ntokens' in logging_outputs[0]:
            ntokens = sum(log['ntokens'] for log in logging_outputs)
            agg_output['ntokens'] = ntokens
        template = 'top{}_acc'
        for key in logging_outputs[0].keys():
            m = re.match(r'correct_(\d+)', key)
            
            if m:
                k = int(m.group(1))
                agg_output[template.format(k)] = sum(log.get(key, 0) for log in logging_outputs) / max(sample_size, 1)
        acc_alias = template.format(1)
        if acc_alias in agg_output:
            agg_output['acc'] = agg_output.pop(acc_alias)
        return agg_output
    
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `aggregate_logging_outputs`. Setting this
        to True will improves distributed training speed.
        """
        return True
