import math
import torch
import torch.nn.functional as F
import numpy as np
from itertools import chain
from sklearn.metrics import roc_auc_score

from xdpx.options import Argument
from xdpx.tasks import Task, register as register_task
from xdpx.modules import cross_entropy
from xdpx.utils import numpy_seed
from . import register, Loss
from torch import nn


@register('bert')
class BertLoss(Loss):
    """
    Implementation for the loss used in masked language model (MLM) + sequence relation prediction (NSP, SOP...) training.
    """

    @staticmethod
    def register(options):
        Loss.register(options)
        options.register(
            Argument('lm_weight', default=1.0, validate=lambda value: value > 0.),
        )

    def __init__(self, args):
        super().__init__(args)
        self.padding_idx = args.pad_index
        self.lm_weight = args.lm_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert reduce is True
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
        logging_output = {}
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
        cls_loss = F.cross_entropy(cls_logits, cls_targets)
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
        logging_output['cls_acc'] = cls_targets.eq(cls_logits.argmax(-1)).float().mean().item()
        if cls_logits.size(1) == 2:
            logging_output.update({
                'cls_prob': F.softmax(cls_logits, 1)[:, 1].tolist(),
                'cls_target': cls_targets.tolist(),
            })
        # loss and other stats are already averaged, so 1 is used as the sample_size
        return loss, 1, logging_output

    def inference(self, model, sample):
        targets = sample['target']
        # compute MLM loss
        masked_tokens = targets.ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()  # number of predicted tokens

        # (Rare case) When all tokens in a batch are not masked, the model results in empty tensor and gives error.
        # It happens when the sequence is short and the batch size is small.
        if sample_size == 0:
            masked_tokens = None

        seq_logits, cls_logits = model(**sample['net_input'], masked_tokens=masked_tokens)
        seq_logits = seq_logits.view(-1, seq_logits.size(-1))
        seq_probs = F.softmax(seq_logits, dim=1)

        seq_prob, seq_pred = seq_probs.topk(5, 1)  # B x k
        mask = seq_prob.cumsum(1).lt(0.95)
        mask.scatter_(1, mask.sum(1, keepdim=True).clamp(0, mask.size(1) - 1), 1)
        seq_pred *= mask.to(seq_pred)
        cls_prob = F.softmax(cls_logits, dim=-1)
        cls_probs, cls_pred = torch.max(cls_prob, dim=1)
        return seq_pred.tolist(), cls_pred.tolist(), cls_probs.tolist()

    def distill(self, model, sample):
        _, cls_logits = model(**sample['net_input'])
        return cls_logits.tolist()

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        lm_loss = sum(log.get('lm_loss', 0) for log in logging_outputs)
        cls_loss = sum(log.get('cls_loss', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / max(sample_size, 1) / math.log(2),
            'lm_loss': lm_loss / max(sample_size, 1) / math.log(2),
            'cls_loss': cls_loss / max(sample_size, 1) / math.log(2),
            'sample_size': sample_size,
        }
        if 'ntokens' in logging_outputs[0]:
            ntokens = sum(log['ntokens'] for log in logging_outputs)
            agg_output['ntokens'] = ntokens
        for key in logging_outputs[0].keys():
            if 'acc' in key:
                agg_output[key] = sum(log.get(key, 0) for log in logging_outputs) / max(sample_size, 1)
        if 'cls_prob' in logging_outputs[0]:
            cls_prob = list(chain.from_iterable(log['cls_prob'] for log in logging_outputs))
            cls_target = list(chain.from_iterable(log['cls_target'] for log in logging_outputs))
            try:
                cls_auc = roc_auc_score(cls_target, cls_prob)
            except ValueError:  # when batch size is small and only one class presented in targets
                cls_auc = 0.0
            agg_output['cls_auc'] = cls_auc
        return agg_output


@register_task('slm')
class BertPretrainTask(Task):
    def inference_step(self, sample, model, loss):
        model.eval()
        orig_text = [''.join(self.processor.decode(sample_i)).replace('[SEP]', '\t') for sample_i in
                     sample['orig_input_ids']]
        text = [''.join(self.processor.decode(sample_ids)).replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]',
                                                                                                             '\t').strip()
                for sample_ids in sample['net_input']['input_ids'].tolist()]
        target_counts = sample['target'].ne(self.args.pad_index).int().sum(1).tolist()
        assert len(text) == len(target_counts)
        with torch.no_grad():
            seq_pred, cls_pred, cls_probs = loss.inference(model, sample)
        mask_pred = []
        i = 0
        for n in target_counts:
            mask_pred.append(
                ' '.join('/'.join(self.processor.decode([p for p in topk if p > 0])) for topk in seq_pred[i: i + n]))
            i += n
        try:
            cls_targets = sample['cls_target'].tolist()
        except KeyError:
            cls_targets = ['' for _ in orig_text]
        return orig_text, cls_targets, text, mask_pred, cls_pred, cls_probs

    @property
    def inference_header(self):
        return 'origin_text cls_targets input_text mask_pred cls_pred cls_probs'.split()

    def distill_step(self, sample, model, loss):
        model.eval()
        orig_text = [''.join(self.processor.decode(sample_i)).replace('[SEP]', '\t') for sample_i in
                     sample['orig_input_ids']]
        with torch.no_grad():
            logits = loss.distill(model, sample)
        try:
            targets = sample['cls_target'].tolist()
        except KeyError:
            targets = ['' for _ in orig_text]
        return orig_text, targets, logits

    def build_dataset(self, data: list, is_train: bool):
        if not is_train:
            # pre-shuffle dev set to make NSP more reasonable
            with numpy_seed(self.args.seed):
                np.random.shuffle(data)

        return super().build_dataset(data, is_train)


@register('bert_pretrain_cl')
class BertLossCL(BertLoss):
    """
    Implementation for the loss used in masked language model (MLM) + contrastive training.
    """

    @staticmethod
    def register(options):
        BertLoss.register(options)
        options.register(
            Argument('temperature', default=0.05, validate=lambda value: value > 0.),
        )

    def __init__(self, args):
        super().__init__(args)
        self.padding_idx = args.pad_index
        self.lm_weight = args.lm_weight
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert reduce is True
        targets = sample['target']
        # compute MLM loss
        masked_tokens = targets.ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()  # number of predicted tokens

        # (Rare case) When all tokens in a batch are not masked, the model results in empty tensor and gives error.
        # It happens when the sequence is short and the batch size is small.
        if sample_size == 0:
            masked_tokens = None

        seq_logits, embedding_z1 = model(**sample['net_input'], masked_tokens=masked_tokens)
        logging_output = {}
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

        # infonce loss
        new_param = {'input_ids': sample['orig_input_ids']}
        _, embedding_z2 = model(**new_param)
        cls_logits = self.similarity(embedding_z1.unsqueeze(1), embedding_z2.unsqueeze(0)) / self.args.temperature
        cls_targets = torch.arange(cls_logits.size(0)).long().to(cls_logits.device)
        loss_fct = nn.CrossEntropyLoss()
        cls_loss = loss_fct(cls_logits, cls_targets)

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
        logging_output['cls_acc'] = cls_targets.eq(cls_logits.argmax(-1)).float().mean().item()
        if cls_logits.size(1) == 2:
            logging_output.update({
                'cls_prob': F.softmax(cls_logits, 1)[:, 1].tolist(),
                'cls_target': cls_targets.tolist(),
            })
        # loss and other stats are already averaged, so 1 is used as the sample_size
        return loss, 1, logging_output


@register('bert_prompt_loss')
class BertPromptLoss(BertLoss):

    @staticmethod
    def register(options):
        BertLoss.register(options)
        options.register(
            Argument('use_r_dropout', default=True),
            Argument('kl_alpha', default=0.5, doc='when r-dropout=true '),
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert reduce is True

        targets = sample['target']
        # compute MLM loss
        masked_tokens = targets.ne(self.padding_idx)
        logging_output = {}
        logits = model(**sample['net_input'], masked_tokens=masked_tokens)

        targets = targets[masked_tokens]
        assert targets.ndimension() == 1
        logits = logits.view(-1, logits.size(-1))
        loss = cross_entropy(
            logits,
            targets,
            ignore_index=self.padding_idx,
        )
        logging_output['ce_loss'] = loss.item()

        label_ids = sample['label_ids'][0]
        ext_count = sample['label_ids'][1]
        batch_size = sample['net_input']['input_ids'].shape[0]

        vocab_size = logits.shape[-1]
        prefix_size = sample['prefix_size']
        label_count = label_ids.shape[0]
        y_true_index = sample['target_id'][::prefix_size]

        def logits_to_probs(logits):
            prediction_probs = F.softmax(logits, dim=-1)
            prediction_probs = torch.reshape(
                prediction_probs, shape=[batch_size, -1, vocab_size])

            probs = torch.ones(size=[batch_size, label_count], device=logits.device)
            # Calculate joint distribution of candidate labels
            for index in range(self.args.label_length):
                probs *= prediction_probs[:, index, label_ids[:, index]]

            assert batch_size % prefix_size == 0, 'batch_size % prefix_size == 0'
            probs = probs.reshape([batch_size // prefix_size, prefix_size, label_count // ext_count, ext_count])
            probs = torch.max(torch.mean(probs, dim=1), dim=-1).values
            return probs

        if self.args.use_r_dropout:
            logits2 = model(**sample['net_input'], masked_tokens=masked_tokens)
            logits2 = logits2.view(-1, logits2.size(-1))
            loss2 = cross_entropy(
                logits2,
                targets,
                ignore_index=self.padding_idx,
            )

            ce_loss = (loss + loss2) * 0.5
            logging_output['ce_loss'] = ce_loss.item()

            p_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits2, dim=-1), reduction='none')
            q_loss = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits, dim=-1), reduction='none')
            p_loss = p_loss.mean()
            q_loss = q_loss.mean()
            kl_loss = (p_loss + q_loss) / 2

            logging_output['kl_loss'] = kl_loss.item()
            loss = ce_loss + kl_loss * self.args.kl_alpha

        logging_output.update({
            'loss': loss.item(),
        })

        # Get max probs label's index
        probs = logits_to_probs(logits)
        max_p, indices = torch.max(probs, dim=1)
        y_pred_index = np.array(indices.tolist())
        y_true_index = np.array(y_true_index.tolist())

        total_num = len(y_true_index)
        correct_num = (y_true_index == y_pred_index).sum()
        acc = correct_num / total_num

        logging_output.update({
            'acc': acc,
        })

        sample_size = masked_tokens.int().sum().item()  # number of predicted tokens
        topk = (1, 5)
        maxk = max(topk)

        _, pred = logits.topk(maxk, 1, True, True)  # B x k
        pred = pred.t()  # k x B
        correct = pred.eq(targets.view(1, -1).expand_as(pred))  # B x 1 -> 1 x B -> k x B
        for k in topk:
            if k == 1:
                name = 'mlm_acc'
            else:
                name = f'mlm_top{k}_acc'
            logging_output[name] = correct[:k].float().sum().item() / max(sample_size, 1)
        # loss and other stats are already averaged, so 1 is used as the sample_size
        return loss, 1, logging_output

    def inference(self, model, sample):
        input_ids = sample['net_input']['input_ids']
        masked_tokens = input_ids.eq(self.args.mask_index)
        logits = model(**sample['net_input'], masked_tokens=masked_tokens)

        label_ids = sample['label_ids'][0]
        ext_count = sample['label_ids'][1]
        batch_size = sample['net_input']['input_ids'].shape[0]

        vocab_size = logits.shape[-1]
        prefix_size = sample['prefix_size']
        label_count = label_ids.shape[0]

        prediction_probs = F.softmax(logits, dim=-1)
        prediction_probs = torch.reshape(
            prediction_probs, shape=[batch_size, -1, vocab_size])

        probs = torch.ones(size=[batch_size, label_count], device=logits.device)
        # Calculate joint distribution of candidate labels
        for index in range(self.args.label_length):
            probs *= prediction_probs[:, index, label_ids[:, index]]

        assert batch_size % prefix_size == 0, 'batch_size % prefix_size == 0'
        probs = probs.reshape([batch_size // prefix_size, prefix_size, label_count // ext_count, ext_count])
        probs = torch.max(torch.mean(probs, dim=1), dim=-1).values

        # Get max probs label's index
        max_p, indices = torch.max(probs, dim=1)
        y_pred_index = np.array(indices.tolist())

        return y_pred_index.tolist(), probs.tolist()

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        lm_loss = sum(log.get('lm_loss', 0) for log in logging_outputs)
        cls_loss = sum(log.get('cls_loss', 0) for log in logging_outputs)
        ce_loss = sum(log.get('ce_loss', 0) for log in logging_outputs)
        kl_loss = sum(log.get('kl_loss', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / max(sample_size, 1) / math.log(2),
            'lm_loss': lm_loss / max(sample_size, 1) / math.log(2),
            'cls_loss': cls_loss / max(sample_size, 1) / math.log(2),
            'ce_loss': ce_loss / max(sample_size, 1) / math.log(2),
            'kl_loss': kl_loss / max(sample_size, 1) / math.log(2),
            'sample_size': sample_size,
        }
        if 'ntokens' in logging_outputs[0]:
            ntokens = sum(log['ntokens'] for log in logging_outputs)
            agg_output['ntokens'] = ntokens
        for key in logging_outputs[0].keys():
            if 'acc' in key:
                agg_output[key] = sum(log.get(key, 0) for log in logging_outputs) / max(sample_size, 1)
        if 'cls_prob' in logging_outputs[0]:
            cls_prob = list(chain.from_iterable(log['cls_prob'] for log in logging_outputs))
            cls_target = list(chain.from_iterable(log['cls_target'] for log in logging_outputs))
            try:
                cls_auc = roc_auc_score(cls_target, cls_prob)
            except ValueError:  # when batch size is small and only one class presented in targets
                cls_auc = 0.0
            agg_output['cls_auc'] = cls_auc
        return agg_output
