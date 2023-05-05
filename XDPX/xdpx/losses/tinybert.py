import math
import torch
import torch.nn.functional as F
from itertools import chain
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, f1_score
from torch.nn import KLDivLoss

from xdpx.options import Argument
from . import register, Loss


@register('tinybert')
class TinyBertBertLoss(Loss):
    """
    Implementation for the loss used in masked language model (MLM) + sequence relation prediction (NSP, SOP...) training.
    """
    @staticmethod
    def register(options):
        Loss.register(options)
        options.register(
            Argument('lm_weight', default=1.0, validate=lambda value: value > 0.),
            Argument('attention_distill', default=True,
                     doc="use attention distill or not"),
            Argument('hidden_distill', default=True,
                     doc="use predict distill or not"),
            # 下面两个Loss打开效果会略微提高，但是会增加耗时
            Argument('mlm_predict_distill', default=False,
                     doc="use predict distill or not"),
            Argument('nsp_predict_distill', default=False,
                     doc="use predict distill or not"),
        )

    def __init__(self, args):
        super().__init__(args)
        self.padding_idx = args.pad_index
        self.lm_weight = args.lm_weight

    def attention_loss(self, raw_attentions, teacher_raw_attentions, attention_mask):
        teacher_layer_num = len(teacher_raw_attentions)
        student_layer_num = len(raw_attentions)
        layers_per_block = int(teacher_layer_num / student_layer_num)
        last_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        last_attention_mask = last_attention_mask.float()
        third_attention_mask = last_attention_mask.permute(0, 1, 3, 2)
        square_attention_mask = third_attention_mask * last_attention_mask
        valid_pos_num = None
        cur_loss = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for layer_id, layer_raw_attentions in enumerate(raw_attentions):
            teacher_layer_id = (layer_id + 1) * layers_per_block - 1
            layer_teacher_raw_attentions = teacher_raw_attentions[teacher_layer_id]
            layer_diff = layer_teacher_raw_attentions - layer_raw_attentions
            layer_diff = (layer_diff ** 2) * square_attention_mask
            if valid_pos_num is None:
                valid_pos = torch.ones(layer_diff.size()).to(device)
                valid_pos = valid_pos * square_attention_mask
                valid_pos_num = torch.sum(valid_pos)
            layer_mse = torch.sum(layer_diff) / valid_pos_num
            cur_loss += layer_mse
        return cur_loss / len(raw_attentions)

    def hidden_loss(self, hidden_states, teacher_hidden_states, attention_mask):
        teacher_layer_num = len(teacher_hidden_states) - 1
        student_layer_num = len(hidden_states) - 1
        layers_per_block = int(teacher_layer_num / student_layer_num)
        last_attention_mask = attention_mask.unsqueeze(2).float()
        valid_pos_num = None
        cur_loss = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for layer_id, layer_hidden_states in enumerate(hidden_states):
            # 0 layer is the embedding + position embedding
            teacher_layer_id = layer_id * layers_per_block
            layer_teacher_hidden_states = teacher_hidden_states[teacher_layer_id]
            layer_diff = layer_teacher_hidden_states - layer_hidden_states
            layer_diff = (layer_diff ** 2) * last_attention_mask
            if valid_pos_num is None:
                valid_pos = torch.ones(layer_diff.size()).to(device)
                valid_pos = valid_pos * last_attention_mask
                valid_pos_num = torch.sum(valid_pos)
            layer_mse = torch.sum(layer_diff) / valid_pos_num
            cur_loss += layer_mse
        return cur_loss / len(hidden_states)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        targets = sample.get('target', None)
        cls_targets = sample.get('cls_target', None)
        # compute MLM loss

        input_ids = sample['net_input']['input_ids']
        attention_mask = sample['attention_mask'] if 'attention_mask' in sample\
            else input_ids.ne(self.args.pad_index).long()
        teacher_outputs, student_outputs = model(**sample['net_input'])

        attention_loss = 0.0
        if self.args.attention_distill:
            attention_loss = self.attention_loss(student_outputs[-1], teacher_outputs[-1], attention_mask)

        hidden_loss = 0.0
        if self.args.hidden_distill:
            hidden_loss = self.hidden_loss(student_outputs[-2], teacher_outputs[-2], attention_mask)

        mlm_distill_loss = 0.0
        if self.args.mlm_predict_distill:
            mask_lm_fct = KLDivLoss(reduction='none')
            distill_lm_loss = mask_lm_fct(
                F.log_softmax(student_outputs[0], dim=-1),
                F.softmax(teacher_outputs[0], dim=-1))
            distill_lm_loss = distill_lm_loss.sum(-1)
            mask = (targets != self.args.pad_index).float()
            mlm_distill_loss = (distill_lm_loss * mask).sum() / mask.sum()

        nsp_distill_loss = 0.0
        if cls_targets is not None and self.args.nsp_predict_distill:
            nsp_label_num = student_outputs[1].size(-1)
            nsp_loss_fct = KLDivLoss(reduction="batchmean")
            nsp_distill_loss = nsp_loss_fct(
                F.log_softmax(student_outputs[1].view(-1, nsp_label_num), dim=-1),
                F.softmax(teacher_outputs[1].view(-1, nsp_label_num), dim=-1))

        if targets is not None and self.args.mlm_predict_distill:
            vocab_size = teacher_outputs[0].size(-1)
            targets = targets.view(-1)
            teacher_mlm_loss = F.cross_entropy(
                teacher_outputs[0].view(-1, vocab_size), targets, ignore_index=self.padding_idx
            )

            student_mlm_loss = F.cross_entropy(
                student_outputs[0].view(-1, vocab_size), targets, ignore_index=self.padding_idx
            )
        if cls_targets is not None and self.args.nsp_predict_distill:
            teacher_cls_loss = F.cross_entropy(teacher_outputs[1], cls_targets)
            student_cls_loss = F.cross_entropy(student_outputs[1], cls_targets)

        loss = attention_loss + hidden_loss + mlm_distill_loss + nsp_distill_loss
        logging_output = {
            'loss': loss.item() if reduce else loss.detach(),
        }
        if cls_targets is not None and self.args.nsp_predict_distill:
            logging_output.update({
                'teacher_cls_loss': teacher_cls_loss.item() if reduce else teacher_cls_loss.detach(),
                'student_cls_loss': student_cls_loss.item() if reduce else student_cls_loss.detach()
            })
        if targets is not None and self.args.mlm_predict_distill:
                logging_output.update({
                'teacher_mlm_loss': teacher_mlm_loss.item() if reduce else teacher_mlm_loss.detach(),
                'student_mlm_loss': student_mlm_loss.item() if reduce else student_mlm_loss.detach(),
            })
        if self.args.mlm_predict_distill:
            logging_output.update({
                'mlm_distill_loss': mlm_distill_loss.item() if reduce else mlm_distill_loss.detach(),
            })
        if self.args.attention_distill:
            logging_output.update({
                'attention_loss': attention_loss.item() if reduce else attention_loss.detach(),
            })
        if self.args.hidden_distill:
            logging_output.update({
                'hidden_loss': hidden_loss.item() if reduce else hidden_loss.detach(),
            })
        return loss, 1, logging_output

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        """Aggregate logging outputs from data parallel training."""
        keys = logging_outputs[0].keys()
        agg_output = {}
        for key in keys:
            sum_value = sum(log.get(key, 0) for log in logging_outputs)
            agg_output[key] = sum_value * 1.0 / sample_size
        agg_output['sample_size'] = sample_size
        return agg_output


@register('tinybert_td')
class TinyBertClsTaskDistillLoss(TinyBertBertLoss):
    def inference(self, model, sample):
        teacher_outputs, student_outputs = model(**sample['net_input'])
        teacher_probs = F.softmax(teacher_outputs[0], dim=-1)
        student_probs = F.softmax(student_outputs[0], dim=-1)
        teacher_max_p, teacher_indices = torch.max(teacher_probs, dim=1)
        student_max_p, student_indices = torch.max(student_probs, dim=1)
        return student_indices.tolist(), student_probs.tolist()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        cls_targets = sample.get('target', None)
        # compute MLM loss

        input_ids = sample['net_input']['input_ids']
        attention_mask = sample['attention_mask'] if 'attention_mask' in sample \
            else input_ids.ne(self.args.pad_index).long()
        teacher_outputs, student_outputs = model(**sample['net_input'])

        attention_loss = 0.0
        if self.args.attention_distill:
            attention_loss = self.attention_loss(student_outputs[-1], teacher_outputs[-1], attention_mask)

        hidden_loss = 0.0
        if self.args.hidden_distill:
            hidden_loss = self.hidden_loss(student_outputs[-2], teacher_outputs[-2], attention_mask)

        nsp_distill_loss = 0.0
        if cls_targets is not None and self.args.nsp_predict_distill:
            nsp_label_num = student_outputs[0].size(-1)
            nsp_loss_fct = KLDivLoss(reduction="batchmean")
            nsp_distill_loss = nsp_loss_fct(
                F.log_softmax(student_outputs[0].view(-1, nsp_label_num), dim=-1),
                F.softmax(teacher_outputs[0].view(-1, nsp_label_num), dim=-1))

        if cls_targets is not None and self.args.nsp_predict_distill:
            teacher_cls_loss = F.cross_entropy(teacher_outputs[0], cls_targets)
            student_cls_loss = F.cross_entropy(student_outputs[0], cls_targets)

        loss = attention_loss + hidden_loss + nsp_distill_loss
        logging_output = {
            'loss': loss.item() if reduce else loss.detach(),
        }
        if cls_targets is not None and self.args.nsp_predict_distill:
            teacher_probs = F.softmax(teacher_outputs[0], dim=-1)
            student_probs = F.softmax(student_outputs[0], dim=-1)
            teacher_max_p, teacher_indices = torch.max(teacher_probs, dim=1)
            student_max_p, student_indices = torch.max(student_probs, dim=1)
            logging_output.update({
                'teacher_cls_loss': teacher_cls_loss.item() if reduce else teacher_cls_loss.detach(),
                'student_cls_loss': student_cls_loss.item() if reduce else student_cls_loss.detach(),
                'cls_target': cls_targets.tolist(),
                'teacher_pred': teacher_indices.tolist(),
                'student_pred': student_indices.tolist(),
            })
            if teacher_probs.size(1) == 2:
                logging_output.update({
                    'teacher_prob': teacher_probs[:, 1].tolist(),
                    'student_prob': student_probs[:, 1].tolist(),
                })

        if self.args.attention_distill:
            logging_output.update({
                'attention_loss': attention_loss.item() if reduce else attention_loss.detach(),
            })
        if self.args.hidden_distill:
            logging_output.update({
                'hidden_loss': hidden_loss.item() if reduce else hidden_loss.detach(),
            })
        return loss, 1, logging_output

    def get_metrics(self, target, pred, prob, prefix):
        agg_output = dict()
        accuracy = accuracy_score(target, pred)
        agg_output['acc'] = accuracy

        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                target, pred, labels=list(range(1, self.args.num_classes)), average='micro')
        except ValueError:
            precision = recall = f1 = 0.0
        agg_output['precision'] = precision
        agg_output['recall'] = recall
        agg_output['f1'] = f1
        if prob is not None:
            try:
                auc = roc_auc_score(target, prob)
            except ValueError:  # when batch size is small and only one class presented in targets
                auc = 0.0
            agg_output['auc'] = auc
        return {prefix + k: v for k, v in agg_output.items()}

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        """Aggregate logging outputs from data parallel training."""
        keys = logging_outputs[0].keys()
        agg_output = {}

        if 'ntokens' in logging_outputs[0]:
            ntokens = sum(log['ntokens'] for log in logging_outputs)
            agg_output['ntokens'] = ntokens

        for key in keys:
            if 'loss' not in key:
                continue
            sum_value = sum(log.get(key, 0) for log in logging_outputs)
            agg_output[key] = sum_value * 1.0 / sample_size
        agg_output['sample_size'] = sample_size

        if 'cls_target' in logging_outputs[0]:
            cls_target = list(chain.from_iterable(log['cls_target'] for log in logging_outputs))
            teacher_pred = list(chain.from_iterable(log['teacher_pred'] for log in logging_outputs))
            student_pred = list(chain.from_iterable(log['student_pred'] for log in logging_outputs))

            teacher_prob = None
            student_prob = None
            if 'teacher_prob' in logging_outputs[0]:
                teacher_prob = list(chain.from_iterable(log['teacher_prob'] for log in logging_outputs))
                student_prob = list(chain.from_iterable(log['student_prob'] for log in logging_outputs))
            agg_output.update(self.get_metrics(cls_target, teacher_pred, teacher_prob, 'teacher_'))
            agg_output.update(self.get_metrics(cls_target, student_pred, student_prob, 'student_'))
        return agg_output
