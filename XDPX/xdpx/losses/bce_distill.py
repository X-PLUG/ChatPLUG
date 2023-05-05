import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from itertools import chain
from xdpx.options import Argument
from . import register
from .bce import BinaryCrossEntropy


@register('bce_distill')
class BCEDistillLoss(BinaryCrossEntropy):
    @staticmethod
    def register(options):
        BinaryCrossEntropy.register(options)
        options.register(
            Argument('distill_temperature', default=1.),
            Argument('gold_loss_lambda', default=0.0),
        )

    def forward(self, model, sample, reduce=True, logits=None):
        if not reduce:
            raise NotImplementedError
        teacher_logits = sample['logits']
        if logits is None:
            student_logits = model(**sample['net_input'])
        else:
            student_logits = logits
        T = self.args.distill_temperature

        logging_output = {}

        teacher_prob = torch.sigmoid(teacher_logits / T)
        distill_loss = F.binary_cross_entropy_with_logits(
            student_logits[:, 1:], teacher_prob[:, 1:], reduction='none').mean()

        if 'target' in sample and (self.args.gold_loss_lambda > 0 or not model.training):
            target = sample['target']
            target_oh = torch.zeros(target.size(0), self.args.num_classes).to(target)
            target_oh.scatter_(1, target.view(-1, 1), 1)
            gold_loss = F.binary_cross_entropy_with_logits(
                student_logits[:, 1:], target_oh[:, 1:].float(), reduction='none').mean()
            student_prob = torch.sigmoid(student_logits)
            indices = self.bce_inference(student_prob)

            a = self.args.gold_loss_lambda
            loss = (1 - a) * distill_loss + a * gold_loss

            logging_output.update({
                'loss': loss.item(),
                'gold_loss': gold_loss.item(),
                'distill_loss': distill_loss.item(),
                # for computing F1
                'target': target.tolist(),
                'pred': indices.tolist(),
            })
        else:
            loss = distill_loss
            logging_output['loss'] = loss.item()

        return loss, 1, logging_output

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        agg_output = {
            'sample_size': sample_size,
        }
        if 'ntokens' in logging_outputs[0]:
            ntokens = sum(log['ntokens'] for log in logging_outputs)
            agg_output['ntokens'] = ntokens
        if max_count:
            sample_size = max(min(max_count, sample_size), 1)

        agg_output['loss'] = sum(log['loss'] for log in logging_outputs) / sample_size

        if 'target' in logging_outputs[0]:
            target = list(chain.from_iterable(log['target'] for log in logging_outputs))
            pred = list(chain.from_iterable(log['pred'] for log in logging_outputs))

            agg_output['gold_loss'] = sum(log['gold_loss'] for log in logging_outputs) / sample_size
            agg_output['distill_loss'] = sum(log['distill_loss'] for log in logging_outputs) / sample_size
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
        return agg_output
