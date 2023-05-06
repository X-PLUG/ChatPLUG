import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from itertools import chain
from typing import List
from xdpx.options import Argument
from xdpx.tasks.distill import DistillTask
from xdpx.tasks import register as register_task
from . import register, Loss


@register('distill')
class DistillLoss(Loss):
    @staticmethod
    def register(options):
        options.register(
            Argument('distill_temperature', default=1.),
        )

    def forward(self, model, sample, reduce=True):
        assert reduce is True
        with torch.no_grad():
            teacher_logits = model.teacher(**sample['net_input'])
        student_logits = model.student(**sample['net_input'])
        T = self.args.distill_temperature

        logging_output = {}
        loss = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                        F.softmax(teacher_logits / T, dim=1), reduction='batchmean') * (T * T)
        if 'target' in sample:
            target = sample['target']
            alpha = self.args.alpha
            loss_ce = F.cross_entropy(student_logits, target)
            logging_output['kd_loss'] = loss.item()
            logging_output['cls_loss'] = loss_ce.item()
            loss = loss * alpha + (1. - alpha) * loss_ce
        logging_output['loss'] = loss.item()

        return loss, 1, logging_output

    def inference(self, model, sample):
        raise NotImplementedError

    def distill(self, model, sample):
        raise NotImplementedError

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss / max(sample_size, 1),
            'sample_size': sample_size,
        }
        if 'ntokens' in logging_outputs[0]:
            ntokens = sum(log['ntokens'] for log in logging_outputs)
            agg_output['ntokens'] = ntokens
        if 'kd_loss' in logging_outputs:
            kd_loss = sum(log.get('kd_loss', 0) for log in logging_outputs)
            cls_loss = sum(log.get('cls_loss', 0) for log in logging_outputs)
            agg_output['kd_loss'] = kd_loss / max(sample_size, 1)
            agg_output['cls_loss'] = cls_loss / max(sample_size, 1)
        return agg_output

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `aggregate_logging_outputs`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register('local_distill')
class LocalDistillLoss(DistillLoss):
    @staticmethod
    def register(options):
        pass

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert reduce is True
        logits = model(**sample['net_input'])
        targets = sample['target']
        loss = ((logits - targets) ** 2).mean()
        logging_output = {
            'loss': loss.item(),
        }
        return loss, 1, logging_output


@register_task('local_distill_exp')
class LocalDistillExportTask(DistillTask):
    def inference_step(self, sample, model, loss):
        model.eval()
        with torch.no_grad():
            logits = model(**sample['net_input'])
            targets = sample['target']
            loss = ((logits - targets) ** 2).mean(1)
        return logits.tolist(), targets.tolist(), loss.tolist()


@register('static_distill')
class StaticDistillLoss(DistillLoss):

    @staticmethod
    def register(options):
        DistillLoss.register(options)
        options.register(
            Argument('negative_classes_in_f1', default=[0], type=List[int], doc='negative classes for micro F1 score'),
        )
        options.register(Argument('gold_loss_lambda', default=0.0))
        options.add_global_constraint(
            lambda args: max(args.negative_classes_in_f1) < args.num_classes,
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

        distill_loss = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                                F.softmax(teacher_logits / T, dim=1), reduction='batchmean') * (T * T)
        if 'target' in sample and (self.args.gold_loss_lambda > 0 or not model.training):
            target = sample['target']

            indices = torch.argmax(student_logits, dim=1)
            lprobs = F.log_softmax(student_logits, dim=-1)
            target = sample['target']
            gold_loss = F.nll_loss(lprobs, target, reduction='mean')

            a = self.args.gold_loss_lambda
            loss = (1 - a) * distill_loss + a * gold_loss

            logging_output.update({
                'loss': loss.item(),
                'gold_loss': gold_loss.item(),
                'kl_loss': distill_loss.item(),
                # for computing F1
                'target': target.tolist(),
                'pred': indices.tolist(),
            })
            if student_logits.size(1) == 2:
                logging_output.update({
                    'prob': torch.exp(lprobs)[:, 1].tolist(),
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

        loss = list(log['loss'] for log in logging_outputs)
        agg_output['loss'] = sum(loss) / sample_size if sample_size > 0 else 0.

        if 'target' in logging_outputs[0]:
            target = list(chain.from_iterable(log['target'] for log in logging_outputs))
            pred = list(chain.from_iterable(log['pred'] for log in logging_outputs))

            agg_output['gold_loss'] = sum(log['gold_loss'] for log in logging_outputs) / sample_size
            agg_output['kl_loss'] = sum(log['kl_loss'] for log in logging_outputs) / sample_size
            accuracy = accuracy_score(target, pred)
            agg_output['acc'] = accuracy
            if self.args.negative_classes_in_f1:
                labels = [i for i in range(self.args.num_classes) if i not in self.args.negative_classes_in_f1]
                try:
                    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, labels=labels, average='micro')
                except ValueError:
                    precision = recall = f1 = 0.0
                agg_output['precision'] = precision
                agg_output['recall'] = recall
                agg_output['f1'] = f1
            if 'prob' in logging_outputs[0]:
                prob = list(chain.from_iterable(log['prob'] for log in logging_outputs))
                try:
                    auc = roc_auc_score(target, prob)
                except ValueError:  # when batch size is small and only one class presented in targets
                    auc = 0.0
                agg_output['auc'] = auc

        return agg_output

    def inference(self, model, sample):
        logits = model(**sample['net_input'])
        probs = F.softmax(logits, dim=-1)
        _, indices = torch.max(probs, dim=1)
        return indices.tolist(), probs.tolist()

    @staticmethod
    def logging_outputs_can_be_summed():
        return False
