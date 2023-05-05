import torch
import torch.nn.functional as F
from xdpx.options import Argument
from . import register, Loss


@register('minilm')
class MiniLMLoss(Loss):
    """
    Implement MINILM https://arxiv.org/pdf/2002.10957.pdf
    Please set the model to minilm
    """
    @staticmethod
    def register(options):
        Loss.register(options)
        from xdpx.models import models, Model
        with options.with_prefix('teacher_'):
            options.register(
                Argument('model', required=True, validate=lambda value: value in models,
                         register=lambda value: Model.build_model_class(value).register),
                domain='teacher_model',
            )

    def __init__(self, args):
        super().__init__(args)
        from xdpx.models import models
        self.teacher_model = models[args.teacher_model].build(args.strip_prefix('teacher_').change(extra_config=dict(
            output_attentions=True,
            output_value_attentions=True,
        )))
        self.padding_idx = args.pad_index

    def attention_kl_loss(self, raw_attentions, teacher_raw_attentions, attention_mask):
        last_s_attention = raw_attentions[-1]  # [B,H,T,T]
        last_t_attention = teacher_raw_attentions[-1]  # [B,H,T,T]

        kl_loss = F.kl_div(
            F.log_softmax(last_s_attention, dim=-1),
            F.softmax(last_t_attention, dim=-1),
            reduction='none'
        ).sum(-1)
        kl_loss = (kl_loss * attention_mask.unsqueeze(1)).sum(2) / attention_mask.sum(1, keepdim=True)
        return kl_loss.mean()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert reduce
        net_input = sample['net_input']
        input_ids = net_input['input_ids']
        mask = input_ids.ne(self.args.pad_index).long()
        net_input['attention_mask'] = mask
        with torch.no_grad():
            teacher_outputs = self.teacher_model.bert_forward(**net_input)
        student_outputs = model.bert_forward(**net_input)
        assert len(teacher_outputs) == len(student_outputs) == 4
        mask = mask.float()
        value_loss = self.attention_kl_loss(student_outputs[-1], teacher_outputs[-1], mask)
        attention_loss = self.attention_kl_loss(student_outputs[-2], teacher_outputs[-2], mask)

        loss = (attention_loss + value_loss) / 2
        logging_output = {
            'loss': loss.item(),
            'value_loss': value_loss.item(),
            'attention_loss': attention_loss.item(),
        }
        return loss, 1, logging_output

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        """Aggregate logging outputs from data parallel training."""
        keys = logging_outputs[0].keys()
        agg_output = {}
        for key in keys:
            sum_value = sum(log.get(key, 0) for log in logging_outputs)
            agg_output[key] = sum_value * 1.0 / max(sample_size, 1)
        agg_output['sample_size'] = sample_size
        return agg_output
