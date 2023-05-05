import torch
import torch.nn as nn
from xdpx.modules.thirdparty.transformers.modeling_bert import BertSelfAttention
from . import register
from .bert import Bert, BertForClassification, BertForLanguageModeling, BertForPretraining
from ..modules.activations import activations


class ReZeroBert(Bert):
    @staticmethod
    def register(options):
        options.register(
            domain='Bert/ReZero'
        )
    
    def build_bert_backend(self):
        super().build_bert_backend()
        self.bert.encoder.layer = nn.ModuleList([
            BertLayer(self.get_backend_config()) for _ in range(self.args.num_hidden_layers)
        ])
    
    @property
    def name_map(self):
        name_map = super().name_map.copy()
        for i in range(self.args.num_hidden_layers):
            name_map.update({
                f'bert.encoder.layer.{i}.intermediate.dense.weight': f'bert.encoder.layer.{i}.linear1.weight',
                f'bert.encoder.layer.{i}.intermediate.dense.bias': f'bert.encoder.layer.{i}.linear1.bias',
                f'bert.encoder.layer.{i}.output.dense.weight': f'bert.encoder.layer.{i}.linear2.weight',
                f'bert.encoder.layer.{i}.output.dense.bias': f'bert.encoder.layer.{i}.linear2.bias',
            })
        return name_map
    
    def bert_forward(self, input_ids, **kwargs):
        outputs = super().bert_forward(input_ids, **kwargs)
        from xdpx.logger import log
        for i in range(self.args.num_hidden_layers):
            log.add_summary(f'res_{i}', self.bert.encoder.layer[i].resweight)
        res_avg = sum(self.bert.encoder.layer[i].resweight.abs() for i in range(self.args.num_hidden_layers))
        res_avg /= self.args.num_hidden_layers
        log.add_summary(f'res_avg', res_avg)
        return outputs


@register('bert_classification_rezero')
class BertClassificationRezero(BertForClassification, ReZeroBert):...

@register('bert_lm_rezero')
class BertForLanguageModelingRezero(BertForLanguageModeling, ReZeroBert):...

@register('bert_pretrain_rezero')
class BertForPretrainingRezero(BertForPretraining, ReZeroBert):...


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        if config.is_decoder:
            raise NotImplementedError
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.resweight = nn.Parameter(torch.Tensor([0]))
        self.activation = activations[config.hidden_act]

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        src = hidden_states
        src2 = self.attention(hidden_states, attention_mask, head_mask)
        outputs = src2[1:]  # add self attentions if we output attention weights
        src2 = src2[0] # no attention weights
        src2 = src2 * self.resweight
        src = src + self.dropout1(src2)

        # Pointwise FF Layer
        src2 = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = src2 * self.resweight
        src = src + self.dropout2(src2)
        return (src,) + outputs


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = nn.ModuleDict({'dense': nn.Linear(config.hidden_size, config.hidden_size)})

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        attention_output = self.output['dense'](self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs