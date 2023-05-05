import os
import json
from typing import List
import torch
import torch.nn as nn
from xdpx.modules.thirdparty.transformers.modeling_bert import (
    BertAttention, BertIntermediate, BertOutput
)
from . import register
from .bert import Bert, BertForClassification, BertForLanguageModeling, BertForPretraining
from ..modules import HashingMemory, LocallyOptimizedHashingMemory
from xdpx.options import Argument, Options
from xdpx.utils import io, parse_model_path, should_save_meta


class AbstractPKMBert(Bert):
    @staticmethod
    def register(options):
        options.register(
            # where to use memory
            Argument('mem_positions', required=True, type=List[int], validate=[
                lambda value: len(value) == len(set(value)),
                lambda value: len(value) > 0,
            ]),
            Argument('pretrained_memories', validate=lambda value: not value or io.exists(value),
                     post_process=parse_model_path, type=str),
            Argument('share_memory', default=False),
        )
        options.add_global_constraint(lambda args: not hasattr(args, 'optimizer') or args.optimizer == 'pair')
        options.add_global_constraint(lambda args: not args.mem_sparse or not hasattr(args, 'optimizer')
                                      or args.second_optimizer == 'sparse_adam')
        options.add_global_constraint(lambda args: max(args.mem_positions) < args.num_hidden_layers)

    def build_bert_backend(self):
        super().build_bert_backend()
        config = self.get_backend_config()
        self.bert.encoder.layer = nn.ModuleList([
            BertLayer(i in self.args.mem_positions, config, self.build_memory)
            for i in range(self.args.num_hidden_layers)
        ])

    def save_bert_config(self, config=None):
        if should_save_meta(self.args):
            config = config or self.get_bert_config()
            options = Options()
            AbstractPKMBert.register(options)
            self.__class__.register(options)
            mem_config = {name: getattr(self.args, name) for name in options.keys()}
            mem_config.pop('pretrained_memories')
            config.update(mem_config)
            with io.open(os.path.join(self.args.save_dir, f'bert_config.json'), 'w') as f:
                json.dump(config, f, indent=2)

    def get_param_groups(self, module, **kwargs):
        ordinary = []
        memory = []
        no_weight_decay = []
        for name, param in module.named_parameters():
            if param.requires_grad:
                if 'LayerNorm' in name or 'bias' in name:
                    no_weight_decay.append(param)
                elif name.endswith(HashingMemory.MEM_VALUES_PARAMS):
                    memory.append(param)
                else:
                    ordinary.append(param)
        return [
            {**kwargs, 'params': ordinary},
            {**kwargs, 'params': memory, 'flag': None},
            {**kwargs, 'params': no_weight_decay, 'weight_decay': 0.0},
        ]

    def trainable_parameters(self):
        param_groups = super().trainable_parameters()
        major_param_groups = []
        second_param_groups = []
        for param_group in param_groups:
            if 'flag' not in param_group:
                major_param_groups.append(param_group)
                continue
            param_group.pop('flag')
            second_param_groups.append(param_group)
        return major_param_groups, second_param_groups

    def _load_and_parse_state_dict(self, path):
        state_dict = super()._load_and_parse_state_dict(path)
        if self.args.pretrained_memories and hasattr(self.args, '__cmd__') and self.args.__cmd__ == 'train' \
                and hasattr(self.args, 'resume') and self.args.resume is False:
            print(f'| Loading pretrained memories for {self.__class__.__name__} from {self.args.pretrained_memories}')
            with io.open(self.args.pretrained_memories, 'rb') as f:
                checkpoint = torch.load(f, map_location='cpu')
                state_dict.update(checkpoint)
            for i in self.args.mem_positions:
                obsolete_keys = [
                    f'bert.encoder.layer.{i}.intermediate.dense.weight',
                    f'bert.encoder.layer.{i}.intermediate.dense.bias',
                    f'bert.encoder.layer.{i}.output.dense.weight',
                    f'bert.encoder.layer.{i}.output.dense.bias',
                ]
                for key in obsolete_keys:
                    if key in state_dict:
                        del state_dict[key]
        return state_dict

    @property
    def name_map(self):
        name_map = super().name_map
        for i, layer_id in enumerate(self.args.mem_positions):
            name_map.update({
                f'bert.encoder.layer.{layer_id}.output.LayerNorm.weight': f'bert.encoder.layer.{layer_id}.layer_norm.weight',
                f'bert.encoder.layer.{layer_id}.output.LayerNorm.bias': f'bert.encoder.layer.{layer_id}.layer_norm.bias',
            })
        return name_map

    def customized_name_mapping(self, state_dict):
        import re
        mem_positions = {str(i): layer_id for i, layer_id in enumerate(self.args.mem_positions)}
        prefix = re.compile(r'memories\.(\d+)')
        for key in list(state_dict.keys()):
            m = prefix.search(key)
            if not m:
                continue
            new_key = f'bert.encoder.layer.{mem_positions[m.group(1)]}.memory' + key[len(m.group()):]
            state_dict[new_key] = state_dict.pop(key)
        return state_dict


class PKMBert(AbstractPKMBert):
    @staticmethod
    def register(options):
        options.register(
            Argument('mem_k_dim', default=256, doc='Memory keys dimension', validate=lambda value: value % 2 == 0),
            Argument('mem_heads', default=4, doc='Number of memory heads'),
            Argument('mem_knn', default=32, doc='Number of memory slots to read / update - k-NN to the query'),
            Argument('mem_keys', default=512, doc='Number of product keys. Total memory size: n ** 2'),
            Argument('query_norm', validate=lambda value: not value or value in ('batchnorm', 'layernorm')),
            Argument('distance_fn', default='dot', validate=lambda value: value in ('dot', 'euc', 'mah', 'mah_fast')),
            Argument('kernel_alpha', default=0.5, doc='default is standard Gaussian kernel'),
            Argument('mem_sparse', default=False, doc='Perform sparse updates for the values'),
            Argument('input_dropout', default=0.),
            Argument('query_dropout', default=0.),
            domain='Bert/PKM'
        )

    def build_memory(self):
        return HashingMemory(
            self.args.hidden_size, self.args.hidden_size, k_dim=self.args.mem_k_dim, n_keys=self.args.mem_keys,
            heads=self.args.mem_heads, knn=self.args.mem_knn, input_dropout=self.args.input_dropout,
            query_dropout=self.args.query_dropout, value_dropout=self.args.hidden_dropout_prob,
            sparse=self.args.mem_sparse, query_norm=self.args.query_norm, distance_fn=self.args.distance_fn,
            share_values=self.args.share_memory,
        )


class LOPKMBert(AbstractPKMBert):
    @staticmethod
    def register(options):
        options.register(
            Argument('mem_k_dim', required=True, type=int,
                     doc='Memory keys dimension', validate=lambda value: value % 2 == 0),
            Argument('mem_heads', required=True, type=int, doc='Number of memory heads'),
            Argument('mem_knn', required=True, type=int,
                     doc='Number of memory slots to read / update - k-NN to the query'),
            Argument('mem_keys', required=True, type=int, doc='Number of product keys. Total memory size: n ** 2'),
            Argument('mem_sparse', default=False, doc='Perform sparse updates for the values'),
            Argument('input_dropout', default=0.),
            domain='Bert/PKM'
        )

    def build_memory(self):
        return LocallyOptimizedHashingMemory(
            self.args.hidden_size, self.args.hidden_size, k_dim=self.args.mem_k_dim, n_keys=self.args.mem_keys,
            heads=self.args.mem_heads, knn=self.args.mem_knn, input_dropout=self.args.input_dropout,
            value_dropout=self.args.hidden_dropout_prob, sparse=self.args.mem_sparse,
            share_values=self.args.share_memory,
        )


@register('bert_classification_pkm')
class BertClassificationPKM(BertForClassification, PKMBert):...

@register('bert_lm_pkm')
class BertForLanguageModelingPKM(BertForLanguageModeling, PKMBert):...

@register('bert_pretrain_pkm')
class BertForPretrainingPKM(BertForPretraining, PKMBert):...

@register('bert_classification_lopkm')
class BertClassificationLOPKM(BertForClassification, LOPKMBert):...

@register('bert_lm_lopkm')
class BertForLanguageModelingLOPKM(BertForLanguageModeling, LOPKMBert):...

@register('bert_pretrain_lopkm')
class BertForPretrainingLOPKM(BertForPretraining, LOPKMBert):...


class BertLayer(nn.Module):
    def __init__(self, has_memory, config, memory_factory):
        super().__init__()
        self.attention = BertAttention(config)
        if config.is_decoder:
            raise NotImplementedError
        if has_memory:
            self.memory = memory_factory()
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.intermediate = BertIntermediate(config)
            self.output = BertOutput(config)
            self.memory = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.memory:
            memory_mask = attention_mask.view(attention_mask.size(0), attention_mask.size(-1), 1) > -1e4
            layer_output = attention_output + self.memory(attention_output, memory_mask)
            layer_output = self.layer_norm(layer_output)
        else:
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs
