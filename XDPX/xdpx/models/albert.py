from typing import List
from xdpx.options import Argument
from xdpx.modules.thirdparty.transformers.modeling_albert import AlbertModel, AlbertConfig
from . import register
from .bert import Bert, BertForClassification, BertForLanguageModeling, BertForPretraining


class AlBert(Bert):
    @staticmethod
    def register(options):
        AlBert.register_bert_config(options)

    @staticmethod
    def register_bert_config(options):
        try:
            Bert.register_bert_config(options)
        except ValueError as e:
            if 'duplicated argument' not in str(e):
                raise
        options.register(
            Argument('embedding_size', default=128),
            Argument('num_hidden_groups', default=1),
            Argument('net_structure_type', default=0),
            Argument('layers_to_keep', default=[], type=List[int]),
            Argument('gap_size', default=0),
            Argument('num_memory_blocks', default=0),
            Argument('inner_group_num', default=1),
            Argument('down_scale_factor', default=1),
            domain='bert_config'
        )

    @property
    def name_map(self):
        return {}

    def build_bert_backend(self):
        config = self.get_bert_config()
        self.save_bert_config(config)
        self.bert = AlbertModel(AlbertConfig(self.args.vocab_size, **config))

    def load_from_tf(self, tf_vars):
        from xdpx.utils.tf_utils import load_tf_weights

        load_tf_weights(
            model=self,
            tf_vars=tf_vars,
            name_map=[
                # If saved from the TF HUB module
                ('module/', ''),

                # Renaming and simplifying
                ('ffn_1', 'ffn'),
                ('attention_1', 'attention'),
                ('transform/', ''),
                ('LayerNorm_1', 'full_layer_layer_norm'),
                ('LayerNorm', 'attention/LayerNorm'),
                ('transformer/', ''),

                # The feed forward layer had an 'intermediate' step which has been abstracted away
                ('intermediate/dense/', ''),
                ('ffn/intermediate/output/dense/', 'ffn_output/'),

                # ALBERT attention was split between self and output which have been abstracted away
                ('/output/', '/'),
                ('/self/', '/'),

                # The pooler is a linear layer
                ('pooler/dense', 'pooler'),

                # The classifier was simplified to predictions from cls/predictions
                ('cls/predictions', 'predictions'),
                ('predictions/attention', 'predictions'),

                # Naming was changed to be more explicit
                ('embeddings/attention', 'embeddings'),
                ('inner_group_', 'albert_layers/'),
                ('group_', 'albert_layer_groups/'),

                # name_map inherited from Model
                *[(key, val) for key, val in self.name_map.items()],
            ],
            name_fn=[
                lambda name: name + '/weight' if name.endswith('_embeddings') else name,
                lambda name: "classifier/" + name if len(name.split("/")) == 1
                                                     and ("output_bias" in name or "output_weights" in name) else name,
            ],
            ignore_vars=[
                'adam_v', 'adam_m', 'global_step',
                # No ALBERT model currently handles the next sentence prediction task
                'seq_relationship',
            ],
            transpose_vars=[
                lambda name: name[-1] == 'kernel' and name[-2] != 'cls',
            ],
            retriever=[
                ('kernel', 'weight'),
                ('gamma', 'weight'),
                ('beta', 'bias'),
                ('output_bias', 'bias'),
                ('output_weights', 'weight'),
                ('squad', 'classifier'),
            ],
            strict=self.args.strict_size
        )


@register('albert_classification')
class AlbertClassification(BertForClassification, AlBert):
    @property
    def name_map(self):
        return {}

@register('albert_lm')
class AlbertForLanguageModeling(BertForLanguageModeling, AlBert):...

@register('albert_pretrain')
class AlbertForPretraining(BertForPretraining, AlBert):...
