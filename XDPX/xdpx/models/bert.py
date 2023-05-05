import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
from functools import lru_cache
from . import register, Model
from xdpx.processors import processors
from xdpx.processors.bert import BertProcessor
from xdpx.modules.activations import activations
from xdpx.modules.layer_norm import LayerNorm
from xdpx.utils import io, should_save_meta
from xdpx.options import Argument, Options


class Bert(Model):
    @staticmethod
    def register(options):
        Bert.register_bert_config(options)
        options.register(
            Argument('layer_lr_decay', default=1., doc='layer-wise learning rate decay', children={
                lambda value: value < 1.: [
                    Argument('keep_emb_lr', default=True, doc='whether to keep emb lr the same as initial'),
                ]
            }),
            Argument('top_lr_ratio', default=1.0, doc='initial lr for the top (non-bert) layers'),
            Argument('fix_lm', default=False, doc='if true, only top layer parameters are updated'),
            Argument('gradient_checkpointing', default=False, doc='if true, model.gradient_checkpointing_enable()'),
            Argument('bert_wd', type=float, doc='if not None, set separate weight decay for bert'),
            Argument('pruned_heads', type=Dict[int, List[int]],
                     post_process=lambda x: {int(key): val for key, val in x.items()} if x is not None else x),
            Argument('extra_config', default={}, type=Dict[str, Any],
                     doc='extra bert configs related to HuggingFace backend'),
            domain='Bert',
        )

        options.add_global_constraint(lambda args: issubclass(processors[args.processor], BertProcessor))
        # options.add_global_constraint(lambda args: args.pad_index == 0)
        options.mark_required('pretrained_model')

    def __init__(self, args):
        super().__init__(args)
        self.bert = None
        self.build_bert_backend()

    def load_from_tf(self, tf_vars):
        from xdpx.utils.tf_utils import load_tf_weights

        load_tf_weights(
            model=self,
            tf_vars=tf_vars,
            name_map=[(key, val) for key, val in self.name_map.items()],
            name_fn=[
                lambda name: name + '/weight' if name.endswith('_embeddings') else name,
            ],
            ignore_vars=[
                'adam_v', 'adam_m', 'global_step'
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

    def get_bert_config(self):
        options = Options()
        self.register_bert_config(options)
        config = {name: getattr(self.args, name) for name in options.keys()}
        config['vocab_size'] = self.args.vocab_size
        config['architectures'] = ['BertModel']
        return config

    def save_bert_config(self, config=None):
        if should_save_meta(self.args):
            config = config or self.get_bert_config()
            with io.open(os.path.join(self.args.save_dir, f'bert_config.json'), 'w') as f:
                json.dump(config, f, indent=2)

    """
    ===========================================================================
    Methods below are most likely to be overwritten when Bert is inherited
    ===========================================================================
    """

    @staticmethod
    def register_bert_config(options):
        options.register(  # default value is bert base
            Argument('attention_probs_dropout_prob', default=0.1),
            Argument('directionality', default="bidi"),
            Argument('hidden_act', default="gelu"),
            Argument('hidden_dropout_prob', default=0.1),
            Argument('hidden_size', default=768),
            Argument('initializer_range', default=0.02),
            Argument('intermediate_size', default=3072),
            Argument('max_position_embeddings', default=512),
            Argument('num_attention_heads', default=12),
            Argument('num_hidden_layers', default=12),
            Argument('type_vocab_size', default=2),
            domain='bert_config'
        )

    def build_bert_backend(self):
        if hasattr(self.args, 'auto_model') and self.args.auto_model is not None:
            from transformers import AutoModel, AutoConfig
            config = AutoConfig.from_pretrained(self.args.auto_model)
            if hasattr(self.args, 'gradient_checkpointing') and self.args.gradient_checkpointing:
                config.gradient_checkpointing = True
            self.save_bert_config(config.__dict__)
            self.bert = AutoModel.from_config(config)

        else:
            # from xdpx.modules.thirdparty.transformers.modeling_bert import BertModel
            from xdpx.modules.thirdparty.transformers421.modeling_bert import BertModel
            config = self.get_bert_config()
            self.save_bert_config(config)
            self.bert = BertModel(self.get_backend_config())
            if hasattr(self.args, 'pruned_heads') and self.args.pruned_heads is not None:
                self.bert.prune_heads(self.args.pruned_heads)

    @lru_cache(maxsize=2)
    def get_backend_config(self, config=None):
        if not config:
            config = self.get_bert_config()
        # from xdpx.modules.thirdparty.transformers.modeling_bert import BertConfig
        # return BertConfig(self.args.vocab_size, **config, **self.extra_config())
        from xdpx.modules.thirdparty.transformers421.modeling_bert import BertConfig
        return BertConfig(**config, **self.extra_config())

    def extra_config(self):
        # huggingface-related config, should no appear in bert_config file
        config = dict(
            layer_norm_eps=1e-12,
            output_attentions=False,
            output_hidden_states=False,
        )
        if hasattr(self.args, 'fp16') and self.args.fp16:
            config['layer_norm_eps'] = 1e-5
        config.update(self.args.extra_config)
        return config

    def get_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def bert_forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        """connects to HuggingFace bert encoder"""
        # returns sequence_output, pooled_output, (hidden_states), (attentions)
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
            **kwargs)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, *args, **kwargs):
        raise NotImplementedError

    @property
    def dummy_inputs(self):
        input_ids = torch.randint(1, self.args.vocab_size, (8, 16))
        return (
            input_ids,
            input_ids.ne(0),
            torch.zeros_like(input_ids),
        )

    def get_param_groups(self, module, **kwargs):
        oridinary = []
        no_weight_decay = []
        for name, param in module.named_parameters():
            if param.requires_grad:
                if 'LayerNorm' in name or 'bias' in name:
                    no_weight_decay.append(param)
                else:
                    oridinary.append(param)
        return [
            {**kwargs, 'params': oridinary},
            {**kwargs, 'params': no_weight_decay, 'weight_decay': 0.0},
        ]

    def trainable_parameters(self):
        param_groups = []
        if self.args.fix_lm:
            for name, module in self.named_children():
                if name == 'bert':
                    param_groups.extend(
                        self.get_param_groups(module.pooler, lr=self.args.learning_rate * self.args.top_lr_ratio))
                else:
                    param_groups.extend(
                        self.get_param_groups(module, lr=self.args.learning_rate * self.args.top_lr_ratio))
            return param_groups

        if self.args.layer_lr_decay == 1 and self.args.top_lr_ratio == 1 and self.args.bert_wd is None:
            return self.get_param_groups(self)

        # add params in seperate groups (even with the same hyper-params) will result in minor numerical differences
        # so the exact match of training trace is not expected
        for name, module in self.named_children():
            if name == 'bert':
                param_groups.extend(
                    self.get_param_groups(module.pooler, lr=self.args.learning_rate * self.args.top_lr_ratio))
            else:
                param_groups.extend(self.get_param_groups(module, lr=self.args.learning_rate * self.args.top_lr_ratio))

        bert_weight_decay = self.args.weight_decay if self.args.bert_wd is None else self.args.bert_wd
        for i, layer in enumerate(self.bert.encoder.layer[::-1]):
            layer_lr = self.args.learning_rate * (self.args.layer_lr_decay ** i)
            param_groups.extend(self.get_param_groups(layer, lr=layer_lr, weight_decay=bert_weight_decay))

        emb_lr = self.args.learning_rate if getattr(self.args, 'keep_emb_lr', True) else layer_lr
        for name, module in self.bert.embeddings.named_children():
            if name == 'word_embeddings' and hasattr(self, 'cls'):
                # the embedding is tied and does not need to add again
                continue
            param_groups.extend(self.get_param_groups(module, lr=emb_lr, weight_decay=bert_weight_decay))

        ground_truth = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_trainable = sum(p.numel() for param_group in param_groups for p in param_group['params'])
        assert ground_truth == num_trainable, f'{num_trainable} of {ground_truth} trainable parameters are registered in the optimizer.'
        return param_groups

    def customized_name_mapping(self, state_dict):
        result = {}
        for k, v in state_dict.items():
            # bert-base-uncased in huggingface has different names in LayerNorm with BertModel
            if 'gamma' in k:
                k = k.replace('gamma', 'weight')
            if 'beta' in k:
                k = k.replace('beta', 'bias')
            if 'roberta' in k:
                k = k.replace('roberta','bert')
            result[k] = v
        return result

    # tf compatibility
    def load_into_tf(self, sess, strict=True):
        from xdpx.utils.tf_utils import load_into_tf
        predict_signature = self.build_tf_graph(sess)

        state_dict = self.state_dict()
        # this one is the same as embedding, so we skip it
        state_dict.pop('cls.predictions.decoder.weight', None)

        load_into_tf(sess, state_dict, name_map=(
            # classification
            ('classifier.weight', 'output_weight'),
            ('classifier.bias', 'output_bias'),
            ('predictions.bias', 'predictions/output_bias'),
            ('seq_relationship.weight', 'seq_relationship/output_weights'),
            ('seq_relationship.bias', 'seq_relationship/output_bias'),
            # BERT base model
            ('layer.', 'layer_'),
            ('word_embeddings.weight', 'word_embeddings'),
            ('position_embeddings.weight', 'position_embeddings'),
            ('token_type_embeddings.weight', 'token_type_embeddings'),
            ('.', '/'),
            ('LayerNorm/weight', 'LayerNorm/gamma'),
            ('LayerNorm/bias', 'LayerNorm/beta'),
            ('/weight', '/kernel'),
        ), tensors_to_transpose=(
            "dense.weight",
            "attention.self.query",
            "attention.self.key",
            "attention.self.value"
        ), strict_size=strict)

        return predict_signature

    def build_tf_graph(self, sess):
        import tensorflow as tf
        from xdpx.modules.thirdparty.bert import modeling, run_pretraining

        bert_config = self.args
        state_dict = self.state_dict()
        graph = sess.graph
        input_ids = tf.placeholder(tf.int64, shape=[None, None], name='input_ids')
        input_mask = tf.placeholder_with_default(tf.ones_like(input_ids), [None, None], name='input_mask')
        segment_ids = tf.placeholder_with_default(tf.zeros_like(input_ids), [None, None], name='segment_ids')
        bert_model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
        output_layer = bert_model.get_pooled_output()
        hidden_size = output_layer.shape[-1].value
        outputs = {}
        if 'cls.predictions.bias' in state_dict:
            masked_lm_positions = tf.placeholder(tf.int32, shape=[None, None], name='masked_lm_positions')
            masked_lm_ids = tf.placeholder(tf.int32, shape=[None, None], name='masked_lm_ids')
            masked_lm_weights = tf.placeholder(tf.float32, shape=[None, None], name='masked_lm_weights')
            _, _, log_probs = run_pretraining.get_masked_lm_output(
                bert_config, bert_model.get_sequence_output(), bert_model.get_embedding_table(),
                masked_lm_positions, masked_lm_ids, masked_lm_weights)
            tf.identity(log_probs, name='lm_log_prob')
            outputs['lm_log_prob'] = graph.get_tensor_by_name('lm_log_prob:0')

        if 'cls.seq_relationship.bias' in state_dict:
            next_sentence_labels = tf.placeholder(tf.int64, shape=[None, None], name='next_sentence_labels')
            _, _, log_probs = run_pretraining.get_next_sentence_output(
                bert_config, bert_model.get_pooled_output(), next_sentence_labels)
            tf.identity(log_probs, name='cls_log_prob')
            outputs['cls_log_prob'] = graph.get_tensor_by_name('cls_log_prob:0')

        elif 'classifier.bias' in state_dict:
            num_classes = state_dict['classifier.bias'].numel()
            output_weights = tf.get_variable('output_weight', [num_classes, hidden_size])
            output_bias = tf.get_variable('output_bias', [num_classes])
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias, name='logits')
            tf.nn.softmax(logits, -1, name='probs')
            tf.argmax(logits, 1, name='predictions')
            outputs.update({
                'predictions': graph.get_tensor_by_name('predictions:0'),
                'prob': graph.get_tensor_by_name('probs:0'),
            })

        sess.run(tf.global_variables_initializer())

        return dict(
            inputs=dict(
                input_ids=graph.get_tensor_by_name('input_ids:0'),
                input_mask=graph.get_tensor_by_name('input_mask:0'),
                segment_ids=graph.get_tensor_by_name('segment_ids:0'),
            ), outputs=outputs
        )

    def dummy_tf_inputs(self, inputs=None):
        if inputs is None:
            inputs = self.dummy_inputs
        return {
            'input_ids': inputs[0].tolist(),
        }


@register('bert_classification')
class BertForClassification(Bert):
    @staticmethod
    def register(options):
        options.register(
            Argument('output_dropout_prob', default=0.0),
            Argument('load_cls_weights', default=False),
        )

    def __init__(self, args):
        super().__init__(args)
        self.classifier = ClassificationHead(
            in_features=self.args.hidden_size,
            out_features=args.num_classes,
        )

    def forward(self, *args, **kwargs):
        # pooled output as default
        features = self.bert_forward(*args, **kwargs)[1]
        features = F.dropout(features, self.args.output_dropout_prob, self.training)
        return self.classifier(features)

    @property
    def name_map(self):
        name_map = super().name_map.copy()
        if self.args.load_cls_weights:
            name_map.update({
                'cls.seq_relationship.weight': 'classifier.weight',
                'cls.seq_relationship.bias': 'classifier.bias',
                'heads.cls.weight': 'classifier.weight',
                'heads.cls.bias': 'classifier.bias',
            })
        name_map.update({
            'heads.classification.weight': 'classifier.weight',
            'heads.classification.bias': 'classifier.bias',
        })
        return name_map


@register('bert_classification_concat')
class BertForClassificationConcat(BertForClassification):
    @staticmethod
    def register(options):
        options.register(
            Argument('highway_layers', type=List[int], default=[-1, -4, -7, -10]),
        )

    def extra_config(self):
        config = super().extra_config()
        config['output_hidden_states'] = True
        return config

    def __init__(self, args):
        super().__init__(args)
        self.pooler = nn.Linear(self.args.hidden_size * len(args.highway_layers), self.args.hidden_size)
        nn.init.normal_(self.pooler.weight, std=self.args.initializer_range)
        nn.init.zeros_(self.pooler.bias)

    def forward(self, *args, **kwargs):
        # get encoder outputs
        features = self.bert_forward(*args, **kwargs)[2]

        features = torch.cat([features[i][:, 0, :] for i in self.args.highway_layers], 1)
        features = torch.tanh(self.pooler(features))

        features = F.dropout(features, self.args.output_dropout_prob, self.training)
        return self.classifier(features)


@register('bert_classification_self_attn')
class BertForClassificationSelfAttn(Bert):
    @staticmethod
    def register(options):
        options.register(
            Argument('attn_hidden_size', default=100),
            Argument('output_dropout_prob', default=0.0),
        )

    def __init__(self, args):
        super().__init__(args)
        self.classifier = ClassificationHead(
            in_features=args.attn_hidden_size,
            out_features=args.num_classes,
        )
        self.dense = nn.Linear(self.args.hidden_size, args.attn_hidden_size)
        self.attention_q = nn.Parameter(
            torch.normal(torch.zeros(1, args.attn_hidden_size), self.args.initializer_range))
        nn.init.normal_(self.dense.weight, std=self.args.initializer_range)
        nn.init.zeros_(self.dense.bias)

    def forward(self, input_ids, **kwargs):
        attention_mask = input_ids.ne(self.args.pad_index).long()
        features = self.bert_forward(input_ids, attention_mask=attention_mask, **kwargs)[0]
        features = torch.tanh(self.dense(features))
        attention_mask = attention_mask.float()
        score = (self.attention_q.unsqueeze(0) * features).sum(2) * attention_mask + (1 - attention_mask) * -1e4
        score = F.softmax(score, 1).unsqueeze(2)
        features = (score * features).sum(1)
        features = F.dropout(features, self.args.output_dropout_prob, self.training)
        return self.classifier(features)

    @property
    def name_map(self):
        name_map = super().name_map.copy()
        name_map.update({
            'meta/agg/attention_q': 'attention_q',
            'meta/agg/dense/bias': 'dense/bias',
            'meta/agg/dense/kernel': 'dense/weight',
        })
        return name_map


@register('bert_lm')
class BertForLanguageModeling(Bert):
    @staticmethod
    def register(options):
        options.add_global_constraint(
            lambda args: args.layer_lr_decay == 1 or args.keep_emb_lr
        )

    def __init__(self, args):
        super().__init__(args)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.cls = nn.ModuleDict()
        self.cls['predictions'] = LMHead(
            hidden_size=self.args.hidden_size,
            vocab_size=self.args.vocab_size,
            embedding_size=getattr(self.args, 'embedding_size', self.args.hidden_size),
            activation_fn=self.args.hidden_act,
            weight=self.get_embeddings().weight,
        )

    def forward(self, *args, masked_tokens=None, **kwargs):
        # use sequence output
        features = self.bert_forward(*args, **kwargs)[0]
        outputs = self.cls['predictions'](features, masked_tokens=masked_tokens)
        return outputs

    @property
    def name_map(self):
        name_map = super().name_map.copy()
        name_map.update({
            'heads.lm.bias': 'cls.predictions.bias',
            'heads.lm.weight': 'cls.predictions.decoder.weight',
            'heads.lm.dense.weight': 'cls.predictions.transform.dense.weight',
            'heads.lm.dense.bias': 'cls.predictions.transform.dense.bias',
            'heads.lm.layer_norm.weight': 'cls.predictions.transform.LayerNorm.weight',
            'heads.lm.layer_norm.bias': 'cls.predictions.transform.LayerNorm.bias',
        })
        return name_map


@register('bert_pretrain')
class BertForPretraining(BertForLanguageModeling):
    """MLM + sequence relation cls"""

    @staticmethod
    def register(options):
        options.add_global_constraint(
            lambda args: args.layer_lr_decay == 1 or args.keep_emb_lr
        )

    def __init__(self, args):
        super().__init__(args)
        self.cls['seq_relationship'] = ClassificationHead(self.args.hidden_size, args.num_classes)

    def forward(self, *args, masked_tokens=None, **kwargs):
        seq_features, agg_features = self.bert_forward(*args, **kwargs)[:2]
        if masked_tokens is not None:
            seq_outputs = self.cls['predictions'](seq_features, masked_tokens=masked_tokens)
        else:
            # skip the computation entirely otherwise a large (and useless) matrix computation will be triggered
            # and will significantly increase the risk of OOM
            seq_outputs = None
        cls_outputs = self.cls['seq_relationship'](agg_features)
        return seq_outputs, cls_outputs

    @property
    def name_map(self):
        name_map = super().name_map.copy()
        name_map.update({
            'heads.cls.weight': 'cls.seq_relationship.weight',
            'heads.cls.bias': 'cls.seq_relationship.bias',
        })
        return name_map


@register('bert_pretrain_cl')
class BertForPretrainingCL(BertForLanguageModeling):
    """MLM + constrastive loss"""

    def __init__(self, args):
        super().__init__(args)
        self.cls['contrastive_predict'] = ClassificationHead(self.args.hidden_size, self.args.hidden_size)

    def forward(self, *args, masked_tokens=None, **kwargs):
        seq_features, agg_features = self.bert_forward(*args, **kwargs)[:2]
        if masked_tokens is not None:
            seq_outputs = self.cls['predictions'](seq_features, masked_tokens=masked_tokens)
        else:
            # skip the computation entirely otherwise a large (and useless) matrix computation will be triggered
            # and will significantly increase the risk of OOM
            seq_outputs = None
        cls_outputs = self.cls['contrastive_predict'](agg_features)
        return seq_outputs, cls_outputs


class LMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, hidden_size, vocab_size, embedding_size, activation_fn='gelu', weight=None):
        super().__init__()
        self.transform = LMTransform(hidden_size, embedding_size, activation_fn)

        # weight is the same as input embedding, but there's a LM-specific bias
        self.decoder = nn.Module()
        if weight is None:
            weight = nn.Linear(embedding_size, vocab_size, bias=False).weight
        self.decoder.weight = weight
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the unmasked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.transform(features)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.decoder.weight) + self.bias
        return x


class LMTransform(nn.Module):
    def __init__(self, hidden_size, embedding_size, activation_fn):
        super().__init__()
        self.dense = nn.Linear(hidden_size, embedding_size)
        self.activation_fn = activations[activation_fn]
        self.LayerNorm = LayerNorm(embedding_size)
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.zeros_(self.dense.bias)

    def forward(self, features):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.LayerNorm(x)
        return x


class ClassificationHead(nn.Linear):
    """Head for sentence-level classification tasks."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)
