import os
import json
import inspect
from functools import partial

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from xdpx.modules.thirdparty.transformers.modeling_bert import (
    BertModel, BertConfig, BertForPreTraining, BertForSequenceClassification,
    BertPreTrainingHeads, BertPreTrainedModel
)
from xdpx.options import Argument
from xdpx.processors import processors
from xdpx.processors.bert import BertProcessor
from xdpx.utils import io, cache_file, download_from_url, parse_model_path, \
    validate_url, should_save_meta

from . import register, Model


class Tinybert(Model):

    @staticmethod
    def register(options):
        options.register(
            Argument('teacher_config_file',
                     default='tests/sample_data/config.json',
                     validate=lambda value: io.exists(value)),
            Argument('teacher_pretrained_model', doc='pytorch state_dict file or tensorflow checkpoint path, '
                                             'can use <best> <last> to refer to the best or last pytorch checkpoint',
                     # must use independent post_process here to support "children"
                     post_process=partial(parse_model_path, args=None), type=str, required=True),
            Argument('teacher_from_tf', default=False),
            Argument('strict_size', default=True,
                     doc='otherwise will ignore size mismatch when loading the pretrained model'),
            Argument('student_config_file',
                     default='tests/sample_data/config.json',
                     validate=lambda value: io.exists(value)),
            Argument('student_pretrained_model',
                     doc='pytorch state_dict file or tensorflow checkpoint path, '
                         'can use <best> <last> to refer to the best or last pytorch checkpoint',
                     # must use independent post_process here to support "children"
                     post_process=partial(parse_model_path, args=None),
                     type=str, required=False),
            domain='Bert',
        )

        options.add_global_constraint(
            lambda args: issubclass(processors[args.processor], BertProcessor))

        def validate_pretrain_path(args):
            value = args.teacher_pretrained_model
            if not value:
                return False
            if value.startswith('http'):
                if args.teacher_from_tf:
                    raise NotImplementedError(
                        'Currently loading a zipfile containing tf checkpoints is not supported.')
                return validate_url(value)
            if args.teacher_from_tf:
                return io.exists(value + '.meta') and io.exists(
                    value + '.index') and io.exists(
                    value + '.data-00000-of-00001')
            return io.exists(value)

        options.add_global_constraint(validate_pretrain_path)

    def __init__(self, args):
        super().__init__(args)

        with io.open(args.teacher_config_file, 'r') as f:
            self.teacher_config = BertConfig.from_dict(json.loads(f.read()))
        with io.open(args.student_config_file, 'r') as f:
            self.student_config = BertConfig.from_dict(json.loads(f.read()))
        if hasattr(args, 'vocab_size'):
            assert self.student_config.vocab_size == args.vocab_size
        else:
            args.vocab_size = self.config.teacher_vocab_size
        assert self.student_config.vocab_size == \
            self.teacher_config.vocab_size

        extra_bert_config = self.get_extra_bert_config()
        for key, val in extra_bert_config.items():
            setattr(self.teacher_config, key, val)
            setattr(self.student_config, key, val)

        self.teacher_model = None
        self.student_model = None
        self.save_bert_config()

    def get_extra_bert_config(self):
        extra_bert_config = {
            'output_hidden_states': True,
            'output_attentions': True,
        }
        return extra_bert_config

    @classmethod
    def build(cls, args):
        model = cls(args)
        teacher_model_path = None
        student_model_path = None
        if args.teacher_pretrained_model and hasattr(args, '__cmd__') and args.__cmd__ == 'train':
            teacher_model_path = args.teacher_pretrained_model
            if args.teacher_pretrained_model.startswith('http'):
                teacher_model_path = download_from_url(args.teacher_pretrained_model)
        if args.student_pretrained_model and hasattr(args, '__cmd__') and args.__cmd__ == 'train':
            student_model_path = args.student_pretrained_model
            if args.student_pretrained_model.startswith('http'):
                student_model_path = download_from_url(
                    args.teacher_pretrained_model)
        if hasattr(args, '__cmd__') and args.__cmd__ == 'train':
            model.load(student_model_path, teacher_model_path, from_tf=args.teacher_from_tf)
        return model

    def load(self, student_path, teacher_path=None, from_tf=False):
        if teacher_path:
            if from_tf:
                load_tf_weights_in_bert(self.teacher_model, self.teacher_config, teacher_path, strict=self.args.strict_size)
            else:
                self._load_from_pt(teacher_path, self.teacher_model)
        if student_path:
            self._load_from_pt(student_path, self.student_model)

    def save_bert_config(self):
        if should_save_meta(self.args):
            with io.open(os.path.join(self.args.save_dir, f'config.json'),
                         'w') as f:
                config = self.student_config.to_dict()
                json.dump(config, f, indent=2)

    def state_dict(self):
        return self.student_model.state_dict()

    def _load_from_pt(self, path, module):
        with io.open(path, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            # assume the teacher and student have same map
            for old_name, new_name in self.name_map.items():
                if old_name in state_dict:
                    state_dict[new_name] = state_dict.pop(old_name)
            # avoid using set substraction to keep the original order
            model_state_dict = module.state_dict()
            model_keys = model_state_dict.keys()
            missing_keys = [key for key in model_keys if key not in state_dict]
            unexpected_keys = [key for key in state_dict if key not in model_keys]
            mismatched_keys = [key for key in model_keys if key in state_dict and
                               state_dict[key].shape != model_state_dict[key].shape]
            if self.args.strict_size and mismatched_keys:
                raise RuntimeError(f'Found size mismatch when strict_size=True: {" ".join(mismatched_keys)}')
            for key in mismatched_keys:
                del state_dict[key]
            module.load_state_dict(state_dict, strict=False)
            print(f'| Weights loaded for {module.__class__.__name__} from {path}.')
            if missing_keys:
                print(f'| Missing keys:\n|\t' + '\n|\t'.join(missing_keys))
            if unexpected_keys:
                print(f'| Unexpected keys:\n|\t' + '\n|\t'.join(unexpected_keys))
            if mismatched_keys:
                print(f'| Mismatched keys:\n|\t' + '\n|\t'.join(mismatched_keys))
            print()

    def bert_forward(self, module, input_ids, attention_mask=None,
                     **kwargs):
        "connects to HuggingFace bert encoder"
        if attention_mask is None:
            attention_mask = input_ids.ne(self.args.pad_index).long()
        bert_args = set(inspect.getfullargspec(module.forward).args[3:])
        assert not kwargs.keys() - bert_args, 'Unexpected arguments for BertModel: ' + ' '.join(
            kwargs.keys() - bert_args)
        if 'inputs_embeds' in kwargs:
            input_ids = None
        # returns sequence_output, pooled_output, (hidden_states), (attentions)
        return module(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def forward(self, input_ids, token_type_ids, **kwargs):
        raise NotImplementedError

    def trainable_parameters(self):

        def get_param_groups(module, **kwargs):
            weight_decay = []
            no_weight_decay = []
            for name, param in module.named_parameters():
                if param.requires_grad:
                    if 'LayerNorm' in name or 'bias' in name:
                        no_weight_decay.append(param)
                    else:
                        weight_decay.append(param)
            return [
                {**kwargs, 'params': weight_decay},
                {**kwargs, 'params': no_weight_decay, 'weight_decay': 0.0},
            ]

        return get_param_groups(self.student_model)

    @property
    def name_map(self):
        return {}

    @property
    def dummy_inputs(self):
        input_ids = torch.randint(1, self.args.vocab_size, (8, 16))
        return input_ids

@register('tinybert_for_gd')
class TinyBertForGD(Tinybert):
    "TinyBert for general distillation"

    def __init__(self, args):
        super().__init__(args)
        self.teacher_model = BertForPreTraining(self.teacher_config)
        self.student_model = TinyBertForPreTraining(
            self.student_config, self.teacher_config.hidden_size)

    def forward(self, input_ids, **kwargs):
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.bert_forward(self.teacher_model, input_ids, **kwargs)
        student_outputs = self.bert_forward(self.student_model, input_ids, **kwargs)
        # compute loss
        return teacher_outputs, student_outputs

    @property
    def name_map(self):
        return {
            "heads.lm.bias": "cls.predictions.bias",
            "heads.lm.dense.bias": "cls.predictions.transform.dense.bias",
            "heads.lm.dense.weight": "cls.predictions.transform.dense.weight",
            "heads.lm.layer_norm.bias": "cls.predictions.transform.LayerNorm.bias",
            "heads.lm.layer_norm.weight": "cls.predictions.transform.LayerNorm.weight",
            "heads.lm.weight": "cls.predictions.decoder.weight",
            'heads.cls.weight': 'cls.seq_relationship.weight',
            'heads.cls.bias': 'cls.seq_relationship.bias'
        }


@register('tinybert_encoder_only_distill')
class TinyBertEncoderOnlyDistill(Tinybert):
    "TinyBert for general distillation"

    def __init__(self, args):
        super().__init__(args)
        self.teacher_model = BertEncoderForPreTraining(self.teacher_config)
        self.student_model = TinyBertEncoderOnly(
            self.student_config, self.teacher_config.hidden_size)

    def forward(self, input_ids, **kwargs):
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.bert_forward(self.teacher_model, input_ids, **kwargs)
        student_outputs = self.bert_forward(self.student_model, input_ids, **kwargs)
        # compute loss
        return teacher_outputs, student_outputs


@register('minilm')
class MiniLM(Tinybert):
    "TinyBert for general distillation"

    def __init__(self, args):
        super().__init__(args)
        self.teacher_model = BertEncoderForPreTraining(self.teacher_config)
        self.student_model = BertEncoderForPreTraining(self.student_config)

    def get_extra_bert_config(self):
        return {
            'output_attentions': True,
            'output_value_attentions': True,
        }

    def forward(self, input_ids, **kwargs):
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.bert_forward(self.teacher_model, input_ids, **kwargs)
        student_outputs = self.bert_forward(self.student_model, input_ids, **kwargs)
        # compute loss
        return teacher_outputs, student_outputs


@register('tinybert_for_td')
class TinyBertForTD(Tinybert):
    "TinyBert for task distillation"

    def __init__(self, args):
        super().__init__(args)
        self.teacher_model = BertForSequenceClassification(self.teacher_config)
        self.student_model = TinyBertForSequenceClassification(
            self.student_config, self.teacher_config.hidden_size)

    def forward(self, input_ids, **kwargs):
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.bert_forward(self.teacher_model, input_ids, **kwargs)
        student_outputs = self.bert_forward(self.student_model, input_ids, **kwargs)
        # compute loss
        return teacher_outputs, student_outputs

    @property
    def name_map(self):
        return {
            'heads.classification.weight': 'classifier.weight',
            'heads.classification.bias': 'classifier.bias',
        }


class TinyBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config, fit_size):
        super(TinyBertForPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        assert self.config.output_hidden_states and self.config.output_attentions
        self.hidden_loss_linear = nn.Linear(config.hidden_size, fit_size)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                masked_lm_labels=None, next_sentence_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output, pooled_output = outputs[:2]

        hidden_states = outputs[2]
        tmp = []
        for l_id, sequence_layer in enumerate(hidden_states):
            tmp.append(self.hidden_loss_linear(sequence_layer))
        hidden_states = tmp
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score, hidden_states) + outputs[3:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


class TinyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, fit_size):
        super(TinyBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        assert self.config.output_hidden_states and self.config.output_attentions
        self.hidden_loss_linear = nn.Linear(config.hidden_size, fit_size)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        hidden_states = outputs[2]
        tmp = []
        for l_id, sequence_layer in enumerate(hidden_states):
            tmp.append(self.hidden_loss_linear(sequence_layer))
        hidden_states = tmp

        outputs = (logits, hidden_states) + outputs[3:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertEncoderForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoderForPreTraining, self).__init__(config)

        self.bert = BertModel(config)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                masked_lm_labels=None, next_sentence_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        return outputs[2:]


class TinyBertEncoderOnly(BertPreTrainedModel):
    def __init__(self, config, fit_size):
        super(TinyBertEncoderOnly, self).__init__(config)

        self.bert = BertModel(config)

        assert self.config.output_hidden_states and self.config.output_attentions
        self.hidden_loss_linear = nn.Linear(config.hidden_size, fit_size)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                masked_lm_labels=None, next_sentence_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output, pooled_output = outputs[:2]

        hidden_states = outputs[2]
        tmp = []
        for l_id, sequence_layer in enumerate(hidden_states):
            tmp.append(self.hidden_loss_linear(sequence_layer))
        hidden_states = tmp

        outputs = (hidden_states,) + outputs[3:]  # add hidden states and attention if they are here

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


def load_tf_weights_in_bert(model, config, tf_path, extra_name_map={}, strict=True):
    """ 
    Reference: https://github.com/huggingface/transformers/blob/master/transformers/modeling_bert.py
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        raise ImportError("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
    
    name_map = {
    }
    
    def back_translate(name):
        "translate pytorch name mapping back to tf"
        name = re.sub(r'\.', '/', name)
        name = name.replace('/weight', '/kernel')
        return name

    name_map.update(extra_name_map)
    name_map = {back_translate(key): back_translate(val) for key, val in name_map.items()}
    
    # Load weights from TF model
    tf_path = io.abspath(tf_path)
    for postfix in '.data-00000-of-00001 .index .meta'.split():
        cached_path = cache_file(tf_path + postfix)
    tf_path = cached_path[:-5]
    init_vars = tf.train.list_variables(tf_path)
    if not init_vars:
        raise FileNotFoundError('invalid tf checkpoint path: ' + tf_path)
    names = []
    arrays = []
    for name, _ in init_vars:
        array = tf.train.load_variable(tf_path, name)
        name = name_map.get(name, name)
        names.append(name)
        arrays.append(array)

    model_vars = list(model.named_parameters())
    num_vars = len(model_vars)
    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            continue
        pointer = model
        try:
            for m_name in name:
                if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                    l = re.split(r'_(\d+)', m_name)
                else:
                    l = [m_name]
                if l[0] == 'kernel' or l[0] == 'gamma':
                    pointer = getattr(pointer, 'weight')
                elif l[0] == 'output_bias' or l[0] == 'beta':
                    pointer = getattr(pointer, 'bias')
                elif l[0] == 'output_weights':
                    pointer = getattr(pointer, 'weight')
                elif l[0] == 'squad':
                    pointer = getattr(pointer, 'classifier')
                else:
                    pointer = getattr(pointer, l[0])
        
                if len(l) >= 2:
                    num = int(l[1])
                    pointer = pointer[num]
            if m_name[-11:] == '_embeddings':
                pointer = getattr(pointer, 'weight')
            elif m_name == 'kernel':
                array = np.transpose(array)
            try:
                assert pointer.shape == array.shape, f'size mismatch for {"/".join(name)}: '\
                    f'expected {str(pointer.shape)}, found {str(array.shape)}'
            except AssertionError as e:
                if strict:
                    raise e
                print('| Skipped due to ' + str(e))
                continue
        except AttributeError:
            print("| Unexpected keys: {}".format("/".join(name)))
            continue
        pointer.data = torch.from_numpy(array)
        model_vars = [(name, var) for name, var in model_vars if var is not pointer]
    print(f'| Load {num_vars - len(model_vars)} variables from {tf_path}')
    if model_vars:
        print('| Missing keys:')
        for name, _ in model_vars:
            print(f'|  {name}')
    return model
