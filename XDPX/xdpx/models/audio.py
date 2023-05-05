import torch
import torch.nn.functional as F
from . import register, Model
from xdpx.options import Argument
from transformers import Wav2Vec2ForCTC, AutoConfig, AutoModel
from .bert import ClassificationHead
from xdpx.utils import io, parse_model_path, download_from_url
from typing import Union, List


class AudioMM(Model):
    @staticmethod
    def register(options):
        options.register(
            Argument('text_auto_config', default="xlm-roberta-base"),
            Argument('audio_auto_config', default="facebook/wav2vec2-large-xlsr-53"),
            Argument('text_pretrained_model',
                     post_process=lambda val: parse_model_path(val, None), type=str,
                     children=[
                         Argument('bert_strict_size', default=True,
                                  doc='otherwise will ignore size mismatch when loading the pretrained model'),
                     ]
                     ),

            Argument('audio_pretrained_model',
                     post_process=lambda val: parse_model_path(val, None), type=str,
                     children=[
                         Argument('audio_strict_size', default=True,
                                  doc='otherwise will ignore size mismatch when loading the pretrained model'),
                     ]
                     ),

            Argument('modality', default='text,audio'),
            Argument('mask_feature_prob', default=0.1),
            Argument('use_feature_extractor_only', default=True),
            Argument('gradient_checkpointing', default=False),
            Argument('freeze_feature_extractor', default=False),
            Argument('freeze_bert', default=False),
            Argument('freeze_wav2vec', default=False),
            Argument('wav2vec_learning_rate', default=1e-4),

            Argument('layer_lr_decay', default=1., doc='layer-wise learning rate decay', children={
                lambda value: value < 1.: [
                    Argument('keep_emb_lr', default=True, doc='whether to keep emb lr the same as initial'),
                ]
            }),
            Argument('top_lr_ratio', default=1.0, doc='initial lr for the top (non-bert) layers'),
            Argument('bert_wd', type=float, doc='if not None, set separate weight decay for bert'),
            domain='AudioMM',
        )

    def __init__(self, args):
        super().__init__(args)
        self.bert = None
        self.wav2vec2 = None
        self.use_audio_modal = 'audio' in self.args.modality
        self.use_text_modal = 'text' in self.args.modality
        if self.args.audio_auto_config and self.use_audio_modal:
            config = AutoConfig.from_pretrained(self.args.audio_auto_config)
            self.audio_hidden_size = config.hidden_size
            config.mask_feature_prob = args.mask_feature_prob

            self.wav2vec2 = Wav2Vec2ForCTC(config)

            if self.args.freeze_feature_extractor:
                self.wav2vec2.freeze_feature_extractor()

            if self.args.freeze_wav2vec:
                self.freeze_wav2vec_model()

            if self.args.gradient_checkpointing:
                self.wav2vec2.gradient_checkpointing_enable()

        if self.args.text_auto_config and self.use_text_modal:
            config = AutoConfig.from_pretrained(self.args.text_auto_config)
            self.bert_hidden_size = config.hidden_size
            self.bert = AutoModel.from_config(config)

            if self.args.freeze_bert:
                self.freeze_bert_model()

            if self.args.gradient_checkpointing:
                self.bert.gradient_checkpointing_enable()

    def freeze_wav2vec_model(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def freeze_bert_model(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    @classmethod
    def build(cls, args):
        model = cls(args)
        text_pretrained_model_path = None
        audio_pretrained_model_path = None
        if args.text_pretrained_model and model.use_text_modal and hasattr(args, '__cmd__') and args.__cmd__ == 'train':
            text_pretrained_model_path = args.text_pretrained_model
            if args.text_pretrained_model.startswith('http'):
                text_pretrained_model_path = download_from_url(args.text_pretrained_model)

        if args.audio_pretrained_model and model.use_audio_modal and hasattr(args,
                                                                             '__cmd__') and args.__cmd__ == 'train':
            audio_pretrained_model_path = args.audio_pretrained_model
            if args.audio_pretrained_model.startswith('http'):
                audio_pretrained_model_path = download_from_url(
                    args.audio_pretrained_model)

        if hasattr(args, '__cmd__') and args.__cmd__ == 'train':
            model.load((text_pretrained_model_path, audio_pretrained_model_path))
        return model

    def load(self, pretrained_model_path, from_tf=False):
        if isinstance(pretrained_model_path, (tuple, list)):
            assert len(pretrained_model_path) == 2
            text_pretrained_model_path, audio_pretrained_model_path = pretrained_model_path
            if text_pretrained_model_path and self.bert is not None:
                self._load_from_pt(text_pretrained_model_path, self.bert, self.args.bert_strict_size)
            if audio_pretrained_model_path and self.wav2vec2 is not None:
                self._load_from_pt(audio_pretrained_model_path, self.wav2vec2, self.args.audio_strict_size)
        else:
            super(AudioMM, self).load(pretrained_model_path, from_tf)

    def _load_from_pt(self, path, module, strict_size=False):
        with io.open(path, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            model_state_dict = module.state_dict()
            model_keys = model_state_dict.keys()

            if module == self.bert:
                for old_name in list(state_dict.keys()):
                    new_name = old_name[old_name.index('.') + 1:]  # remove bert. /roberta. prefix
                    state_dict[new_name] = state_dict.pop(old_name)
            if module == self.wav2vec2:
                for old_name in list(state_dict.keys()):
                    if 'wav2vec2.wav2vec2.' in old_name:
                        new_name = old_name[old_name.index('.') + 1:]  # remove wav2vec2 dup prefix
                        state_dict[new_name] = state_dict.pop(old_name)

            missing_keys = [key for key in model_keys if key not in state_dict]
            unexpected_keys = [key for key in state_dict if key not in model_keys]
            mismatched_keys = [key for key in model_keys if key in state_dict and
                               state_dict[key].shape != model_state_dict[key].shape]
            if strict_size and mismatched_keys:
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

    def bert_forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        """connects to HuggingFace bert encoder"""
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            **kwargs)

    def wav2vec_forward(self, audio_input_values, audio_attention_mask=None, **kwargs):
        """connects to HuggingFace , return last_hidden_state or extract_features"""
        if self.args.use_feature_extractor_only:
            feature_extractor = self.wav2vec2.wav2vec2.feature_extractor
            extract_features = feature_extractor(audio_input_values)
            extract_features = extract_features.transpose(1, 2)
            return extract_features
        else:
            output = self.wav2vec2.wav2vec2(
                input_values=audio_input_values,
                attention_mask=audio_attention_mask,
                output_hidden_states=False,
                return_dict=True,
                **kwargs)
            return output.last_hidden_state

    def wav2vec_aggregate(self, hidden_states, audio_attention_mask=None, pooling_method="mean"):
        if audio_attention_mask is not None:
            padding_mask = self.wav2vec2._get_feature_vector_attention_mask(hidden_states.shape[1],
                                                                            audio_attention_mask)
        else:
            padding_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)

        if pooling_method == "max":
            hidden_states[~padding_mask] = -1e4
            pooled_output = hidden_states.max(dim=1).values
        elif pooling_method == "mean":
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.mean(dim=1)
        else:
            raise NotImplementedError

        return pooled_output

    def forward(self, input_ids, audio_input_values, audio_attention_mask, *args, **kwargs):
        raise NotImplementedError

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

        # add params in seperate groups (even with the same hyper-params) will result in minor numerical differences
        # so the exact match of training trace is not expected
        top_learning_rate = self.args.learning_rate * self.args.top_lr_ratio
        bert_weight_decay = self.args.weight_decay if self.args.bert_wd is None else self.args.bert_wd
        for name, module in self.named_children():
            if name == 'bert':
                param_groups.extend(
                    self.get_param_groups(module.pooler, lr=top_learning_rate))

                for i, layer in enumerate(self.bert.encoder.layer[::-1]):
                    layer_lr = self.args.learning_rate * (self.args.layer_lr_decay ** i)
                    param_groups.extend(self.get_param_groups(layer, lr=layer_lr, weight_decay=bert_weight_decay))

                emb_lr = self.args.learning_rate if getattr(self.args, 'keep_emb_lr', True) else layer_lr
                for name, module in self.bert.embeddings.named_children():
                    if name == 'word_embeddings' and hasattr(self, 'cls'):
                        # the embedding is tied and does not need to add again
                        continue
                    param_groups.extend(self.get_param_groups(module, lr=emb_lr, weight_decay=bert_weight_decay))

            elif name == 'wav2vec2':
                param_groups.extend(self.get_param_groups(module, lr=self.args.wav2vec_learning_rate))
            else:
                param_groups.extend(self.get_param_groups(module, lr=top_learning_rate))

        ground_truth = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_trainable = sum(p.numel() for param_group in param_groups for p in param_group['params'])
        assert ground_truth == num_trainable, f'{num_trainable} of {ground_truth} trainable parameters are registered in the optimizer.'
        return param_groups

    @property
    def name_map(self):
        return {}

    @property
    def dummy_inputs(self):
        input_ids = torch.randint(1, self.args.vocab_size, (4, 16))
        audio_input_values = torch.randn((4, 50000))
        return input_ids, audio_input_values


@register('audiomm_classification')
class AudioMMConcatForClassification(AudioMM):
    @staticmethod
    def register(options):
        options.register(
            Argument('output_dropout_prob', default=0.0),
            Argument('wav2vec_pooling_method', default="max"),
        )

    def __init__(self, args):
        super().__init__(args)
        assert self.use_audio_modal or self.use_text_modal
        in_features = 0
        if self.use_text_modal:
            in_features += self.bert_hidden_size
        if self.use_audio_modal:
            in_features += self.audio_hidden_size

        self.classifier = ClassificationHead(
            in_features=in_features,
            out_features=args.num_classes,
        )

    def forward(self, input_ids, audio_input_values, audio_attention_mask=None, *args, **kwargs):
        features = []
        if self.use_text_modal:
            text_features = self.bert_forward(input_ids).pooler_output
            features.append(text_features)

        if self.use_audio_modal:
            hidden_states = self.wav2vec_forward(audio_input_values, audio_attention_mask)
            audio_features = self.wav2vec_aggregate(hidden_states, audio_attention_mask,
                                                    self.args.wav2vec_pooling_method)  # batch_size, dim
            features.append(audio_features)

        if len(features) > 1:
            features = torch.cat(features, dim=1)
        else:
            features = features[0]
        features = F.dropout(features, self.args.output_dropout_prob, self.training)
        return self.classifier(features)


@register('audiomm_flex_classification')
class AudioMMFlexForClassification(AudioMM):
    @staticmethod
    def register(options):
        options.register(
            Argument('output_dropout_prob', default=0.0),
            Argument('wav2vec_pooling_method', default="mean"),
        )

    def __init__(self, args):
        super().__init__(args)
        assert self.use_audio_modal or self.use_text_modal
        self.classifier_audio_modal, self.classifier_text_modal = None, None
        if self.use_text_modal:
            self.classifier_text_modal = ClassificationHead(
                in_features=self.bert_hidden_size,
                out_features=args.num_classes,
            )
        if self.use_audio_modal:
            self.classifier_audio_modal = ClassificationHead(
                in_features=self.audio_hidden_size,
                out_features=args.num_classes,
            )

    def forward(self, input_ids, audio_input_values, audio_attention_mask=None, *args, **kwargs):
        logits1, logits2 = None, None
        if self.use_text_modal and input_ids is not None:
            text_features = self.bert_forward(input_ids).pooler_output
            text_features = F.dropout(text_features, self.args.output_dropout_prob, self.training)
            logits1 = self.classifier_text_modal(text_features)
        elif self.use_text_modal:
            print('| WARNING: use_text_modal is true but input_ids is None')

        if self.use_audio_modal and audio_input_values is not None:
            hidden_states = self.wav2vec_forward(audio_input_values, audio_attention_mask)
            audio_features = self.wav2vec_aggregate(hidden_states, audio_attention_mask,
                                                    self.args.wav2vec_pooling_method)  # batch_size, dim
            audio_features = F.dropout(audio_features, self.args.output_dropout_prob, self.training)
            logits2 = self.classifier_audio_modal(audio_features)
        elif self.use_audio_modal:
            print('| WARNING: use_audio_modal is true but audio_input_values is None')

        if logits1 is not None and logits2 is not None:
            logits = logits1 + logits2
        elif logits1 is not None:
            logits = logits1
        else:
            logits = logits2
        return logits, logits1, logits2


@register('audiomm_cl_alignment')
class AudioMMCLAlignment(AudioMM):
    @staticmethod
    def register(options):
        options.register(
            Argument('output_dropout_prob', default=0.0),
            Argument('wav2vec_pooling_method', default="max"),
        )

    def __init__(self, args):
        super().__init__(args)
        assert self.use_audio_modal and self.use_text_modal

        self.classifier = None
        if self.audio_hidden_size != self.bert_hidden_size:
            self.classifier = ClassificationHead(
                in_features=self.audio_hidden_size,
                out_features=self.bert_hidden_size,
            )

    def forward(self, input_ids, audio_input_values, audio_attention_mask=None, *args, **kwargs):
        text_features = self.bert_forward(input_ids).pooler_output
        hidden_states = self.wav2vec_forward(audio_input_values, audio_attention_mask)
        audio_features = self.wav2vec_aggregate(hidden_states, audio_attention_mask,
                                                self.args.wav2vec_pooling_method)  # batch_size, dim
        if self.classifier is not None:
            audio_features = F.dropout(audio_features, self.args.output_dropout_prob, self.training)
            audio_features = self.classifier(audio_features)

        return text_features, audio_features
