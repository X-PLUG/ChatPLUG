import torch
import torch.nn.functional as F
from . import register, Model
from xdpx.options import Argument
from transformers import AutoConfig, AutoModel
from .bert import ClassificationHead
from xdpx.utils import io, parse_model_path, download_from_url
from typing import Union, List

@register('bert_dpr_qa')
class BertDPRQA(Model):
    @staticmethod
    def register(options):
        options.register(
            Argument('q_auto_config', default="hfl/chinese-roberta-wwm-ext"),
            Argument('p_auto_config', default="hfl/chinese-roberta-wwm-ext"),
            Argument('q_pretrained_model',
                     post_process=lambda val: parse_model_path(val, None), type=str,
                     children=[
                         Argument('bert_strict_size', default=True,
                                  doc='otherwise will ignore size mismatch when loading the pretrained model'),
                     ]
                     ),

            Argument('p_pretrained_model',
                     post_process=lambda val: parse_model_path(val, None), type=str
                     ),
            Argument('layer_lr_decay', default=1., doc='layer-wise learning rate decay', children={
                lambda value: value < 1.: [
                    Argument('keep_emb_lr', default=True, doc='whether to keep emb lr the same as initial'),
                ]
            }),
            Argument('top_lr_ratio', default=1.0, doc='initial lr for the top (non-bert) layers'),
            Argument('bert_wd', type=float, doc='if not None, set separate weight decay for bert'),
            domain='DPR',
        )

    def __init__(self, args):
        super().__init__(args)
        self.q_bert = None
        self.p_bert = None

        config = AutoConfig.from_pretrained(self.args.q_auto_config)
        self.q_hidden_size = config.hidden_size
        self.q_bert = AutoModel.from_config(config)

        config = AutoConfig.from_pretrained(self.args.p_auto_config)
        self.p_hidden_size = config.hidden_size
        self.p_bert = AutoModel.from_config(config)

    @classmethod
    def build(cls, args):
        model = cls(args)
        q_pretrained_model_path = None
        p_pretrained_model_path = None
        if args.q_pretrained_model and hasattr(args, '__cmd__') and args.__cmd__ == 'train':
            q_pretrained_model_path = args.q_pretrained_model
            if q_pretrained_model_path.startswith('http'):
                q_pretrained_model_path = download_from_url(q_pretrained_model_path)

        if args.p_pretrained_model and hasattr(args, '__cmd__') and args.__cmd__ == 'train':
            p_pretrained_model_path = args.p_pretrained_model
            if p_pretrained_model_path.startswith('http'):
                p_pretrained_model_path = download_from_url(p_pretrained_model_path)

        if hasattr(args, '__cmd__') and args.__cmd__ == 'train':
            model.load((q_pretrained_model_path, p_pretrained_model_path))
        return model

    def load(self, pretrained_model_path, from_tf=False):
        if isinstance(pretrained_model_path, (tuple, list)):
            assert len(pretrained_model_path) == 2
            q_pretrained_model_path, p_pretrained_model_path = pretrained_model_path
            if q_pretrained_model_path and self.q_bert is not None:
                self._load_from_pt(q_pretrained_model_path, self.q_bert, self.args.bert_strict_size)
            if p_pretrained_model_path and self.p_bert is not None:
                self._load_from_pt(p_pretrained_model_path, self.p_bert, self.args.bert_strict_size)
        else:
            super(BertDPRQA, self).load(pretrained_model_path, from_tf)

    def _load_from_pt(self, path, module, strict_size=False):
        with io.open(path, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            model_state_dict = module.state_dict()
            model_keys = model_state_dict.keys()

            for old_name in list(state_dict.keys()):
                new_name = old_name[old_name.index('.') + 1:]  # remove bert. /roberta. prefix
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

    def encode_question(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        """connects to HuggingFace bert encoder"""
        return self.q_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            **kwargs)

    def encode_passage(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        """connects to HuggingFace bert encoder"""
        return self.p_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            **kwargs)

    def forward(self, question_input_ids, passage_input_ids, *args, **kwargs):
        q_features = self.encode_question(question_input_ids).pooler_output
        p_features = self.encode_passage(passage_input_ids).pooler_output
        return q_features, p_features

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


