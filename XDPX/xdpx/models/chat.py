import torch
from icecream import ic
from transformers import AutoConfig, T5ForConditionalGeneration

from xdpx.options import Argument
from xdpx.utils import io, download_from_url

from xdpx.modules.thirdparty.palm import PalmConfig, PalmForConditionalGeneration
from xdpx.modules.thirdparty.palm import WrapPalmForConditionalGeneration
from xdpx.modules.thirdparty.plugv2.model_plug import PlugConfig, PlugForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from . import register, Model
from copy import deepcopy
try:
    import allspark
    import torch.utils.dlpack
except Exception as e:
    print(e)


class ChatBase(Model):
    @staticmethod
    def register(options):
        options.register(
            Argument('gradient_checkpointing', default=False),
            domain='ChatBase',
        )

    @classmethod
    def build(cls, args):
        model = cls(args)
        if hasattr(args, '__cmd__') and args.__cmd__ == 'train':
            pretrained_model_path = args.pretrained_model
            if pretrained_model_path.startswith('http'):
                pretrained_model_path = download_from_url(args.pretrained_model)
            model.load(pretrained_model_path)
        return model

    def eval(self):
        super().eval()
        self.backbone.eval()

    def generate(self, input_ids, *args, **kwargs):
        self.backbone.eval()
        with torch.no_grad():
            response = self.backbone.generate(input_ids, *args, **kwargs)
        return response

    def forward(self, input_ids, decoder_input_ids, *args, **kwargs):
        return self.backbone.forward(input_ids=input_ids, labels=decoder_input_ids, *args, **kwargs)

    def get_param_groups(self, module, **kwargs):
        oridinary = []
        no_weight_decay = []
        for name, param in module.named_parameters():
            if param.requires_grad:
                if 'layer_norm' in name or 'bias' in name:
                    no_weight_decay.append(param)
                else:
                    oridinary.append(param)
        return [
            {**kwargs, 'params': oridinary},
            {**kwargs, 'params': no_weight_decay, 'weight_decay': 0.0},
        ]

    @property
    def name_map(self):
        return {}

    @property
    def dummy_inputs(self):
        input_ids = torch.randint(1, self.args.vocab_size, (4, 16))
        return input_ids


@register('t5chat')
class T5Chat(ChatBase):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = None
        if self.args.auto_model:
            print(f'| before AutoConfig.from_pretrained..')
            config = AutoConfig.from_pretrained(self.args.auto_model)
            print(f'| after AutoConfig.from_pretrained..')

            if self.args.gradient_checkpointing:
                config.use_cache = False
                self.backbone = T5ForConditionalGeneration(config)
                self.backbone.gradient_checkpointing_enable()
            else:
                self.backbone = T5ForConditionalGeneration(config)
            print(f'| after T5ForConditionalGeneration create..')
        if hasattr(args,'core_chat_half_precision') and args.core_chat_half_precision and torch.cuda.is_available():
            self.backbone.half()

    def load(self, pretrained_model_path, from_tf=False):
        with io.open(pretrained_model_path, 'rb') as f:
            print(f'| before torch.load..')
            state_dict = torch.load(f, map_location='cpu')
            print(f'| after torch.load..')
            model_state_dict = self.backbone.state_dict()
            model_keys = model_state_dict.keys()

            for old_name in list(state_dict.keys()):
                if old_name.startswith("module."):
                    new_name = old_name.replace("module.", "")
                    state_dict[new_name] = state_dict.pop(old_name)
                    old_name = new_name
                if old_name.startswith('backbone.encoder.encoder'):
                    # wrap encoder name map
                    new_name = old_name.replace('backbone.encoder.encoder', 'encoder').replace('module.layer', 'layer')
                    state_dict[new_name] = state_dict.pop(old_name)
                elif old_name.startswith('backbone'):
                    new_name = old_name[old_name.index('.') + 1:]  # remove backbone. prefix
                    state_dict[new_name] = state_dict.pop(old_name)

            missing_keys = [key for key in model_keys if key not in state_dict]
            unexpected_keys = [key for key in state_dict if key not in model_keys]
            mismatched_keys = [key for key in model_keys if key in state_dict and
                               state_dict[key].shape != model_state_dict[key].shape]
            for key in mismatched_keys:
                del state_dict[key]

            self.backbone.load_state_dict(state_dict, strict=False)
            print(f'| Weights loaded for {self.backbone.__class__.__name__} from {pretrained_model_path}.')
            if missing_keys:
                print(f'| Missing keys:\n|\t' + '\n|\t'.join(missing_keys))
            if unexpected_keys:
                print(f'| Unexpected keys:\n|\t' + '\n|\t'.join(unexpected_keys))
            if mismatched_keys:
                print(f'| Mismatched keys:\n|\t' + '\n|\t'.join(mismatched_keys))
            print()

    def forward(self, input_ids, decoder_input_ids, *args, **kwargs):
        attention_mask = input_ids.ne(0).long()
        return self.backbone.forward(input_ids=input_ids, attention_mask=attention_mask, labels=decoder_input_ids, *args, **kwargs)


@register('fidt5chat')
class FIDT5Chat(T5Chat):
    def __init__(self, args, backbone=None):
        super().__init__(args)
        self.use_checkpoint = self.args.gradient_checkpointing
        if backbone is not None:
            self.backbone = backbone
        self.wrap_encoder()

    def wrap_encoder(self):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        # from xdpx.utils.thirdparty.onnx_transformers.models.t5.onnx_model import OnnxT5
        # if isinstance(self.backbone, OnnxT5):
        #     self.backbone.encoder = EncoderWrapper(self.backbone.encoder, use_checkpoint=self.use_checkpoint,
        #                                            bool_apply_checkpoint_wrapper=False)
        # else:
        #     self.backbone.encoder = EncoderWrapper(self.backbone.encoder, use_checkpoint=self.use_checkpoint)
        self.backbone.encoder = EncoderWrapper(self.backbone.encoder, use_checkpoint=self.use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.backbone.encoder = self.backbone.encoder.encoder
        block = []
        for mod in self.backbone.encoder.block:
            block.append(mod.module)
        block = torch.nn.ModuleList(block)
        self.backbone.encoder.block = block

    def load(self, pretrained_model_path, from_tf=False):  # only invoked when model is not onnx format
        self.unwrap_encoder()
        super().load(pretrained_model_path)
        self.wrap_encoder()

    def generate(self, input_ids, *args, **kwargs):
        self.backbone.eval()
        with torch.no_grad():
            self.backbone.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
            response = self.backbone.generate(input_ids, *args, **kwargs)
        return response

    def forward(self, input_ids, decoder_input_ids, *args, **kwargs):
        attention_mask = input_ids.ne(0).long()
        if input_ids is not None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.backbone.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
            attention_mask = attention_mask.view(attention_mask.size(0),-1)
        return self.backbone.forward(input_ids=input_ids, attention_mask=attention_mask, labels=decoder_input_ids,
                                     *args, **kwargs)

    @property
    def dummy_inputs(self):
        input_ids = torch.randint(1, self.args.vocab_size, (4, 5, 16))
        return input_ids

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder, use_checkpoint=False, bool_apply_checkpoint_wrapper=True):
        super().__init__()

        self.encoder = encoder
        if bool_apply_checkpoint_wrapper:
            apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, *args, **kwargs, ):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        if attention_mask is not None:
            attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, *args, **kwargs)
        if isinstance(outputs, tuple):
            outputs = (outputs[0].view(bsz, self.n_passages * passage_length, -1),) + outputs[1:]
        else:
            outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, self.n_passages * passage_length, -1)
        return outputs


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = torch.nn.ModuleList(block)
    t5stack.block = block


class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, *args, **kwargs):

        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, *args, **kwargs)
        return output



@register('plugv2_chat')
class PlugV2Chat(ChatBase):

    @staticmethod
    def register(options):
        options.register(
            Argument('plug_config_file', default="")
        )

    def __init__(self, args):
        super().__init__(args)
        plug_config = PlugConfig.from_json_file(args.plug_config_file)
        self.backbone = PlugForConditionalGeneration(plug_config)

        if hasattr(args,'core_chat_half_precision') and args.core_chat_half_precision and torch.cuda.is_available():
            self.backbone.half()

    def load(self, pretrained_model_path, from_tf=False):
        with io.open(pretrained_model_path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            for key in list(checkpoint.keys()):
                value = checkpoint[key]
                if key.endswith("bert.embeddings.token_type_embeddings.weight") and value.shape[0] == 2:
                    checkpoint.pop(key)
                    continue
                # for old plugv2 version
                if key.startswith("translator"):
                    checkpoint.pop(key)
                    continue
                if key.startswith("module."):
                    checkpoint[key.replace("module.", "")] = checkpoint[key]
                    checkpoint.pop(key)
                if key.startswith("backbone.plug.bert.bert."):
                    checkpoint[key.replace("backbone.plug.bert.bert.", "bert.")] = checkpoint[key]
                    checkpoint.pop(key)
                elif key.startswith("backbone.plug."):
                    checkpoint[key.replace("backbone.plug.", "")] = checkpoint[key]
                    checkpoint.pop(key)
            msg = self.backbone.plug.load_state_dict(checkpoint, strict=False)
            print(f'| {msg}')

    def generate(self, input_ids, token_type_ids=None, *args, **kwargs):
        pred_result = self.backbone.translate(input_ids=input_ids, token_type_ids=token_type_ids, *args, **kwargs)[
            'predictions']
        response = [x[0].tolist() for x in pred_result]
        response = torch.tensor(response)
        return response

    def forward(self, input_ids, decoder_input_ids, token_type_ids=None, *args, **kwargs):
        loss = self.backbone.forward(src=input_ids, tgt=decoder_input_ids, token_type_ids=token_type_ids)
        return Seq2SeqLMOutput(
            loss=loss
        )


class PlugV2EncoderWrapper(torch.nn.Module):

    def __init__(self, bert):
        super().__init__()

        self.bert = bert
        self.n_passages = None

    def set_n_passages(self, n_passages):
        self.n_passages = n_passages

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, *args, **kwargs):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(bsz * self.n_passages, passage_length)
        if attention_mask is not None:
            attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        outputs = self.bert(input_ids, attention_mask, token_type_ids=token_type_ids, *args, **kwargs)
        if isinstance(outputs, tuple):
            outputs = (outputs[0].view(bsz, self.n_passages * passage_length, -1),) + outputs[1:]
        else:
            outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, self.n_passages * passage_length, -1)
        return outputs


@register('plugv2_fidchat')
class PlugV2FidChat(PlugV2Chat):
    def __init__(self, args, backbone=None):
        super().__init__(args)
        self.wrap_encoder()

    def wrap_encoder(self):
        self.backbone.plug.bert = PlugV2EncoderWrapper(self.backbone.plug.bert)

    def unwrap_encoder(self):
        self.backbone.plug.bert = self.backbone.plug.bert.bert

    def load(self, pretrained_model_path, from_tf=False):  # only invoked when model is not onnx format
        self.unwrap_encoder()
        super().load(pretrained_model_path)
        self.wrap_encoder()

    def generate(self, input_ids, token_type_ids=None, *args, **kwargs):
        n_passages = input_ids.size(1)
        self.backbone.plug.bert.set_n_passages(n_passages)
        input_ids = input_ids.view(input_ids.size(0), -1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(token_type_ids.size(0), -1)
        response = super().generate(input_ids, token_type_ids=token_type_ids, *args, **kwargs)
        return response

    def forward(self, input_ids, decoder_input_ids, token_type_ids, *args, **kwargs):
        if input_ids is not None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                n_passages = input_ids.size(1)
                self.backbone.plug.bert.set_n_passages(n_passages)
            input_ids = input_ids.view(input_ids.size(0), -1)
            token_type_ids = token_type_ids.view(input_ids.size(0), -1)
        return super().forward(input_ids, decoder_input_ids=decoder_input_ids, token_type_ids=token_type_ids, *args,
                               **kwargs)

    def load_allspark(self, allspark_gen_cfg):
        model_config = {
            "attention_probs_dropout_prob": 0.1,
            "attention_type": "self",
            "activation_function": "GELU_TANH",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 768,
            "layer_norm_eps": 1e-6,
            "layernorm_epsilon": 1e-6,
            "max_position_embeddings": 2048,
            "model_type": "PLUG",
            "num_attention_heads": 8,
            "num_hidden_layers": 12,
            "enc_layers": 12,
            "transformers_version": "4.22.0",
            "type_vocab_size": 2,
            "vocab_size": 50272,
            "decoder_start_token_id": 101,
        }

        self.as_engine = allspark.Engine()
        self.as_engine.set_device_type("CUDA")

        gen_cfg = {
            'eos_token_id': 102,
            'fusion_in_decoder': True,
        }
        gen_cfg.update(allspark_gen_cfg)

        torch_model = deepcopy(self.backbone.eval())
        torch_model['plug.decoder.pos_emb.pe'] = torch_model['plug.decoder.pos_emb.pe'].squeeze(0)

        self.as_engine.build_model_from_torch(
            model_name="palmv2_chat",
            model_type="PLUG",
            data_type="float32",
            graph_with_weights=False,
            torch_model=torch_model,
            model_config=model_config,
            generate_config=gen_cfg,
        )

    def generate_allspark(self, torch_input, max_length, bad_words_ids=None):
        torch_input["attention_mask"] = (torch_input["input_ids"] != 0).to(torch.int64)
        print(torch_input["attention_mask"].device)

        bad_words_ids = [] if bad_words_ids is None else bad_words_ids
        out = self.as_engine.run_text_generation("palmv2_chat", {
            "input_ids": torch.utils.dlpack.to_dlpack((torch_input["input_ids"])),
            "attention_mask": torch.utils.dlpack.to_dlpack(torch_input["attention_mask"]),
            "token_type_ids": torch.utils.dlpack.to_dlpack(torch_input["token_type_ids"]),
        }, max_length=max_length, bad_words_ids=bad_words_ids)
        return torch.utils.dlpack.from_dlpack(out["generated_ids"])
