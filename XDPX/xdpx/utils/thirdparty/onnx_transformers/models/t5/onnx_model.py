import os
import torch
from transformers import T5ForConditionalGeneration, AutoConfig, MT5Config
from .t5_encoder_decoder_init import T5EncoderDecoderInitHelper, T5EncoderDecoderInitInputs
from .t5_decoder import T5DecoderInputs, T5DecoderHelper
from .t5_encoder import T5EncoderInputs, T5EncoderHelper
from .t5_decoder import T5DecoderInit
from ...benchmark_helper import create_onnxruntime_session
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from .past_helper import PastKeyValuesHelper
import functools
import operator
from icecream import ic
from typing import Any, Dict
from transformers.file_utils import ModelOutput

'''
# example
# python3 -m xdpx.utils.thirdparty.onnx_transformers.models.t5.convert_to_onnx -m t5-small -w --use_gpu
# python3 -m xdpx.utils.thirdparty.onnx_transformers.models.t5.onnx_model

# on A100
ic| t_input: 'translate English to French: The universe is a dark forest.'
ic| torch_output: "L'univers est une forêt sombre."
ic| onnx_output: "L'univers est une forêt sombre."

ic| get_latency_result(torch_runtimes, 1): {'QPS': '16.72',
                                            'average_latency_ms': '59.81',
                                            'latency_90_percentile': '59.96',
                                            'latency_95_percentile': '60.13',
                                            'latency_99_percentile': '61.49',
                                            'latency_variance': '0.00',
                                            'test_times': 50}
ic| get_latency_result(onnx_runtimes, 1): {'QPS': '33.44',
                                           'average_latency_ms': '29.90',
                                           'latency_90_percentile': '30.02',
                                           'latency_95_percentile': '32.19',
                                           'latency_99_percentile': '32.76',
                                           'latency_variance': '0.00',
                                           'test_times': 50}
'''


class T5Encoder(torch.nn.Module):
    def __init__(self, encoder_sess, config):
        super().__init__()
        self.encoder_sess = encoder_sess
        self.main_input_name = "input_ids"
        self.config = config

    def forward(
            self,
            input_ids,
            attention_mask,
            inputs_embeds=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        inputs = T5EncoderInputs(input_ids, attention_mask)
        ort_outputs = T5EncoderHelper.onnxruntime_inference(self.encoder_sess, inputs)
        last_hidden_states = ort_outputs['hidden_states']

        return BaseModelOutput(last_hidden_states)


class T5DecoderInit(torch.nn.Module):
    def __init__(self, decoder_init_sess, config):
        super().__init__()
        self.decoder_init_sess = decoder_init_sess
        self.config = config

    def forward(self, decoder_input_ids, encoder_attention_mask, encoder_hidden_states):
        inputs = T5DecoderInputs(decoder_input_ids, encoder_attention_mask, encoder_hidden_states)
        ort_outputs = T5DecoderHelper.onnxruntime_inference(self.decoder_init_sess, inputs)
        logits = ort_outputs['logits']

        present_names = PastKeyValuesHelper.get_past_names(self.config.num_layers, present=True)
        past_self, past_cross = present_names[:2 * self.config.num_layers], present_names[
                                                                            2 * self.config.num_layers:]

        past_key_values = []
        for i in range(self.config.num_layers):
            tensors = (
                ort_outputs[past_self[i * 2]],
                ort_outputs[past_self[i * 2 + 1]],
                ort_outputs[past_cross[i * 2]],
                ort_outputs[past_cross[i * 2 + 1]],
            )

            past_key_values.append(tuple(tensors))
        past_key_values = tuple(past_key_values)

        return logits, past_key_values


class T5Decoder(torch.nn.Module):
    def __init__(self, decoder_sess, config):
        super().__init__()
        self.decoder_sess = decoder_sess
        self.config = config

    def forward(self, decoder_input_ids, attention_mask, encoder_hidden_states, past_key_values):
        input_past_key_values = past_key_values
        flatten_past_key_values = []

        for i in range(self.config.num_layers):
            flatten_past_key_values.append(past_key_values[i][0])
            flatten_past_key_values.append(past_key_values[i][1])
        for i in range(self.config.num_layers):
            flatten_past_key_values.append(past_key_values[i][2])
            flatten_past_key_values.append(past_key_values[i][3])

        inputs = T5DecoderInputs(decoder_input_ids,
                                 attention_mask,
                                 encoder_hidden_states,
                                 flatten_past_key_values)

        ort_outputs = T5DecoderHelper.onnxruntime_inference(self.decoder_sess, inputs)

        logits = ort_outputs['logits']

        present_names = PastKeyValuesHelper.get_past_names(self.config.num_layers, present=True)
        present_self_names = present_names[: 2 * self.config.num_layers]

        past_key_values = []
        assert len(present_names) % 4 == 0
        for i in range(self.config.num_layers):
            tensors = (
                ort_outputs[present_self_names[i * 2]],
                ort_outputs[present_self_names[i * 2 + 1]],
                input_past_key_values[i][2],
                input_past_key_values[i][3]
            )

            past_key_values.append(tuple(tensors))
        past_key_values = tuple(past_key_values)
        return logits, past_key_values


class OnnxT5(T5ForConditionalGeneration):
    """creates a T5 model using onnx sessions (encode, decoder & init_decoder)"""

    def __init__(self, pretrained_version='google/mt5-base', save_dir=None, provider='cuda', config=None):
        if not config:
            config = AutoConfig.from_pretrained(pretrained_version)
        super().__init__(config)

        # monkeypatch to work for MT5
        if (
                isinstance(pretrained_version, str)
                and "mt5" in pretrained_version.lower()
        ) or (
                hasattr(pretrained_version, "name_or_path")
                and "mt5" in pretrained_version.name_or_path
        ):
            self.model_type = "mt5"
            self.config_class = MT5Config
            self._keys_to_ignore_on_load_missing = [
                r"encoder\.embed_tokens\.weight",
            ]
            self._keys_to_ignore_on_save = [
                r"encoder\.embed_tokens\.weight",
            ]
        assert save_dir is not None
        model_name = pretrained_version.split('/')[-1]
        # path_to_encoder_decoder_init = os.path.join(save_dir, f'{model_name}_encoder_decoder_init.onnx')
        # self.encoder_decoder_init_sess = create_ort_session(path_to_encoder_decoder_init,
        #         #                                                     use_gpu=torch.cuda.is_available())

        path_to_encoder = os.path.join(save_dir, f'{model_name}_encoder.onnx')
        path_to_decoder_init = os.path.join(save_dir, f'{model_name}_decoder_init.onnx')
        path_to_decoder = os.path.join(save_dir, f'{model_name}_decoder.onnx')

        provider_options = {
            'TensorrtExecutionProvider': {
                'device_id': 0,
                'trt_max_workspace_size': 2147483648,
                'trt_fp16_enable': True,
            }
        }
        encoder_sess = create_onnxruntime_session(path_to_encoder, use_gpu=torch.cuda.is_available(), provider=provider,
                                                  provider_options=provider_options)
        decoder_init_sess = create_onnxruntime_session(path_to_decoder_init, use_gpu=torch.cuda.is_available(),
                                                       provider=provider, provider_options=provider_options)
        decoder_sess = create_onnxruntime_session(path_to_decoder, use_gpu=torch.cuda.is_available(), provider=provider,
                                                  provider_options=provider_options)

        self.encoder = T5Encoder(encoder_sess, config)
        self.decoder = T5Decoder(decoder_sess, config)
        self.decoder_init = T5DecoderInit(decoder_init_sess, config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids, attention_mask)

        encoder_hidden_states = encoder_outputs[0]

        if past_key_values is not None:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

        if past_key_values is None:
            # runs only for the first time:
            logits, past_key_values = self.decoder_init(decoder_input_ids, attention_mask, encoder_hidden_states)

        else:
            logits, past_key_values = self.decoder(decoder_input_ids,
                                                   attention_mask,
                                                   encoder_hidden_states,
                                                   past_key_values)

        return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import timeit
    from ...benchmark_helper import get_latency_result

    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    t_input = "translate English to French: The universe is a dark forest."
    token = tokenizer(t_input, return_tensors='pt')
    input_ids = token['input_ids'].to(device)
    attention_mask = token['attention_mask'].to(device)

    torch_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    torch_model = torch_model.to(device)

    tokens = torch_model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  num_beams=2)

    torch_output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
    ic(t_input)
    ic(torch_output)

    torch_runtimes = timeit.repeat(lambda: torch_model.generate(input_ids=input_ids,
                                                                attention_mask=attention_mask,
                                                                num_beams=1), repeat=50, number=1)
    ic(get_latency_result(torch_runtimes, 1))

    onnx_model = OnnxT5('t5-small', save_dir='./onnx_models')
    onnx_model = onnx_model.to(device)

    tokens = onnx_model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 num_beams=2)
    onnx_output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
    ic(onnx_output)

    onnx_runtimes = timeit.repeat(lambda: onnx_model.generate(input_ids=input_ids,
                                                              attention_mask=attention_mask,
                                                              num_beams=1), repeat=50, number=1)
    ic(get_latency_result(onnx_runtimes, 1))

    # tensorrt_model = OnnxT5('t5-small', save_dir='./onnx_models', provider='tensorrt')
    # tensorrt_model = tensorrt_model.to(device)
    #
    # tokens = tensorrt_model.generate(input_ids=input_ids,
    #                                  attention_mask=attention_mask,
    #                                  num_beams=2)
    # tensorrt_output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
    # ic(tensorrt_output)
    #
    # tensorrt_runtimes = timeit.repeat(lambda: tensorrt_model.generate(input_ids=input_ids,
    #                                                                   attention_mask=attention_mask,
    #                                                                   num_beams=1), repeat=50, number=1)
    # ic(get_latency_result(tensorrt_runtimes, 1))
