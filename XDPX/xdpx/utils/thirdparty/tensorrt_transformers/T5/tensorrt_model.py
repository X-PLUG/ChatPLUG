# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# example:
# python3 -m xdpx.utils.thirdparty.tensorrt_transformers.T5.convert_to_trt -m t5-small
# python3 -m xdpx.utils.thirdparty.tensorrt_transformers.T5.convert_to_trt -m checkpoint-44220.pt

import argparse
import copy
import logging
import os
import sys

import torch
import tensorrt as trt
from pathlib import Path

from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers import AutoConfig, MT5Config

# huggingface
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
)

from .trt import T5TRTEncoder, T5TRTDecoder, TRTHFRunner
from .T5ModelConfig import T5ModelTRTConfig, T5Metadata
from .export import T5DecoderTRTEngine, T5EncoderTRTEngine
from ..NNDF.networks import NetworkMetadata, Precision

logger = logging.getLogger("")

T5_VARIENTS = ['t5-small', 't5-base', 't5-large', 'mt5-small', 'mt5-base', 'mt5-large']


class T5Encoder(torch.nn.Module):
    def __init__(self, encoder_engine):
        super().__init__()
        self.encoder_engine = encoder_engine

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
        last_hidden_states = self.encoder_engine(input_ids)
        return BaseModelOutput(last_hidden_states)


class TrtT5(T5ForConditionalGeneration):
    """creates a T5 model using tensorrt engine"""

    def __init__(self, pretrained_version='google/mt5-base', save_dir=None, provider='cuda', config=None,
                 t5_variant='mt5-base'):
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

        # load TensorRT engine
        tfm_config = T5Config(
            use_cache=True,
            num_layers=T5ModelTRTConfig.NUMBER_OF_LAYERS[t5_variant],
        )
        metadata = NetworkMetadata(variant=t5_variant, precision=Precision(fp16=True), other=T5Metadata(kv_cache=False))

        t5_trt_encoder = T5TRTEncoder(f'{save_dir}/{t5_variant}-encoder.onnx.engine', metadata, tfm_config)
        t5_trt_decoder = T5TRTDecoder(f'{save_dir}/{t5_variant}-decoder-with-lm-head.onnx.engine', metadata, tfm_config)
        self.encoder = T5Encoder(t5_trt_encoder)
        self.decoder = t5_trt_decoder

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

        # TODO: not work yet
        # logits, past_key_values = self.decoder(decoder_input_ids,
        #                                        attention_mask,
        #                                        encoder_hidden_states,
        #                                        past_key_values)

        # return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--t5_varient",
        required=False,
        default='t5-small',
        type=str,
        help="Model path, or pretrained model name in the list: " + ", ".join(T5_VARIENTS),
    )

    parser.add_argument(
        "--save_dir",
        required=False,
        type=str,
        default=os.path.join(".", "trt_models"),
        help="Output directory",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    print(f"Arguments:{args}")


if __name__ == "__main__":
    main()
