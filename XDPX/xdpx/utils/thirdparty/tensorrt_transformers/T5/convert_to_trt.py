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

# huggingface
from transformers import (
    MT5ForConditionalGeneration,
    T5ForConditionalGeneration
)

from .T5ModelConfig import T5ModelTRTConfig
from ..NNDF.networks import NetworkMetadata, Precision
from .T5ModelConfig import T5ModelTRTConfig, T5Metadata
from .export import T5EncoderTorchFile, T5DecoderTorchFile
from .export import T5DecoderONNXFile, T5EncoderONNXFile
from polygraphy.backend.trt import Profile

logger = logging.getLogger("")

T5_VARIENTS = ['t5-small', 't5-base', 't5-large', 'mt5-small', 'mt5-base', 'mt5-large']


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name_or_path",
        required=False,
        default='t5-small',
        type=str,
        help="Model path, or pretrained model name in the list: " + ", ".join(T5ModelTRTConfig.TARGET_MODELS),
    )

    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_models"),
        help="Directory to cache pre-trained models",
    )

    parser.add_argument(
        "--t5_varient",
        required=False,
        default='t5-small',
        type=str,
        help="Model path, or pretrained model name in the list: " + ", ".join(T5_VARIENTS),
    )

    parser.add_argument(
        "--output",
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
    T5_VARIENT = args.t5_varient
    cache_dir = args.cache_dir
    output_dir = args.output if not args.output.endswith(".engine") else os.path.dirname(args.output)

    ## load torch model
    model_name_or_path = args.model_name_or_path
    if model_name_or_path.startswith('oss://'):
        from xdpx.utils import io
        from xdpx.options import Options
        from transformers import AutoConfig
        from xdpx.utils import parse_model_path
        if io.isfile(model_name_or_path):
            save_dir = os.path.dirname(model_name_or_path)
            checkpoint = model_name_or_path
        else:
            save_dir = model_name_or_path
            checkpoint = parse_model_path(os.path.join(save_dir, '<last>'))
        if not io.exists(os.path.join(save_dir, 'args.py')):
            with io.open(checkpoint, 'rb') as f:  # checkpoint is self-contained t5 model
                model_name = model_name_or_path.split('/')[-1]
                t5_model = torch.load(f, map_location=torch.device(
                    'cuda') if torch.cuda.is_available() else torch.device('cpu'))
        else:
            with io.open(os.path.join(save_dir, 'args.py')) as f:  # checkpoint is state_dict
                xargs = Options.parse_tree(eval(f.read()))
                config = AutoConfig.from_pretrained(xargs.auto_model)
                t5_model = T5ForConditionalGeneration(config)
                model_name = xargs.auto_model.split('/')[-1]
                with io.open(checkpoint, 'rb') as f:
                    state_dict = torch.load(f, map_location=torch.device(
                        'cuda') if torch.cuda.is_available() else torch.device('cpu'))

                    for old_name in list(state_dict.keys()):
                        if old_name.startswith('backbone.encoder.encoder'):
                            # wrap encoder name map
                            new_name = old_name.replace('backbone.encoder.encoder', 'encoder').replace(
                                'module.layer',
                                'layer')
                            state_dict[new_name] = state_dict.pop(old_name)
                        elif old_name.startswith('backbone'):
                            new_name = old_name[old_name.index('.') + 1:]  # remove backbone. prefix
                            state_dict[new_name] = state_dict.pop(old_name)

                    t5_model.load_state_dict(state_dict, strict=True)

    elif 'mt5' in model_name_or_path:
        model_name = model_name_or_path.split('/')[-1]
        t5_model = MT5ForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    else:
        model_name = model_name_or_path.split('/')[-1]
        t5_model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir=cache_dir)

    ## convert to onnx
    onnx_model_path = './models/{}/ONNX'.format(model_name)
    Path(onnx_model_path).mkdir(parents=True, exist_ok=True)

    metadata = NetworkMetadata(variant=T5_VARIENT, precision=Precision(fp16=True),
                               other=T5Metadata(kv_cache=False))

    encoder_onnx_model_fpath = model_name + "-encoder.onnx"
    decoder_onnx_model_fpath = model_name + "-decoder-with-lm-head.onnx"

    t5_encoder = T5EncoderTorchFile(t5_model.to('cpu'), metadata)
    t5_decoder = T5DecoderTorchFile(t5_model.to('cpu'), metadata)

    onnx_t5_encoder = t5_encoder.as_onnx_model(
        os.path.join(onnx_model_path, encoder_onnx_model_fpath), force_overwrite=True
    )
    onnx_t5_decoder = t5_decoder.as_onnx_model(
        os.path.join(onnx_model_path, decoder_onnx_model_fpath), force_overwrite=True
    )

    logger.info(f"Convert ONNX Done! Path: {onnx_model_path}")

    ## convert to trt
    tensorrt_model_path = './models/{}/tensorrt'.format(model_name)
    Path(tensorrt_model_path).mkdir(parents=True, exist_ok=True)
    # Decoder optimization profiles
    batch_size = 1
    max_sequence_length = T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[T5_VARIENT]
    decoder_profile = Profile()
    decoder_profile.add(
        "input_ids",
        min=(batch_size, 1),
        opt=(batch_size, max_sequence_length // 2),
        max=(batch_size, max_sequence_length),
    )
    decoder_profile.add(
        "encoder_hidden_states",
        min=(batch_size, 1, max_sequence_length),
        opt=(batch_size, max_sequence_length // 2, max_sequence_length),
        max=(batch_size, max_sequence_length, max_sequence_length),
    )

    # Encoder optimization profiles
    encoder_profile = Profile()
    encoder_profile.add(
        "input_ids",
        min=(batch_size, 1),
        opt=(batch_size, max_sequence_length // 2),
        max=(batch_size, max_sequence_length),
    )

    t5_trt_encoder_engine = T5EncoderONNXFile(
        os.path.join(onnx_model_path, encoder_onnx_model_fpath), metadata
    ).as_trt_engine(os.path.join(tensorrt_model_path, encoder_onnx_model_fpath) + ".engine", profiles=[encoder_profile])

    t5_trt_decoder_engine = T5DecoderONNXFile(
        os.path.join(onnx_model_path, decoder_onnx_model_fpath), metadata
    ).as_trt_engine(os.path.join(tensorrt_model_path, decoder_onnx_model_fpath) + ".engine", profiles=[decoder_profile])

    logger.info(f"Convert TRT Done! Path: {tensorrt_model_path}")


if __name__ == "__main__":
    main()
