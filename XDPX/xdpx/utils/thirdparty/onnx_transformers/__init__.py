# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import xdpx.utils.thirdparty.onnx_transformers.models.gpt2.convert_to_onnx

# added for backward compatible
import xdpx.utils.thirdparty.onnx_transformers.models.gpt2.gpt2_helper

from typing import Dict, Tuple

import numpy as np
import torch
from onnxruntime import InferenceSession, IOBinding

# https://github.com/pytorch/pytorch/blob/ac79c874cefee2f8bc1605eed9a924d80c0b3542/torch/testing/_internal/common_utils.py#L349
numpy_to_torch_dtype_dict = {
    bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

torch_to_numpy_dtype_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}


def gess_output_shape(inputs: Dict[str, torch.Tensor], model_onnx: InferenceSession) -> Dict[str, Tuple[int]]:
    """
    Try to guess output tensor shape from input tensors and axis names saved in ONNX model.
    Can only work if all output dim are fixed or linked to input axis.
    :param inputs: input tensors
    :param model_onnx: ONNX model
    :return: a dict {axis name: nb dim}
    """
    axis: Dict[str, int] = dict()
    for input_onnx in model_onnx.get_inputs():
        tensor = inputs[input_onnx.name]
        axis.update({axis_name: shape for shape, axis_name in zip(tensor.shape, input_onnx.shape)})
    shapes = dict()
    for output_onnx in model_onnx.get_outputs():
        output_shape = list()
        for shape in output_onnx.shape:  # type: Union[int, str]
            if isinstance(shape, str):
                if shape in axis:
                    shape = axis[shape]
                elif shape.isdigit():
                    shape = int(shape)
                else:
                    if '+' in shape:  ## hack support dynamic 'past_decode_sequence_length + 1'
                        a, b = [t.strip() for t in shape.split('+')]
                        shape = axis[a] + int(b)
                    else:
                        raise Exception(output_onnx)

            output_shape.append(shape)
        shapes[output_onnx.name] = tuple(output_shape)
    return shapes


def inference_onnx_binding(
        model_onnx: InferenceSession, inputs: Dict[str, torch.Tensor], device: str, device_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Performs inference on ONNX Runtime in an optimized way.
    In particular, avoid some tensor copy from GPU to host by using Torch tensors directly.
    :param model_onnx: ONNX model
    :param inputs: input torch tensor
    :param device: where to run the inference. One of [cpu, cuda]
    :param device_id: ID of the device where to run the inference, to be used when there are multiple GPUs, etc.
    :return: a dict {axis name: output tensor}
    """
    assert device in ["cpu", "cuda"]

    assert len(inputs) == len(model_onnx.get_inputs())
    binding: IOBinding = model_onnx.io_binding()
    for input_onnx in model_onnx.get_inputs():
        tensor: torch.Tensor = inputs[input_onnx.name]
        tensor = tensor.contiguous()
        if tensor.dtype in [torch.int64, torch.long]:
            # int32 mandatory as input of bindings, int64 not supported, or compile onnxruntime from source
            tensor = tensor.type(dtype=torch.int32).to(device)
        binding.bind_input(
            name=input_onnx.name,
            device_type=device,
            device_id=device_id,
            element_type=torch_to_numpy_dtype_dict[tensor.dtype],
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )
        inputs[input_onnx.name] = tensor
    outputs = dict()
    output_shapes = gess_output_shape(inputs=inputs, model_onnx=model_onnx)

    for axis_name, shape in output_shapes.items():
        tensor = torch.empty(shape, dtype=torch.float32, device=device).contiguous()
        outputs[axis_name] = tensor
        binding.bind_output(
            name=axis_name,
            device_type=device,
            device_id=device_id,
            element_type=np.float32,  # hard coded output type
            shape=tuple(shape),
            buffer_ptr=tensor.data_ptr(),
        )
    model_onnx.run_with_iobinding(binding)
    return outputs
