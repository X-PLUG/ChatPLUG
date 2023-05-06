# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return layer_norm_factory(export)(normalized_shape, eps, elementwise_affine)


def layer_norm_factory(export=False):
    # TODO: use apex FusedLayerNorm when available which is 3x faster. Currently amp O1 level
    #  will not correctly cast its inputs and can only accept fp32 inputs
    return torch.nn.LayerNorm
