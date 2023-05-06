import os
import torch
from typing import Dict
from xdpx.utils import io
from xdpx.options import Argument
from . import register


@register('torchscript')
class TorchScriptModel:
    @staticmethod
    def register(options):
        options.register(
            Argument('model_path', required=True, validate=lambda val: io.exists(val)),
        )

    def __init__(self, args):
        self.model = torch.jit.load(args.model_path)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@register('savedmodel')
class TFSavedModel:
    @staticmethod
    def register(options):
        options.register(
            Argument('model_path', required=True, validate=lambda val: io.exists(os.path.join(val, 'saved_model.pb'))),
            Argument('signature_def_key', default='serving_default'),
            Argument('input_flags', default={}, type=Dict[str, any]),
            Argument('input_name_map', default={}, type=Dict[str, str]),
            Argument('output_name', required=True, doc='name for logits'),
        )

    def __init__(self, args):
        from xdpx.utils.tf_utils import load_tf_savedmodel
        self.predict_fn = load_tf_savedmodel(args.model_path, args.signature_def_key)

    def forward(self, **kwargs):
        for key, val in self.args.input_name_map.items():
            kwargs[val] = kwargs.pop(key)
        return self.predict_fn(**kwargs, **self.args.input_flags)[self.args.output_name]

