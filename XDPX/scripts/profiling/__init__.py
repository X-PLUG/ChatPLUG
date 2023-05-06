import os
import importlib
from typing import Dict, Any
from functools import partial
from xdpx.utils import register
from xdpx.options import Options, Argument
from xdpx.processors import processors
from xdpx.models import Model, models


configs = {}
register = partial(register, registry=configs)


class Profile:
    def build_module(self, **kwargs) -> Model:
        options = build_basic_options()
        config = self.model_config()
        config.update(dict(**kwargs))
        args = options.parse_dict(config)
        module = models[args.model].build(args)
        return module

    def model_config(self):
        raise NotImplementedError

    def create_fake_data(self) -> Dict[str, Any]:
        raise NotImplementedError


def build_basic_options():
    options = Options()
    options.register(
        Argument(
            'processor', required=True, 
            validate=lambda value: value in processors.keys(), 
            register=lambda value: processors[value].register
        ),
        domain='processor'
    )
    options.register(
        Argument('model', required=True, validate=lambda value: value in models,
            register=lambda value: Model.build_model_class(value).register), 
        domain='model',
    )
    return options


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('.' + module_name, __name__)
