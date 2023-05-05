import os
import importlib
from typing import Dict
from functools import partial
from xdpx.options import Options
from xdpx.utils import register

class Tokenizer:
    @classmethod
    def register(cls, options):
        pass
    
    def __init__(self, args):
        self.args = args
    
    def encode(self, x: str) -> list:
        raise NotImplementedError

    def decode(self, x: list) -> str:
        raise NotImplementedError
    
    def _autoset_meta(self):
        return {}

    def meta(self):
        options = Options()
        self.__class__.register(options)
        return {**{name: getattr(self.args, name) for name in options.keys()}, **self._autoset_meta()}


tokenizers: Dict[str, Tokenizer] = {}
register = partial(register, registry=tokenizers)


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('.' + module_name, __name__)
