import os
import random
import importlib
from typing import Dict
from functools import partial
from xdpx.utils import register


class Tuner:
    @staticmethod
    def register(options):
        pass

    def __init__(self, args, archive, space):
        self.args = args
        self.archive = archive
        self.space = space
        self.random = random.Random(args.seed)
    
    def suggest(self, config: dict) -> dict:
        """this operation is in-place"""
        raise NotImplementedError

    def update(self) -> dict:
        pass


class SpaceExhausted(Exception):
    ...


tuners: Dict[str, Tuner.__class__] = {}
register = partial(register, registry=tuners)


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('.' + module_name, __name__)
