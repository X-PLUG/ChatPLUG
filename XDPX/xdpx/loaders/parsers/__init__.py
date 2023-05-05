import os
import importlib
from typing import List
from tqdm import tqdm
from functools import partial
from xdpx.utils import register, io
from xdpx.options import Argument


parsers = {}
register = partial(register, registry=parsers)


@register('none')
class Parser:
    @staticmethod
    def register(options):
        pass

    def __init__(self, args):
        self.args = args

    def parse_line(self, line: str) -> List[str]:
        return [line.strip()]

    def open_file(self, path):
        """open a line-based file"""
        f = io.open(path)
        if hasattr(self.args, 'start_line'):
            for _ in range(int(self.args.start_line)):
                f.readline()  # use next(f) will cause OSError: telling position disabled by next() call
        start = f.tell()
        num_sections = len(self.parse_line(next(f)))
        f.seek(start)
        total = 0
        for _ in f:
            total += 1
        f.seek(start)
        return f, total, num_sections

    def close_file(self, f):
        f.close()


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('.' + module_name, __name__)
