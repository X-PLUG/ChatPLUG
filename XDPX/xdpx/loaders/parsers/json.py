import json
from typing import List
from xdpx.options import Argument
from xdpx.utils import io
from . import Parser, register


@register('jsonl')
class JSONLParser(Parser):
    """Each line is a json string"""
    @staticmethod
    def register(options):
        options.register(
            Argument('fields', required=True, type=List[str]),
        )

    def parse_line(self, line):
        data = json.loads(line)
        return [data[field] for field in self.args.fields if field in data]


@register('json')
class JSONParser(JSONLParser):
    def parse_line(self, obj):
        return [obj[field] for field in self.args.fields if field in obj]

    def open_file(self, path):
        with io.open(path) as json_f:
            f = json.load(json_f)
        total = len(f)
        num_sections = len((self.parse_line(f[0])))
        return f, total, num_sections

    def close_file(self, f):
        ...
