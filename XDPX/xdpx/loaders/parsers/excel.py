from typing import List
import pandas as pd
from xdpx.options import Argument
from xdpx.utils import io
from . import Parser, register


@register('excel')
class ExcelParser(Parser):
    @staticmethod
    def register(options):
        options.register(
            Argument('sheet_name', required=True),
            Argument('fields', type=List[str], doc='if None, assume no header line and use all columns'),
        )

    def parse_line(self, obj):
        if self.args.fields:
            return [obj[field] for field in self.args.fields if field in obj]
        else:
            return list(obj.values())

    def open_file(self, path):
        with io.open(path, 'rb') as excel_f:
            f = pd.read_excel(excel_f, sheet_name=self.args.sheet_name, header=0 if self.args.fields else None)
        f = [row.to_dict() for _, row in f.iterrows()]
        total = len(f)
        num_sections = len(self.parse_line(f[0]))
        return f, total, num_sections

    def close_file(self, f):
        ...
