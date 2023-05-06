import csv
from xdpx.options import Argument
from . import Parser, register


@register('csv')
class CSVParser(Parser):
    @staticmethod
    def register(options):
        options.register(
            Argument('delimiter', default='\t'),
            Argument('quotechar', default=None),
            Argument('start_line', default=0, doc='0 means without headers'),
        )

    def parse_line(self, line):
        return next(csv.reader([line.replace('\0', '').strip()], delimiter=self.args.delimiter,
                               quotechar=self.args.quotechar,
                               quoting=csv.QUOTE_NONE if self.args.quotechar is None else 0))
