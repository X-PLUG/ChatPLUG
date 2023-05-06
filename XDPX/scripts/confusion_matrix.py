import sys
import csv
import math
import pandas as pd
from xdpx.options import Arg, Options, Argument
from xdpx.bootstrap import bootstrap
from xdpx.utils import io
from sklearn.metrics import confusion_matrix

"""
export confusion matrix (in table files)
"""


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('predict_file', required=True, validate=lambda val: io.exists(val)),
        Argument('target_map_file', required=True, validate=lambda val: io.exists(val)),
        Argument('out_file', type=str),
        Argument('target_col_name', default='target'),
        Argument('pred_col_name', default='pred'),

    )
    bootstrap(options, main, __file__, argv)


def main(cli_args: Arg):
    with io.open(cli_args.predict_file) as f:
        data = pd.read_csv(f, sep='\t', header=0, quoting=csv.QUOTE_NONE)
    with io.open(cli_args.target_map_file) as f:
        targets = [line.strip().lower() for line in f]
    y_true = [x.lower() for x in data[cli_args.target_col_name]]
    y_pred = [x.lower() for x in data[cli_args.pred_col_name].tolist()]

    cm = confusion_matrix(y_true, y_pred, labels=targets)
    cm = pd.DataFrame(cm, index=targets, columns=targets)
    cm.replace(0, math.nan, inplace=True)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    with io.open(cli_args.out_file, 'w') as f:
        cm.to_csv(f)


if __name__ == "__main__":
    cli_main()
