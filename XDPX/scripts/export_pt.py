import sys
import os
import json
import torch
from typing import Dict, Any
from xdpx.options import Arg, Options, Argument
from xdpx.models import models
from xdpx.bootstrap import bootstrap
from xdpx.utils import io

"""
Export pytorch model from tf checkpoints or huggingface pytorch checkpoints.
"""


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('tf_path', required=True),
        Argument('bert_config', required=True),
        Argument('out_dir', required=True),
        Argument('from_tf', default=True, children=[
            Argument('tf_format', default='checkpoint', validate=lambda val: val in ('checkpoint', 'savedmodel'),
                     children={lambda val: val == 'savedmodel': [
                         Argument('signature_def_key', default='serving_default'),
                     ]}),
        ]),
        Argument('out_name', default='bert_model.pt'),
        Argument('model_class', default='bert_pretrain', validate=lambda value: value in models, children=[
            Argument('extra_configs', default={}, type=Dict[str, Any]),
        ]),
        # TODO: parse this automatically??
        Argument('num_classes', default=2, doc='number of classes in the seq-cls task'),
        Argument('strict_size', default=True),
    )
    bootstrap(options, main, __file__, argv)


def main(cli_args):
    args = Arg()
    args.strict_size = True
    args.__cmd__ = cli_args.__cmd__
    args.extra_config = {}
    with io.open(cli_args.bert_config) as f:
        config = json.load(f)
    args.num_classes = cli_args.num_classes
    args.tf_format = getattr(cli_args, 'tf_format', None)
    args.signature_def_key = getattr(cli_args, 'signature_def_key', None)
    args.strict_size = cli_args.strict_size
    args.hidden_dropout_prob = args.attention_probs_dropout_prob = None
    for key, val in config.items():
        setattr(args, key, val)

    for key, val in cli_args.extra_configs.items():
        setattr(args, key, val)
    os.makedirs(cli_args.out_dir, exist_ok=True)

    model = models[cli_args.model_class](args)
    model.load(cli_args.tf_path, from_tf=cli_args.from_tf)
    with io.open(os.path.join(cli_args.out_dir, cli_args.out_name), 'wb') as f:
        torch.save(model.state_dict(), f)
    with io.open(os.path.join(cli_args.out_dir, 'config.json'), 'w') as f:
        json.dump(model.get_bert_config(), f)


if __name__ == "__main__":
    cli_main()
