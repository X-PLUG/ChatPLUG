import os
import sys
import torch
from xdpx.tasks import tasks
from xdpx.utils import io, parse_model_path, cache_file
from xdpx.bootstrap import bootstrap
from xdpx.options import Options, Argument


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('save_dir', required=True, validate=lambda val: io.exists(
            os.path.join(val, 'args.py'))),
        Argument('checkpoint', doc='Full path is needed. If not provided, use the best checkpoint in save_dir',
                 post_process=parse_model_path, type=str, default='<best>',
                 validate=lambda val: io.exists(val)),
        Argument('out_dir', required=True),
    )
    bootstrap(options, main, __file__, argv)


def main(cli_args):
    print(f'Using Python', '.'.join(map(str, sys.version_info)))
    print(f'Using PyTorch {torch.__version__}')
    with io.open(os.path.join(cli_args.save_dir, 'args.py')) as f:
        args = Options.parse_tree(eval(f.read()))
    args.strict_size = True
    task = tasks[args.task](args)
    model = task.build_model(args)
    model.load(cli_args.checkpoint)
    model.eval()

    if io.exists(cli_args.out_dir):
        io.rmtree(cli_args.out_dir)
    io.makedirs(cli_args.out_dir)

    inputs = model.dummy_inputs
    traced_model = torch.jit.trace(model, inputs)
    save_path = os.path.join(cli_args.out_dir, 'model.pth')
    cache_path = cache_file(save_path, dry=True)
    torch.jit.save(traced_model, cache_path)
    traced_model = torch.jit.load(cache_path)
    assert torch.allclose(model(*inputs), traced_model(*inputs))
    io.move(cache_path, save_path)
    root = args.save_dir if io.exists(os.path.join(args.save_dir, 'vocab.txt')) else args.data_dir
    io.copy(os.path.join(root, 'vocab.txt'),
            os.path.join(cli_args.out_dir, 'vocab.txt'))
    io.copy(os.path.join(root, 'target_map.txt'),
            os.path.join(cli_args.out_dir, 'labels.txt'))



if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    cli_main()
