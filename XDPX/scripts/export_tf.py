import os
import sys
import shutil
from typing import Dict, Any, Union
from xdpx.options import Options, Argument, Arg
from xdpx.utils import io, cache_file, parse_model_path
from xdpx.bootstrap import bootstrap
from xdpx.tasks import tasks


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('save_dir', required=True, validate=lambda val: io.exists(os.path.join(val, 'args.py'))),
        Argument('checkpoint', doc='Full path is needed. If not provided, use the best checkpoint in save_dir',
                 post_process=parse_model_path, type=str, default='<best>',
                 validate=lambda val: io.exists(val)),
        Argument('extra_config', type=Union[Dict[str, Any], str]),
        Argument('out_dir', required=True),
        Argument(
            'export_format', default='savedmodel', validate=lambda value: value in ('savedmodel', 'checkpoint'),
            children={
                lambda value: value == 'checkpoint': [
                    Argument('out_name', default='bert_model.ckpt'),
                ],
                lambda value: value == 'savedmodel': [
                    Argument('signature_def_key', default='serving_default'),
                    Argument('init_tables', default=True),
                    Argument('fix_len', default=False, doc='whether to fix input length'),
                    Argument('check_outputs', default=False),
                ],
            }
        ),
        Argument('strict_size', default=True),
    )
    bootstrap(options, main, __file__, argv)


def main(cli_args: Arg):
    import tensorflow as tf
    print('\n| Use tf version:', tf.__version__)

    with io.open(os.path.join(cli_args.save_dir, 'args.py')) as f:
        args = Options.parse_tree(eval(f.read()))
        if cli_args.extra_config is not None:
            if isinstance(cli_args.extra_config, dict):
                args = args.change(**cli_args.extra_config)
            else:
                args = args.change(**Options.load_hjson(cli_args.extra_config))
    args.strict_size = True
    args.__cmd__ = cli_args.__cmd__
    if cli_args.export_format == 'savedmodel':
        args.init_tables = cli_args.init_tables
        args.fix_len = cli_args.fix_len

    task = tasks[args.task](args)
    model = task.build_model(args)
    loss = task.build_loss(args)

    # load checkpoint
    model.load(cli_args.checkpoint)

    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        predict_signature = model.load_into_tf(sess, strict=cli_args.strict_size)

        if cli_args.export_format == 'checkpoint':
            io.makedirs(cli_args.out_dir)
            saver = tf.train.Saver()
            cache_outdir = cache_file(cli_args.out_dir, dry=True)
            if os.path.exists(cache_outdir):
                shutil.rmtree(cache_outdir)
            save_path = os.path.join(cache_outdir, cli_args.out_name)
            saver.save(sess, save_path)
            io.copytree(cache_outdir, cli_args.out_dir)
        elif cli_args.export_format == 'savedmodel':
            cache_outdir = cache_file(cli_args.out_dir, dry=True)
            if os.path.exists(cache_outdir):
                shutil.rmtree(cache_outdir)
            builder = tf.saved_model.builder.SavedModelBuilder(cache_outdir)
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op') if cli_args.init_tables else None
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    cli_args.signature_def_key:
                        tf.saved_model.signature_def_utils.predict_signature_def(**predict_signature)
                },
                legacy_init_op=legacy_init_op,
            )
            builder.save()
            io.copytree(cache_outdir, cli_args.out_dir)

            assets_dir = os.path.join(cli_args.out_dir, 'assets')
            io.makedirs(assets_dir, exist_ok=True)
            root = args.save_dir if io.exists(os.path.join(args.save_dir, 'vocab.txt')) else args.data_dir
            io.copy(os.path.join(root, 'vocab.txt'),
                    os.path.join(assets_dir, 'vocab.txt'))
            io.copy(os.path.join(root, 'target_map.txt'),
                    os.path.join(assets_dir, 'labels.txt'))
        else:
            raise NotImplementedError

        bert_config_file = os.path.join(cli_args.save_dir, 'config.json')
        if io.exists(bert_config_file):
            io.copy(bert_config_file, os.path.join(cli_args.out_dir, 'config.json'))

    if cli_args.export_format == 'savedmodel' and cli_args.check_outputs:
        from tensorflow.contrib import predictor
        import torch
        import torch.nn.functional as F
        cwd = os.getcwd()
        io.copytree(cli_args.out_dir, cache_outdir)
        os.chdir(cache_outdir)
        predict_fn = predictor.from_saved_model('.', signature_def_key=cli_args.signature_def_key)
        os.chdir(cwd)

        # pytorch forward
        model.eval()
        inputs = model.dummy_inputs
        with torch.no_grad():
            logits = model(*inputs)
            if isinstance(logits, tuple):
                logits = logits[-1]
            prob = loss.get_prob(logits)

        # tf_forward
        feed_dict = model.dummy_tf_inputs(inputs)
        tf_outputs = predict_fn(feed_dict)
        ref_prob = torch.from_numpy(tf_outputs['prob'])

        if not torch.allclose(prob, ref_prob, atol=1e-5, rtol=1e-3):
            raise RuntimeError('prediction results differ between the original one and the exported SavedModel.')
        print('Predictions from TF & Pytorch match.')


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cli_main()
