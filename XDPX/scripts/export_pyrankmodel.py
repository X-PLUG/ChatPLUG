import json
import os
import sys
import shutil
import configparser
import tarfile

from scripts.utils.bert_fusion.bert_fusion_helper import create_bert_nodes, \
    save_graph_def, frozen_graph_maker
from scripts.utils.bert_fusion.modify_bert import load_graph_def
from xdpx.utils import io, parse_model_path, download_from_url
from xdpx.bootstrap import bootstrap
from xdpx.options import Options, Argument
from xdpx.models.bert import BertForClassification

import tensorflow as tf
from scripts.export_tf import load_pt_into_tf

if tf.__version__ >= '2.0.0':
    import tensorflow.compat.v1 as tf


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('save_dir', required=True, validate=lambda val: io.exists(
            os.path.join(val, 'args.py'))),
        Argument('runtime_env', required=False, doc="Basic runtime environment",
                 default="http://algoop.oss-cn-hangzhou-zmf.aliyuncs.com/"
                         "pyrank-eas-service/release/"
                         "pyrank-tinybert-eas-env.tar.gz"),
        Argument('inject_env', required=False, default=False,
                 doc="Building environment with ENV."),
        Argument('checkpoint', doc='Full path is needed. If not provided, use the best checkpoint in save_dir',
                 post_process=parse_model_path, type=str, default='<best>',
                 validate=lambda val: io.exists(val)),
        Argument('out_dir', required=True),
        Argument('pyrank_config', required=False, type=str,
                 doc='default pyrank config file, support http link or local file'),
        Argument('output_eas_model', required=False, default=False,
                 doc="output the model can serving on eas",),
    )
    bootstrap(options, main, __file__, argv)


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz", compresslevel=5) as tar:
        tar.add(source_dir, arcname='')


def fusion(saved_model_dir, bert_json, output_tensors,
           frozen_pb_file, fused_pb_file):

    num_hidden_layers = bert_json['num_hidden_layers']
    num_attention_heads = bert_json['num_attention_heads']

    print("Set number of attention layers as", num_hidden_layers,
          ",number of attention heads as", num_attention_heads)

    print("Read bert model from:", saved_model_dir,
          "with output tensor as", output_tensors)
    print(f"freeze the saved_model as {frozen_pb_file}")
    # load graph_def
    frozen_graph_maker(saved_model_dir, frozen_pb_file, output_tensors)
    graph_def = load_graph_def(frozen_pb_file)
    # add new bert nodes
    new_graph_def = create_bert_nodes(graph_def, num_hidden_layers, num_attention_heads)
    print(f"freeze the new graph_def as {fused_pb_file}")
    save_graph_def(new_graph_def, fused_pb_file, output_tensors)


def check_tf_version():
    tf_version = tf.__version__
    print("\n| tensorflow version is ", tf_version)
    # choose library depend on version
    fileds = list(map(int, tf_version.split('.')))
    success = False
    if fileds[0] >= 2:
        success = True
    elif fileds[0] == 1:
        if fileds[1] >= 12:
            success = True
    if not success:
        print('| tf version neet >= 1.12')
        sys.exit(1)


def save_serving_config(model_dir, cli_args):
    # generate config
    source_file_dir = os.path.dirname(os.path.abspath(__file__))
    if cli_args.pyrank_config is not None:
        config_file_path = cli_args.pyrank_config
        if config_file_path.startswith('http'):
            config_file_path = download_from_url(config_file_path)
    else:
        config_file_path = os.path.join(source_file_dir,
                                        'utils/pyrank_res/basic_config.ini')
    config = configparser.ConfigParser()
    config.read(config_file_path)
    # last_dir = model_dir.strip('/').split('/')[-1]
    config['model']['model_path'] = \
        'resources/' + 'fused_saved_model.pb'
    config['model']['vocab_path'] = \
        'resources/' + 'assets/vocab.txt'
    config['model']['label_path'] = \
        'resources/' + 'assets/labels.txt'
    with open(os.path.join(model_dir, 'config.ini'), 'w') as f:
        config.write(f)


def main(cli_args):
    check_tf_version()
    with io.open(os.path.join(cli_args.save_dir, 'args.py')) as f:
        args = Options.parse_tree(eval(f.read()))
    args.strict_size = True
    model = BertForClassification(args)
    model.load(cli_args.checkpoint)

    bert_config_file = os.path.join(cli_args.save_dir, 'config.json')
    tf.reset_default_graph()
    if os.path.exists(cli_args.out_dir):
        shutil.rmtree(cli_args.out_dir)
    model_dir = os.path.join(cli_args.out_dir, "resources")
    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        load_pt_into_tf(sess, bert_config_file, model.state_dict())
        graph = sess.graph

        builder = tf.saved_model.builder.SavedModelBuilder(model_dir)

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs={
                        'input_ids': graph.get_tensor_by_name('input_ids:0'),
                        'input_mask': graph.get_tensor_by_name('input_mask:0'),
                        'segment_ids': graph.get_tensor_by_name('segment_ids:0'),
                    }, outputs={
                        'predictions': graph.get_tensor_by_name('predictions:0'),
                        'probs': graph.get_tensor_by_name('probs:0'),
                    }
                )}
        )
        builder.save()

        assets_dir = os.path.join(model_dir, 'assets')
        os.makedirs(assets_dir, exist_ok=True)
        root = args.save_dir if io.exists(os.path.join(args.save_dir, 'vocab.txt')) else args.data_dir
        io.copy(os.path.join(root, 'vocab.txt'),
                os.path.join(assets_dir, 'vocab.txt'))
        io.copy(os.path.join(root, 'target_map.txt'),
                os.path.join(assets_dir, 'labels.txt'))

        with io.open(bert_config_file, 'r') as f:
            bert_json = json.loads(f.read())
            output_nodes = ['predictions', 'probs']
            frozen_pb_file = os.path.join(model_dir, 'frozen_saved_model.pb')
            fused_pb_file = os.path.join(model_dir, 'fused_saved_model.pb')
            fusion(model_dir, bert_json, output_nodes,
                   frozen_pb_file, fused_pb_file)

        save_serving_config(cli_args.out_dir, cli_args)
        if cli_args.inject_env:
            print("building output model with runtime")
            runtime_env = download_from_url(cli_args.runtime_env)
            with tarfile.open(runtime_env) as f:
                f.extractall(cli_args.out_dir)

        # make eas serving model
        if cli_args.output_eas_model:
            eas_model_path = os.path.join(cli_args.out_dir,
                                          'serving_model.tar.gz')
            print("create eas serving path:", eas_model_path)
            make_tarfile(
                eas_model_path,
                cli_args.out_dir
            )

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    cli_main()
