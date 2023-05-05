from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys, re, argparse, json
import numpy as np
from collections import defaultdict
from scripts.utils.bert_fusion.modify_bert import *

tf.logging.set_verbosity(tf.logging.ERROR)

# calculate padding size for x
def cal_padding(x):
    ld = x.shape[-1]
    paddings = [[0, 0], [0, ((ld + 127)//128)*128+16 - ld]]
    return paddings

def pad_graph_def(graph_def):
    new_nodes = defaultdict(dict)
    qkv = defaultdict(dict)
    for node in graph_def.node:
        result = re.search(r"^(?P<prefix>.*)bert/encoder/layer_(?P<layer_name>[0-9]+)/(?P<kernel_name>[a-zA-Z/]+)$", node.name)
        if result is not None:
            layer_name = result.group("layer_name")
            kernel_name = result.group("kernel_name")
            if kernel_name in ["attention/output/dense/kernel", "intermediate/dense/kernel", "output/dense/kernel"]:
                dtype = node.attr["dtype"]
                value = get_node_value(graph_def, node.name)
                paddings = cal_padding(value)
                value = np.pad(value, paddings, 'constant', constant_values=(0))
                new_nodes[layer_name][kernel_name] = create_new_const_node(dtype, value, node.name + "_padded")
            elif kernel_name in ["attention/self/key/kernel", "attention/self/query/kernel", "attention/self/value/kernel", "attention/self/key/bias", "attention/self/query/bias", "attention/self/value/bias"]:
                prefix = result.group("prefix")
                qkv[layer_name][kernel_name] = get_node_value(graph_def, node.name)

    # Merge q, k, v
    for layer_name in qkv:
        qkv_k = np.concatenate((qkv[layer_name]["attention/self/query/kernel"], qkv[layer_name]["attention/self/key/kernel"], qkv[layer_name]["attention/self/value/kernel"]), axis = 1)
        paddings = cal_padding(qkv_k)
        qkv_k = np.pad(qkv_k, paddings, 'constant', constant_values=(0))
        new_nodes[layer_name]['attention/self/qkv/kernel'] = create_new_const_node(dtype, qkv_k, prefix+"bert/encoder/layer_"+layer_name+"/attention/self/qkv/kernel")
        qkv_b = np.concatenate((qkv[layer_name]["attention/self/query/bias"], qkv[layer_name]["attention/self/key/bias"], qkv[layer_name]["attention/self/value/bias"]), axis = 0)
        new_nodes[layer_name]['attention/self/qkv/bias'] = create_new_const_node(dtype, qkv_b, prefix+"bert/encoder/layer_"+layer_name+"/attention/self/qkv/bias")

    for layer_name in new_nodes:
        graph_def.node.extend(new_nodes[layer_name].values())
    return prefix

def create_bert_node(layer_name, node_name, input_node, attention_mask, num_attention_heads):
    bert_node = create_node('BertAttention', node_name, [input_node,
                                                         attention_mask,
                                                         layer_name+'/attention/self/qkv/kernel',
                                                         layer_name+'/attention/self/qkv/bias',
                                                         layer_name+'/attention/output/dense/kernel_padded',
                                                         layer_name+'/attention/output/dense/bias',
                                                         layer_name+'/attention/output/LayerNorm/beta',
                                                         layer_name+'/attention/output/LayerNorm/gamma',
                                                         layer_name+'/intermediate/dense/kernel_padded',
                                                         layer_name+'/intermediate/dense/bias',
                                                         layer_name+'/output/dense/kernel_padded',
                                                         layer_name+'/output/dense/bias',
                                                         layer_name+'/output/LayerNorm/beta',
                                                         layer_name+'/output/LayerNorm/gamma'])
    set_attr_dtype(bert_node, "T", tf.float32)
    bert_node.attr["num_attention_heads"].CopyFrom(attr_value_pb2.AttrValue(i=num_attention_heads))
    return bert_node

def create_bert_nodes(graph_def, num_hidden_layers = 12, num_attention_heads = 12):
    prefix = pad_graph_def(graph_def)
    print ("weights prefix:", prefix)
    # find attention_mask
    attention_mask, bert_input, bert_output = (None, None, None)
    for node in graph_def.node:
        print(node.name)
        if node.name.endswith('bert/encoder/layer_0/attention/self/mul') and node.op == 'Mul':
            print ('attenion_mask:', node.name)
            attention_mask = node.name
        elif node.name.endswith('bert/encoder/layer_0/attention/self/mul_1') and node.op == 'Mul':
            # 有些版本导出第一个node就是mul_1，有些版本导出第一个node是mul，避免覆盖
            if attention_mask is None:
                print ('attenion_mask:', node.name)
                attention_mask = node.name
        elif 'bert/embeddings/LayerNorm/batchnorm/add_1' in node.name and node.op == 'Add':
            print ('input_node:', node.name)
            bert_input = node.name
        elif 'bert/encoder/layer_' + str(num_hidden_layers - 1) + '/output/LayerNorm/batchnorm/add_1' in node.name and node.op == 'Add':
            print ('output_node:', node.name)
            bert_output = node.name

    assert bert_input, "Cannot find bert attention input node"
    assert attention_mask, "Cannot find bert attention mask node"
    assert bert_output, "Cannot find bert attention output node"

    bert_nodes = []
    input_node = bert_input
    is_first = 1
    layer_ids = range(num_hidden_layers)
    if num_hidden_layers == 4:
        layer_ids = range(4)

    for i in layer_ids:
        bert_node = create_bert_node(prefix+'bert/encoder/layer_'+str(i),
                                     'BertAttention'+str(i),
                                     input_node,
                                     attention_mask,
                                     num_attention_heads)
        bert_nodes.append(bert_node)
        input_node = 'BertAttention'+str(i)
    graph_def.node.extend(bert_nodes)

    for node in graph_def.node:
        if not node.input:
            continue
        for i in range(len(node.input)):
            if str(node.input[i]) == bert_output:
                node.input[i] = bert_nodes[-1].name
                print('**** Modified the input node of %s' % node.name)
                break
    return graph_def


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model = parser.add_mutually_exclusive_group(required=True)
    # provid frozen model pb file or export dir for saved_model
    model.add_argument("--frozen_pb_file", type=str, help="frozen model pb file")
    model.add_argument("--saved_model_dir", type=str, help="export dir for saved_model")
    parser.add_argument("--output_tensors", nargs='+', type=str, help="ouput tensors", required=True)
    parser.add_argument("--num_hidden_layers", default=12, type=int, help="number of attention layers")
    parser.add_argument("--num_attention_heads", default=12, type=int, help="number of attention heads")
    parser.add_argument("--bert_json", type=str, default='bert.json', help="name of the bert json file")

    args = parser.parse_args()
    if args.saved_model_dir is not None:
        print ("Read bert model from:", args.saved_model_dir, "with output tensor as", args.output_tensors)
        print ("freeze the saved_model as frozen_saved_model.pb")
        frozen_graph_maker(args.saved_model_dir, "frozen_saved_model.pb", args.output_tensors)
        frozen_pb_file = "frozen_saved_model.pb"
    else:
        print ("Read bert model from:", args.frozen_pb_file, "with output tensor as", args.output_tensors)
        frozen_pb_file = args.frozen_pb_file

    # load graph_def
    graph_def = load_graph_def(frozen_pb_file)

    if (os.path.isfile(args.bert_json)):
        print ("Read bert config from:", args.bert_json)
        with open(args.bert_json) as json_file:
            data = json.load(json_file)
            num_hidden_layers = data['num_hidden_layers']
            num_attention_heads = data['num_attention_heads']
    else:
        num_hidden_layers = args.num_hidden_layers
        num_attention_heads = args.num_attention_heads

    print ("Set number of attention layers as", num_hidden_layers, ",number of attention heads as", num_attention_heads)

    # add new bert nodes
    new_graph_def = create_bert_nodes(graph_def, num_hidden_layers, num_attention_heads)
    print ("freeze the new graph_def as fused_saved_model.pb")
    save_graph_def(new_graph_def, 'fused_saved_model.pb', args.output_tensors)