# -*- coding:utf-8 -*-
import tensorflow as tf

if (tf.__version__) >= '2.0.0':
    import tensorflow.compat.v1 as tf

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
import sys, re
import numpy as np


def frozen_graph_maker(export_dir, output_graph, output_nodes):
    with tf.Session(graph=tf.Graph(),
                    config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING],
                            export_dir)
        # output_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            sess.graph_def,
            output_nodes,
            # The output node names are used to select the usefull nodes
            variable_names_blacklist=['global_step']
        )
    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())


def set_attr_shape(node, key, value):
    try:
        node.attr[key].CopyFrom(
            attr_value_pb2.AttrValue(
                shape=tensor_shape.as_shape(value).as_proto()))
    except KeyError:
        pass


def set_attr_tensor(node, key, value, dtype, shape=None):
    try:
        node.attr[key].CopyFrom(
            attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                value, dtype=dtype, shape=shape)))
    except KeyError:
        pass


def create_node(op, name, inputs):
    new_node = node_def_pb2.NodeDef()
    new_node.op = op
    new_node.name = name
    for input_node in inputs:
        new_node.input.extend([input_node])
    return new_node


def set_attr_dtype(node, key, value):
    try:
        node.attr[key].CopyFrom(
            attr_value_pb2.AttrValue(type=value.as_datatype_enum))
    except KeyError:
        pass


def create_constant_node(name, value, dtype, shape=None):
    node = create_node("Const", name, [])
    set_attr_dtype(node, "dtype", dtype)
    set_attr_tensor(node, "value", value, dtype, shape)
    return node


def create_placeholder(name):
    node = create_node('Placeholder', name, [])
    set_attr_dtype(node, "dtype", dtypes.int32)
    set_attr_shape(node, "shape", [None, 128])
    return node


def load_graph_def(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        try:
            graph_def.ParseFromString(f.read())
            return graph_def
        except DecodeError:
            sm = saved_model_pb2.SavedModel()
            with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
                sm.ParseFromString(compat.as_bytes(f.read()))
                return sm.meta_graphs[0].graph_def


def create_new_const_node(dtype, value, name):
    new_node = node_def_pb2.NodeDef()
    new_node.op = "Const"
    new_node.name = name
    new_node.attr["dtype"].CopyFrom(dtype)
    new_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            value, dtypes.float32, value.shape)))
    return new_node


def find_node_by_name(graph_def, name):
    for node in graph_def.node:
        if (node.name == name):
            return node


def get_node_value(graph_def, name):
    node = find_node_by_name(graph_def, name)
    if (node.op == 'Const'):
        return tensor_util.MakeNdarray(node.attr['value'].tensor)
    elif (node.op == 'Identity'):
        return get_node_value(graph_def, node.input[0])
    else:
        return None


def save_graph_def(graph_def, new_pb_file, output_nodes):
    with tf.Session() as sess:
        converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                           graph_def,
                                                                           output_nodes)
        tf.train.write_graph(converted_graph_def, './', new_pb_file,
                             as_text=False)
