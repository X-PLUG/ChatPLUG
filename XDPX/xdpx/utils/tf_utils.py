import os
import re
import torch
import numpy as np
from tqdm import tqdm
from xdpx.utils import io, cache_file


def import_tensorflow():
    try:
        import tensorflow as tf
        from packaging import version
        if version.parse(tf.__version__) >= version.parse('2.0.0'):
            import tensorflow.compat.v1 as tf
    except ImportError:
        raise ImportError(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
    return tf


def load_vars_from_tf_checkpoint(tf_path):
    tf = import_tensorflow()
    tf_path = io.abspath(tf_path)
    for postfix in '.data-00000-of-00001 .index'.split():
        cached_path = cache_file(tf_path + postfix)
    tf_path = cached_path[:-6]

    names = []
    arrays = []
    init_vars = tf.train.list_variables(tf_path)
    for name, _ in init_vars:
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    return names, arrays


def load_tf_savedmodel(tf_path, signature_def_key='serving_default'):
    # signature_def can be listed by
    #    saved_model_cli show --all --dir save_dir/
    import_tensorflow()
    from tensorflow.contrib import predictor
    tf_path = io.abspath(tf_path)

    for file in io.listdir(tf_path, recursive=True, full_path=True):
        cache_file(file)
    tf_path = cache_file(tf_path, dry=True)

    cwd = os.getcwd()
    os.chdir(tf_path)
    predict_fn = predictor.from_saved_model(tf_path, signature_def_key=signature_def_key)
    os.chdir(cwd)
    return predict_fn


def load_vars_from_tf_savedmodel(tf_path, signature_def_key='serving_default'):
    predict_fn = load_tf_savedmodel(tf_path, signature_def_key)
    graph = predict_fn.graph
    sess = predict_fn.session
    names = []
    arrays = []
    for op in graph.get_operations():
        if op.op_def and op.op_def.name == 'VariableV2':
            array = op.values()[0].eval(session=sess)
            names.append(op.name)
            arrays.append(array)
    return names, arrays


def load_tf_weights(model, tf_vars, name_map=[], name_fn=[], ignore_vars=[], transpose_vars=[],
                    retriever=[], strict=True):
    """ Load tf checkpoints in a pytorch model.
    model: XDPX Model instance
    tf_path: saved directory if savedmodel==True else tf checkpoint path
    name_map: extra name mapping from tf names to pytorch names
    savedmodel: whether to read the savedmodel format or tf checkpoints
    """

    def back_translate(name):
        """translate pytorch name mapping back to tf"""
        name = re.sub(r'\.', '/', name)
        name = name.replace('/weight', '/kernel')
        return name
    name_map = [(back_translate(orig_key), back_translate(new_key)) for orig_key, new_key in name_map]

    names, arrays = tf_vars
    model_vars = list(model.named_parameters())
    num_vars = len(model_vars)

    for name, array in zip(names, arrays):
        for orig_name, new_name in name_map:
            name = name.replace(orig_name, new_name)
        for fn in name_fn:
            name = fn(name)
        if any(ignore in name for ignore in ignore_vars):
            continue

        name = name.split("/")

        pointer = model
        try:
            for m_name in name:
                if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                    scope_names = re.split(r"_(\d+)", m_name)
                else:
                    scope_names = [m_name]

                for tf_key, key in retriever:
                    if tf_key == scope_names[0]:
                        pointer = getattr(pointer, key)
                        break
                else:
                    pointer = getattr(pointer, scope_names[0])
                if len(scope_names) >= 2:
                    num = int(scope_names[1])
                    pointer = pointer[num]

            if any(fn(name) for fn in transpose_vars):
                array = np.transpose(array)
            try:
                assert pointer.shape == array.shape, f'size mismatch for {"/".join(name)}: ' \
                                                     f'expected {str(pointer.shape)}, found {str(array.shape)}'
            except AssertionError as e:
                if strict:
                    raise e
                print('| Skipped due to ' + str(e))
                continue
        except AttributeError as e:
            print("| Unexpected keys: {}".format("/".join(name)))
            continue
        pointer.data = torch.from_numpy(array)
        model_vars = [(name, var) for name, var in model_vars if var is not pointer]

    print(f'| Load {num_vars - len(model_vars)} variables from Tensorflow')
    if model_vars:
        print('| Missing keys:')
        for name, _ in model_vars:
            print(f'|  {name}')
    return model


def create_vocab(filename='assets/vocab.txt', default=1):
    from tensorflow.python.ops.lookup_ops import HashTable
    from tensorflow.python.ops.lookup_ops import TextFileIdTableInitializer

    # placeholders
    table = HashTable(
        TextFileIdTableInitializer(filename=filename, name='vocab'),
        default_value=default,  # index of UNK
    )
    return table


def create_target_map(filename='assets/labels.txt', default='UNKNOWN'):
    from tensorflow.python.ops.lookup_ops import HashTable, TextFileStringTableInitializer
    table = HashTable(
        TextFileStringTableInitializer(filename, name='labels'),
        default_value=default
    )
    return table


def create_sequence_input(name, vocab=None, maxlen: int=None):
    tf = import_tensorflow()
    if vocab is not None:
        string = tf.placeholder(tf.string, [None, maxlen], name=f'{name}_str')
        x = tf.placeholder_with_default(tf.cast(vocab.lookup(string), tf.int32), [None, maxlen], name=name)
    else:
        x = tf.placeholder(tf.int32, [None, maxlen], name=name)
    mask = tf.expand_dims(tf.cast(tf.not_equal(x, 0), tf.float32), dim=-1)
    return x, mask


def load_into_tf(sess, state_dict, name_map, tensors_to_transpose=[], strict_size=False):
    """load pytorch state_dict into tf session"""
    tf = import_tensorflow()
    graph = sess.graph

    def to_tf_var_name(name: str):
        for patt, repl in iter(name_map):
            name = name.replace(patt, repl)
        return name

    with tqdm(state_dict, leave=False) as progress:
        added = []
        placeholders = {}
        for var_name in progress:
            tf_name = to_tf_var_name(var_name) + ':0'
            torch_tensor = state_dict[var_name].cpu().numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            try:
                tf_var = graph.get_tensor_by_name(tf_name)
                dtype = tf_var.dtype
                if dtype not in placeholders:
                    placeholders[dtype] = tf.placeholder(dtype,
                                                         name='load_' + re.search(r"'(.+)'", str(dtype)).group(1))
                placeholder = placeholders[dtype]
                if tf_var.shape != torch_tensor.shape:
                    print(f'| Mismatched keys: {var_name} (expect {tf_var.shape}, found {torch_tensor.shape})')
                    if strict_size:
                        print('If you want to skip mismatched weights, set "strict_size" to false.')
                        raise ValueError
                    continue
                tf_tensor = sess.run(tf.assign(tf_var, placeholder), feed_dict={placeholder: torch_tensor})
                added.append(tf_name)
            except KeyError as e:
                progress.write(f'| Unexpected keys: {var_name} (tf_name: {tf_name})')
            progress.update()
            progress.postfix = tf_name.ljust(80)[:80]
        tf_names = [v.name for v in tf.trainable_variables()]
        for name in set(tf_names) - set(added):
            print(f'| Missing keys: {name}')
