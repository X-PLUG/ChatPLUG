import torch
import torch.nn as nn
from typing import List
from packaging import version

from . import register, Model
from xdpx.modules import (
    Embedding,
    ConvLayer,
    MaxPooling,
    AttnPooling,
    LinearProjection,
)
from xdpx.options import Argument


@register('match_cnn')
class MatchCNN(Model):
    @staticmethod
    def register(options):
        options.register(
            Argument('embedding_dim', type=int, required=True),
            Argument('hidden_size', type=int, required=True),
            Argument('dropout', default=0.2),
            Argument('fix_embeddings', default=False),
            Argument('encoder_layers', type=int),
            Argument('kernel_sizes', type=List[int], default=[3],
                     validate=lambda value: all(k % 2 == 1 for k in value)),
            Argument('pooling', default='max', validate=lambda value: value in ['max', 'attn'], children={
                lambda value: value == 'attn': [
                    Argument('pool_hidden_size', type=int,
                             doc='if not None, use input projection with the specified hidden size'),
                    Argument('pool_output_size', type=int,
                             doc='if not None, use output projection with the specified output size'),
                ]
            }),
            Argument('prediction', default='full', validate=lambda value: value in predictions),
            domain='RE2',
        )

    def __init__(self, args):
        super().__init__(args)
        self.dropout = args.dropout
        self.embedding = Embedding(args.vocab_size, args.embedding_dim, args.fix_embeddings, args.dropout)
        self.encoder = ConvLayer(
            dropout=args.dropout, hidden_size=args.hidden_size, kernel_sizes=args.kernel_sizes,
            enc_layers=args.encoder_layers, activation='gelu', residual=getattr(args, 'residual', False),
            input_size=args.embedding_dim)
        output_size = args.hidden_size
        if args.pooling == 'max':
            self.pooling = MaxPooling()
        else:
            self.pooling = AttnPooling(args.hidden_size, args.pool_hidden_size, args.pool_output_size)
            if args.pool_output_size:
                output_size = args.pool_output_size
        self.prediction = predictions[args.prediction](output_size, args.num_classes, args.dropout)

    @classmethod
    def build(cls, args):
        model = super().build(args)
        model.load_embeddings()
        return model

    def get_embeddings(self):
        return self.embedding

    def forward(self, tokens1, tokens2, mask1, mask2, **unused):
        a, b = tokens1, tokens2
        mask_a, mask_b = mask1.unsqueeze(2), mask2.unsqueeze(2)

        a = self.embedding(a)
        b = self.embedding(b)

        a = self.encoder(a, mask_a)
        b = self.encoder(b, mask_b)

        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)
        return self.prediction(a, b)

    @property
    def dummy_inputs(self):
        a = torch.randint(0, self.args.vocab_size, (8, 16))
        b = torch.randint(0, self.args.vocab_size, (8, 16))
        return a, b, a > 0, b > 0


class Prediction(nn.Module):
    def __init__(self, input_size, num_classes, dropout, inp_features=2):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(dropout),
            LinearProjection(input_size * inp_features, input_size, activation='gelu'),
            nn.Dropout(dropout),
            LinearProjection(input_size, num_classes),
        )

    def forward(self, a, b):
        return self.dense(torch.cat([a, b], dim=-1))


class AdvancedPrediction(Prediction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, inp_features=4)

    def forward(self, a, b):
        return self.dense(torch.cat([a, b, a - b, a * b], dim=-1))


class SymmetricPrediction(AdvancedPrediction):
    def forward(self, a, b):
        return self.dense(torch.cat([a, b, (a - b).abs(), a * b], dim=-1))


class ClsPrediction(Prediction):
    def forward(self, a, b=None):
        assert b is None
        return self.dense(a)


predictions = {
    'simple': Prediction,
    'full': AdvancedPrediction,
    'symmetric': SymmetricPrediction,
}


@register('cls_cnn')
class ClassificationCnn(Model):
    @property
    def loss_type(self):
        return 'cross_entropy'

    @staticmethod
    def register(options):
        options.register(
            Argument('embedding_dim', type=int, required=True),
            Argument('hidden_size', type=int, required=True),
            Argument('dropout', default=0.2),
            Argument('fix_embeddings', default=False),
            Argument('encoder_layers', type=int),
            Argument('kernel_sizes', type=List[int], default=[3],
                     validate=lambda value: all(k % 2 == 1 for k in value)),
            Argument('residual', default=False),
            Argument('pooling', default='max', validate=lambda value: value in ['max', 'attn'], children={
                lambda value: value == 'attn': [
                    Argument('pool_hidden_size', type=int,
                             doc='if not None, use input projection with the specified hidden size'),
                    Argument('pool_output_size', type=int,
                             doc='if not None, use output projection with the specified output size'),
                ]
            }),
            domain='ClsCnn',
        )

    def __init__(self, args):
        super().__init__(args)
        self.dropout = args.dropout
        self.embedding = Embedding(args.vocab_size, args.embedding_dim, args.fix_embeddings, args.dropout)
        self.encoder = ConvLayer(
            dropout=args.dropout, hidden_size=args.hidden_size, kernel_sizes=args.kernel_sizes,
            enc_layers=args.encoder_layers, activation='gelu', residual=getattr(args, 'residual', False),
            input_size=args.embedding_dim)
        output_size = args.hidden_size
        if args.pooling == 'max':
            self.pooling = MaxPooling()
        else:
            self.pooling = AttnPooling(args.hidden_size, args.pool_hidden_size, args.pool_output_size)
            if args.pool_output_size:
                output_size = args.pool_output_size
        self.prediction = ClsPrediction(output_size, args.num_classes, args.dropout, 1)

    def get_embeddings(self):
        return self.embedding

    @classmethod
    def build(cls, args):
        model = super().build(args)
        model.load_embeddings()
        return model

    def forward(self, tokens, mask, **unused):
        a = tokens
        mask_a = mask.unsqueeze(2)
        a = self.embedding(a)
        a = self.encoder(a, mask_a)
        a = self.pooling(a, mask_a)
        return self.prediction(a)

    @property
    def dummy_inputs(self):
        g = torch.Generator()
        g.manual_seed(1)
        tokens = torch.randint(1, self.args.vocab_size, (8, self.args.max_len), generator=g)
        tokens[-2:, -2:] = 0
        return tokens, tokens > 0

    def build_tf_graph(self, sess):
        import tensorflow as tf
        from xdpx.utils.tf_utils import create_vocab, create_target_map, create_sequence_input
        from xdpx.modules.thirdparty.re2.modules.embedding import Embedding
        from xdpx.modules.thirdparty.re2.modules.encoder import Encoder
        from xdpx.modules.thirdparty.re2.modules.pooling import pooling
        from xdpx.modules.thirdparty.re2.modules import dense

        # placeholders
        inputs, mask = create_sequence_input('input', vocab=create_vocab(),
                                             maxlen=self.args.max_len if self.args.fix_len else None)
        target_map = create_target_map()
        dropout_keep_prob = tf.placeholder_with_default(tf.constant(1.0), (), name='dropout_keep_prob')

        args = self.args.change(
            enc_layers=self.args.encoder_layers,
            num_vocab=self.args.vocab_size
        )

        # forward
        x = Embedding(args)(inputs, dropout_keep_prob)
        x = Encoder(args)(x, mask, dropout_keep_prob)
        x = pooling(x, mask)
        with tf.variable_scope('prediction'):
            x = tf.nn.dropout(x, dropout_keep_prob)
            x = dense(x, self.args.hidden_size, activation=tf.nn.relu, name='dense_1')
            x = tf.nn.dropout(x, dropout_keep_prob)
            logits = dense(x, self.args.num_classes, activation=None, name='dense_2')

        if self.loss_type == 'cross_entropy':
            tf.nn.softmax(logits, dim=1, name='prob')
            pred_index = tf.argmax(input=logits, axis=1, name='pred_index')
            target_map.lookup(tf.cast(pred_index, tf.int64), name='pred')
        elif self.loss_type == 'bce':
            assert self.args.predict_threshold is not None
            threshold = [[0.5] + self.args.predict_threshold]
            threshold = tf.placeholder_with_default(tf.constant(threshold), [1, None], name='bce_threshold')

            prob = tf.nn.sigmoid(logits, name='prob')
            # assume minimum threshold > 0.1
            prob = tf.where(prob < threshold, tf.ones_like(prob) * 0.01, prob)
            prob = tf.concat([tf.ones((tf.shape(prob)[0], 1)) * 0.1, prob[:, 1:]], axis=1)
            pred_index = tf.argmax(input=prob, axis=1, name='pred_index')
            target_map.lookup(tf.cast(pred_index, tf.int64), name='pred')
        else:
            raise NotImplementedError

        sess.run(tf.global_variables_initializer())
        graph = sess.graph
        inputs = {
            'input_ids': graph.get_tensor_by_name('input:0'),
        }
        outputs = {
            'prob': graph.get_tensor_by_name('prob:0'),
            'pred_index': graph.get_tensor_by_name('pred_index:0'),
        }
        if self.args.init_tables:
            inputs['input_text'] = graph.get_tensor_by_name('input_str:0')
            if version.parse(tf.__version__) >= version.parse('1.14'):
                pred_name = 'pred/LookupTableFindV2:0'
            else:
                pred_name = 'pred:0'
            outputs['pred'] = graph.get_tensor_by_name(pred_name)
        return dict(
            inputs=inputs, outputs=outputs
        )

    def dummy_tf_inputs(self, inputs=None):
        if inputs is None:
            inputs = self.dummy_inputs
        inputs, mask = inputs
        return dict(
            input_ids=inputs.tolist(),
        )

    def load_into_tf(self, sess, strict=True):
        from xdpx.utils.tf_utils import load_into_tf
        predict_signature = self.build_tf_graph(sess)

        load_into_tf(sess, self.state_dict(), name_map=(
            # BERT base model
            ('encoders.', 'cnn_'),
            ('.model.0', '_0'),
            ('.model.1', '_1'),
            ('.model.2', '_2'),
            ('model.', ''),
            ('embedding.embedding.weight', 'embedding/embedding_matrix'),
            ('dense.1', 'dense_1'),
            ('dense.3', 'dense_2'),
            ('weight_v', 'weight'),
            ('.', '/'),
        ), tensors_to_transpose=(
            'weight_g',
            'weight_v'
        ), strict_size=strict)

        return predict_signature


@register('xdp_cls_cnn')
class XDPClassificationCnn(ClassificationCnn):
    def build_tf_graph(self, sess):
        import tensorflow as tf
        from tensorflow.python.ops.lookup_ops import HashTable
        from tensorflow.python.ops.lookup_ops import TextFileIdTableInitializer, TextFileStringTableInitializer
        from xdpx.modules.thirdparty.re2.modules.embedding import Embedding
        from xdpx.modules.thirdparty.re2.modules.encoder import Encoder
        from xdpx.modules.thirdparty.re2.modules.pooling import pooling
        from xdpx.modules.thirdparty.re2.modules import dense

        # tables
        vocab = HashTable(
            TextFileIdTableInitializer(filename='assets/vocab.txt', name='vocab'),
            default_value=1,  # index of UNK
        )
        target_map = HashTable(
            TextFileStringTableInitializer(filename='assets/labels.txt', name='labels'),
            default_value="UNKNOWN"
        )
        # placeholders
        string = tf.placeholder(tf.string, [None, None], name=f'inputs_x')
        tf.placeholder(tf.string, [None, None, None], name='char_inputs_x')
        inputs = tf.placeholder_with_default(tf.cast(vocab.lookup(string), tf.int32), [None, None], name='input_ids')
        dropout_keep_prob = tf.placeholder_with_default(tf.constant(1.0), (), name='dropout_keep_prob')
        tf.placeholder_with_default(tf.constant(False), (), name='is_train')
        tf.placeholder_with_default(tf.constant(0.3), (), name='threshold')
        tf.placeholder_with_default(tf.constant(3, dtype=tf.int64), (), name='top_n')
        tf.placeholder(tf.string, )
        mask = tf.expand_dims(tf.cast(tf.not_equal(inputs, 0), tf.float32), dim=-1)

        if self.loss_type == 'bce':
            assert self.args.predict_threshold is not None
            threshold = [[0.5] + self.args.predict_threshold]
            threshold = tf.placeholder_with_default(tf.constant(threshold), [1, None], name='bce_threshold')
        args = self.args.change(
            enc_layers=self.args.encoder_layers,
            num_vocab=self.args.vocab_size
        )

        # forward
        x = Embedding(args)(inputs, dropout_keep_prob)
        x = Encoder(args)(x, mask, dropout_keep_prob)
        x = pooling(x, mask)
        with tf.variable_scope('prediction'):
            x = tf.nn.dropout(x, dropout_keep_prob)
            x = dense(x, self.args.hidden_size, activation=tf.nn.relu, name='dense_1')
            x = tf.nn.dropout(x, dropout_keep_prob)
            logits = dense(x, self.args.num_classes, activation=None, name='dense_2')

        if self.loss_type == 'cross_entropy':
            prob = tf.nn.softmax(logits, name='prob')
            origin_prob = prob
        elif self.loss_type == 'bce':
            prob = tf.nn.sigmoid(logits, name='prob')
            origin_prob = prob
            # assume minimum threshold > 0.1
            prob = tf.where(prob < threshold, tf.ones_like(prob) * 0.01, prob)
            prob = tf.concat([tf.ones((tf.shape(prob)[0], 1)) * 0.1, prob[:, 1:]], axis=1)
        else:
            raise NotImplementedError
        sorted_probs, sorted_label_indices = tf.nn.top_k(prob, 3, sorted=True)
        sorted_labels = target_map.lookup(tf.cast(sorted_label_indices, tf.int64), name='sorted_labels')
        if version.parse(tf.__version__) >= version.parse('1.14'):
            tf.identity(tf.gather(origin_prob, sorted_label_indices, batch_dims=-1), name='sorted_probs')
        else:
            tf.identity(tf.gather(origin_prob, sorted_label_indices, axis=-1), name='sorted_probs')
        tf.identity(sorted_labels[:, 0], name='pred')

        sess.run(tf.global_variables_initializer())
        graph = sess.graph
        if version.parse(tf.__version__) >= version.parse('1.14'):
            sorted_labels_name = 'sorted_labels/LookupTableFindV2:0'
        else:
            sorted_labels_name = 'sorted_labels:0'
        return dict(
            inputs={
                'char_inputs_x': graph.get_tensor_by_name('char_inputs_x:0'),
                'inputs_x': graph.get_tensor_by_name('inputs_x:0'),
                'input_ids': graph.get_tensor_by_name('input_ids:0'),
                'is_train': graph.get_tensor_by_name('is_train:0'),
                'threshold': graph.get_tensor_by_name('threshold:0'),
                'top_n': graph.get_tensor_by_name('top_n:0'),
            }, outputs={
                'prediction': graph.get_tensor_by_name('pred:0'),
                'prob': graph.get_tensor_by_name('prob:0'),
                'sorted_labels': graph.get_tensor_by_name(sorted_labels_name),
                'sorted_probs': graph.get_tensor_by_name('sorted_probs:0'),
            }
        )

    def dummy_tf_inputs(self, inputs=None):
        if inputs is None:
            inputs = self.dummy_inputs
        inputs, mask = inputs
        return dict(
            input_ids=inputs.tolist(),
        )


@register('cls_cnn_bce')
class ClassificationCnnBCE(ClassificationCnn):
    @property
    def loss_type(self):
        return 'bce'


@register('xdp_cls_cnn_bce')
class XDPClassificationCnnBCE(XDPClassificationCnn, ClassificationCnnBCE):
    ...
