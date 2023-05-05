import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import List

from . import register, Model
from xdpx.modules import (
    Embedding,
    ConvLayer,
    MaxPooling,
    LinearProjection,
    Alignment,
    MappedAlignment,
)
from xdpx.options import Argument


@register('re2')
class RE2(Model):
    @staticmethod
    def register(options):
        options.register(
            Argument('embedding_dim', type=int, required=True),
            Argument('hidden_size', type=int, required=True),
            Argument('blocks', type=int, required=True),
            Argument('dropout', default=0.2),
            Argument('fix_embeddings', default=True),
            Argument('encoder_layers', type=int, required=True),
            Argument('kernel_sizes', type=List[int], default=[3], validate=lambda value: all(k % 2 == 1 for k in value)),
            Argument('alignment', default='linear', validate=lambda value: value in ('identity', 'linear')),
            Argument('prediction', default='full', validate=lambda value: value in predictions),
            domain='RE2',
        )

    def __init__(self, args):
        super().__init__(args)
        self.dropout = args.dropout
        self.embedding = Embedding(args.vocab_size, args.embedding_dim, args.fix_embeddings, args.dropout)
        self.blocks = nn.ModuleList([nn.ModuleDict({
            'encoder': ConvLayer(
                dropout=args.dropout, hidden_size=args.hidden_size, kernel_sizes=args.kernel_sizes, 
                enc_layers=args.encoder_layers, activation='gelu',
                input_size=args.embedding_dim if i == 0 else args.embedding_dim + args.hidden_size),
            'alignment': alignments(args)(
                args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
            'fusion': Fusion(
                args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2,
                args.hidden_size, args.dropout),
        }) for i in range(args.blocks)])
        self.connection = AugmentedResidual()
        self.pooling = MaxPooling()
        self.prediction = predictions[args.prediction](args)
    
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
        res_a, res_b = a, b

        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)
                b = self.connection(b, res_b, i)
                res_a, res_b = a, b
            a_enc = block['encoder'](a, mask_a)
            b_enc = block['encoder'](b, mask_b)
            a = torch.cat([a, a_enc], dim=-1)
            b = torch.cat([b, b_enc], dim=-1)
            align_a, align_b = block['alignment'](a, b, mask_a, mask_b)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)
        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)
        return self.prediction(a, b)

    @property
    def dummy_inputs(self):
        a = torch.randint(0, self.args.vocab_size, (8, 16))
        b = torch.randint(0, self.args.vocab_size, (8, 12))
        return a, b, a > 0, b > 0

    def build_tf_graph(self, sess):
        import tensorflow as tf
        from xdpx.utils.tf_utils import create_vocab, create_sequence_input
        from xdpx.modules.thirdparty.re2.network import Network

        # placeholders
        vocab = create_vocab()
        q1, q1_mask = create_sequence_input('q1', vocab)
        q2, q2_mask = create_sequence_input('q2', vocab)
        dropout_keep_prob = tf.placeholder_with_default(tf.constant(1.0), (), name='dropout_keep_prob')

        assert len(self.args.kernel_sizes) == 1
        network = Network(self.args.change(fusion='full', connection='aug', enc_layers=self.args.encoder_layers,
                                           kernel_size=self.args.kernel_sizes[0], num_vocab=self.args.vocab_size))
        logits = network(q1, q2, q1_mask, q2_mask, dropout_keep_prob)
        tf.nn.softmax(logits, dim=1, name='prob')
        tf.argmax(input=logits, axis=1, name='pred')

        sess.run(tf.global_variables_initializer())
        graph = sess.graph
        return dict(
            inputs={
                'inputs_q1': graph.get_tensor_by_name('q1_str:0'),
                'inputs_q2': graph.get_tensor_by_name('q2_str:0'),
                'inputs_q1_id': graph.get_tensor_by_name('q1:0'),
                'inputs_q2_id': graph.get_tensor_by_name('q2:0'),
                'dropout_keep_prob': graph.get_tensor_by_name('dropout_keep_prob:0'),
            }, outputs={
                'pred': graph.get_tensor_by_name('pred:0'),
                'prob': graph.get_tensor_by_name('prob:0'),
            }
        )

    def dummy_tf_inputs(self, inputs=None):
        if inputs is None:
            inputs = self.dummy_inputs
        q1, q2, mask1, mask2 = inputs
        return dict(
            inputs_q1_id=q1.tolist(),
            inputs_q2_id=q2.tolist(),
        )

    def load_into_tf(self, sess, strict=True):
        from xdpx.utils.tf_utils import load_into_tf
        predict_signature = self.build_tf_graph(sess)

        load_into_tf(sess, self.state_dict(), name_map=(
            # BERT base model
            ('blocks.', 'block-'),
            ('encoders.', 'cnn_'),
            ('.model.0', '_0'),
            ('.model.1', '_1'),
            ('.model.2', '_2'),
            ('.model.3', '_2'),
            ('.model.4', '_2'),
            ('fusion.fusion1', 'align/orig'),
            ('fusion.fusion2', 'align/sub'),
            ('fusion.fusion3', 'align/mul'),
            ('fusion.fusion', 'align/proj'),
            ('embedding.embedding.weight', 'embedding/embedding_matrix'),
            ('model.0.', ''),
            ('model.', ''),
            ('dense.1', 'dense_1'),
            ('dense.3', 'dense_2'),
            ('weight_v', 'weight'),
            ('.', '/'),
        ), tensors_to_transpose=(
            'weight_g',
            'weight_v'
        ), strict_size=strict)

        return predict_signature


class AugmentedResidual(nn.Module):
    def forward(self, x, res, i):
        if i == 1:
            return torch.cat([x, res], dim=-1)  # res is embedding
        hidden_size = x.size(-1)
        x = (res[:, :, :hidden_size] + x) * math.sqrt(0.5)
        return torch.cat([x, res[:, :, hidden_size:]], dim=-1)  # latter half of res is embedding


class Fusion(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.dropout = dropout
        self.fusion1 = LinearProjection(input_size * 2, hidden_size, activation='gelu')
        self.fusion2 = LinearProjection(input_size * 2, hidden_size, activation='gelu')
        self.fusion3 = LinearProjection(input_size * 2, hidden_size, activation='gelu')
        self.fusion = LinearProjection(hidden_size * 3, hidden_size, activation='gelu')

    def forward(self, x, align):
        x1 = self.fusion1(torch.cat([x, align], dim=-1))
        x2 = self.fusion2(torch.cat([x, x - align], dim=-1))
        x3 = self.fusion3(torch.cat([x, x * align], dim=-1))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = F.dropout(x, self.dropout, self.training)
        return self.fusion(x)


class Prediction(nn.Module):
    def __init__(self, args, inp_features=2):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(args.dropout),
            LinearProjection(args.hidden_size * inp_features, args.hidden_size, activation='gelu'),
            nn.Dropout(args.dropout),
            LinearProjection(args.hidden_size, args.num_classes),
        )

    def forward(self, a, b):
        return self.dense(torch.cat([a, b], dim=-1))


class AdvancedPrediction(Prediction):
    def __init__(self, args):
        super().__init__(args, inp_features=4)

    def forward(self, a, b):
        return self.dense(torch.cat([a, b, a - b, a * b], dim=-1))


class SymmetricPrediction(AdvancedPrediction):
    def forward(self, a, b):
        return self.dense(torch.cat([a, b, (a - b).abs(), a * b], dim=-1))


def alignments(args):
    if args.alignment == 'identity':
        return Alignment
    elif args.alignment == 'linear':
        return partial(MappedAlignment, hidden_size=args.hidden_size, dropout=args.dropout)
    else:
        raise NotImplementedError(f'unknown alignment type "{args.alignment}"')


predictions = {
    'simple': Prediction,
    'full': AdvancedPrediction,
    'symmetric': SymmetricPrediction,
}
