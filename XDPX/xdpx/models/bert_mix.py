from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xdpx.modules import (
    Embedding,
    ConvLayer,
    MaxPooling,
    LinearProjection,
    KeyAttnPooling,
)
from xdpx.utils.versions import torch_lt_120, torch_ge_150
from xdpx.options import Argument
from . import register
from .bert import Bert


@register('bert_mix')
class BertMix(Bert):
    @staticmethod
    def register(options):
        options.register(
            Argument('embedding_dim', type=int, required=True),
            Argument('task_type', validate=lambda x: x in ['classification', 'matching'], required=True, 
                children={
                    lambda value: value == 'matching': [
                        Argument('prediction', validate=lambda value: value in ('simple', 'dense'), default='simple'),
                    ],
                    lambda value: value == 'classification': [
                        Argument('pooling', default='max', validate=lambda value: value in ('max', 'max_attn')),
                    ],
                }),
            Argument('output_dropout_prob', default=0.0),
            Argument('word_emb', default=True, children=[
                Argument('word_emb_proj', default=None, type=Optional[int]),
            ]),
            Argument('char_emb', default=True, children=[
                Argument('bert_seq_layer', default=-1, type=int,
                    doc='0th layer means bert embedding layer; transformer layers start from 1.'),
                Argument('char_layer_norm', default=True),
                Argument('char_hidden_size', default=768, type=int),
            ]),
            Argument('concat_cls', default=True),
            Argument('fusion', default=True, children=[
                Argument('fusion_hidden_size', default=150),
                Argument('fusion_layers', default=2, validate=lambda value: value in (1, 2)),
                Argument('fusion_activation', default=False),
            ]),
            Argument('pooler', default=False),
            Argument('seq_pooler', default=False),
        )
        options.register(
            Argument('char_encoder_layers', default=1),
            Argument('kernel_sizes', type=List[int], default=3,
                post_process=lambda value: [value] if isinstance(value, int) else value),
            Argument('residual', default=False),
            Argument('encoder_layers', default=3),
            Argument('encoder_hidden_size', default=200),
            domain='cnn'
        )
        options.set_default('strict_size', False, strict=False)
        options.set_default('extra_config', {'output_hidden_states': True})
        options.add_global_constraint(lambda args: args.word_emb or args.char_emb)
        options.add_global_constraint(lambda args: not args.word_emb or args.word_vocab_size)

    def __init__(self, args):
        super().__init__(args)
        if args.char_emb:
            self.char_encoder = ConvLayer(dropout=self.args.hidden_dropout_prob,
                hidden_size=args.char_hidden_size, kernel_sizes=args.kernel_sizes,
                enc_layers=args.char_encoder_layers, activation='gelu',
                input_size=self.args.hidden_size, residual=args.residual)
            self.char_pooling = MaxPooling()
            if args.char_layer_norm:
                # whether to use default eps?
                self.char_layer_norm = nn.LayerNorm(args.char_hidden_size)
        if args.word_emb:
            self.embedding = Embedding(
                args.word_vocab_size, args.embedding_dim, fix_embeddings=False, dropout=self.args.hidden_dropout_prob)
            if args.word_emb_proj:
                self.embedding_proj = nn.Sequential(
                    LinearProjection(args.embedding_dim, args.word_emb_proj, activation='gelu'),
                    nn.LayerNorm(args.word_emb_proj),
                )
        enc_in = (args.char_hidden_size if args.char_emb else 0) + args.embedding_dim * args.word_emb
        self.encoder = ConvLayer(
            dropout=self.args.hidden_dropout_prob,
            hidden_size=args.encoder_hidden_size, kernel_sizes=args.kernel_sizes,
            enc_layers=self.args.encoder_layers, activation='gelu',
            input_size=enc_in, 
            residual=args.residual)
        if args.fusion:
            fusion_act = 'tanh' if args.fusion_activation else 'linear'
            if args.fusion_layers == 2:
                self.fusion = nn.Sequential(
                    LinearProjection(
                        args.encoder_hidden_size + enc_in,
                        args.encoder_hidden_size, activation='gelu'),
                    # nn.Dropout(self.top_dropout),
                    LinearProjection(args.encoder_hidden_size, args.fusion_hidden_size, activation=fusion_act),
                )
            else:
                assert args.fusion_layers == 1
                self.fusion = nn.Sequential(
                    LinearProjection(
                        args.encoder_hidden_size + enc_in, 
                        args.fusion_hidden_size, activation=fusion_act),
                )
        out_hidden_size = (args.fusion_hidden_size if args.fusion else args.encoder_hidden_size)
        if args.task_type == 'matching':
            raise NotImplementedError
        else:
            if args.pooling == 'max_attn':
                self.pooling = KeyAttnPooling(self.args.hidden_size, key_size=self.args.hidden_size)
                raise NotImplementedError
            self.word_pooling = MaxPooling()
            cls_hidden_size = self.args.hidden_size
            if args.seq_pooler:
                self.seq_pooler = LinearProjection(
                    out_hidden_size,
                    args.fusion_hidden_size, activation='tanh'
                )
                out_hidden_size = args.fusion_hidden_size
            if args.pooler:
                self.pooler = LinearProjection(
                    out_hidden_size + args.concat_cls * cls_hidden_size, 
                    args.fusion_hidden_size, activation='tanh')
                out_hidden_size = args.fusion_hidden_size
            else:
                out_hidden_size = out_hidden_size + args.concat_cls * cls_hidden_size
        self.classifier = LinearProjection(out_hidden_size, args.num_classes)
    
    @classmethod
    def build(cls, args):
        model = super().build(args)
        if args.word_emb:
            model.load_embeddings('embedding')
        return model
        
    def forward(self, input_ids, token_mask, words, word_mask, word_begin_mask, **kwargs):
        if self.args.char_emb or self.args.concat_cls:
            _, cls_token, bert_features = self.bert_forward(input_ids, attention_mask=token_mask, **kwargs)
            if self.args.char_emb:
                bert_features = bert_features[self.args.bert_seq_layer]
            else:
                bert_features = None
        else:
            cls_token = bert_features  = None
        return self.cnn_forward(bert_features, token_mask, words, word_mask, word_begin_mask, cls_token)
    
    def cnn_forward(self, bert_features, token_mask, words, word_mask, word_begin_mask, cls_token=None):
        # use [SEP] as the tailing character of the last word, which may improve performance
        token_mask = torch.unsqueeze(token_mask[:, 1:], 2)
        word_begin_mask = torch.cat([word_begin_mask, torch.zeros(word_begin_mask.size(0), 1).to(word_begin_mask)], dim=1)
        word_mask = word_mask.unsqueeze(2)

        if self.args.char_emb:                                     # exclude [CLS]
            char_emb, char_mask = self.build_char_emb(bert_features[:, 1:, :], token_mask, word_begin_mask)
            B, Lw, Lc, D = char_emb.size()
            char_emb = self.char_encoder(char_emb.view(-1, Lc, D), char_mask.view(-1, Lc, 1)).view(B, Lw, Lc, -1)
            char_emb = self.char_pooling(char_emb, char_mask, dim=2)
            if self.args.char_layer_norm:
                char_emb = self.char_layer_norm(char_emb)
            features = char_emb
        if self.args.word_emb:
            word_emb = self.embedding(words)
            if self.args.word_emb_proj:
                word_emb = self.embedding_proj(word_emb)
            features = word_emb
        if self.args.char_emb and self.args.word_emb:
            features = torch.cat([char_emb, word_emb], 2)
        
        enc_out = self.encoder(features, word_mask)
        if self.args.fusion:
            features = torch.cat([features, enc_out], 2)
            features = self.fusion(features)
        else:
            features = enc_out
        if self.args.task_type == 'classification':
            x1 = self.word_pooling(features, word_mask, dim=1)
        else:
            raise NotImplementedError
        
        if self.args.seq_pooler:
            x1 = self.seq_pooler(x1)
        if self.args.concat_cls:
            x1 = torch.cat([x1, cls_token], 1)
        if self.args.pooler:
            x1 = self.pooler(x1)
        x1 = F.dropout(x1, self.args.output_dropout_prob, self.training)
        return self.classifier(x1)

    @staticmethod
    def build_char_emb(tokens, tokens_mask, word_begin_mask):
        """
        word_begin_mask: B x L
        """
        batches = torch.ones_like(word_begin_mask).cumsum(0) - 1
        rows = word_begin_mask.cumsum(1) - 1
        columns = (1 - word_begin_mask).cumsum(1)
        if torch_lt_120():
            m = np.roll(columns.cpu().numpy(), 1, axis=1)
            m[:, 0] = 0
            m *= word_begin_mask.cpu().numpy()
            m = np.maximum.accumulate(m, 1)
            columns = columns - columns.new_tensor(m)
        else:
            m = columns.roll(1, 1)
            m[:, 0] = 0
            m *= word_begin_mask
            if torch_ge_150():
                m = m.cummax(1)[0]
            else:
                m = torch.from_numpy(np.maximum.accumulate(m.cpu().numpy(), 1)).to(m)
            columns -= m
        
        mask = tokens_mask.squeeze(2)
        batches = batches.masked_select(mask)
        rows = rows.masked_select(mask)
        columns = columns.masked_select(mask)

        max_word_len = word_begin_mask.sum(1).max()
        max_char_len = columns.max() + 1
        output = torch.zeros(tokens.size(0), max_word_len, max_char_len, tokens.size(2)).to(tokens)
        output_mask = torch.zeros_like(output[:, :, :, :1])

        values = tokens.masked_select(tokens_mask).view(-1, tokens.size(2))
        output[batches, rows, columns, :] = values
        output_mask[batches, rows, columns, :] = 1
        if torch_lt_120():
            output_mask = output_mask.byte()
        else:
            output_mask = output_mask.bool()
        return output, output_mask

    @property
    def dummy_inputs(self):
        tokens = torch.randint(1, self.args.vocab_size, (8, 18))
        token_mask = tokens.ge(0)
        words = torch.randint(1, self.args.word_vocab_size, (8, 8))
        word_mask = words.ge(0)
        word_begin_mask = torch.ones(8, 16)
        word_begin_mask[:, 1::2] = 0
        word_begin_mask = word_begin_mask.long()
        return tokens, token_mask, words, word_mask, word_begin_mask
