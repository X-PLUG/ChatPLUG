import os
import torch
from typing import List, Iterator
from xdpx.options import Argument
from xdpx.utils import io, download_from_url
from . import register
from .bert import BertProcessor
from ..dictionary import Dictionary


class BertMixProcessor(BertProcessor):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('word_vocab_file', doc='If None, build from source data.', children={
                lambda value: value is None: [
                    Argument('min_word_count', default=-1),
                    Argument('max_word_vocab', default=-1),
                ]
            }),
            Argument('word_vocab_size', type=int, doc='auto set in preprocessing'),
        )
    
    def __init__(self, args):
        super().__init__(args)
        self._word_dictionary = None
    
    @property
    def word_dictionary(self) -> Dictionary:
        if not self._word_dictionary:
            path = os.path.join(self.args.data_dir, self._default_word_vocab_path)
            if io.exists(path):
                self._word_dictionary = Dictionary.load(path, pad=self.args.pad_word, unk=self.args.unk_word)
            elif self.args.word_vocab_file:
                if path != io.abspath(self.args.word_vocab_file):
                    io.copy(self.args.word_vocab_file, path)
                self._word_dictionary = Dictionary.load(path, pad=self.args.pad_word, unk=self.args.unk_word)
            else:
                # Build vocab from data. Only make sense in preprocessing steps.
                if not hasattr(self.args, 'data_source'):
                    raise ValueError(f'word_vocab_file must be provided in non-preprocessing phases.')
                dictionary = Dictionary(pad=self.args.pad_word, unk=self.args.unk_word)
                for data in self.args.__datasets__.values():
                    for word in self.word_stream(data):
                        dictionary.add_symbol(word)
                if self.args.pretrained_embeddings:
                    pretrained_embeddings = self.args.pretrained_embeddings
                    if pretrained_embeddings.startswith('http'):
                        pretrained_embeddings = download_from_url(pretrained_embeddings)
                    pretrained = self.load_embedding_dict(pretrained_embeddings)
                else:
                    pretrained = []
                dictionary.finalize(threshold=self.args.min_word_count, nwords=self.args.max_word_vocab, pretrained=pretrained)
                dictionary.save(path)
                self._word_dictionary = dictionary
            self._word_vocab_size = len(self._word_dictionary)

        return self._word_dictionary
    
    @property
    def _default_word_vocab_path(self):
        return 'word_vocab.txt'

    @property
    def resources(self):
        return super().resources + ['word_vocab.txt']
            
    @staticmethod
    def target_stream(data):
        for sample in data:
            if 'target' in sample:
                yield sample['target']
    
    def _autoset_meta(self):
        meta = super()._autoset_meta()
        meta.update({
            'word_vocab_file': self._default_word_vocab_path,
            'word_vocab_size': self._word_vocab_size,
        })
        return meta

    @property
    def _dict_for_emb(self):
        return self.word_dictionary

    @staticmethod
    def word_stream(data: List[dict]) -> Iterator[str]:
        raise NotImplementedError


@register('bert_single_mix')
class BertSingleMixProcessor(BertMixProcessor):
    @staticmethod
    def word_stream(data: List[dict]) -> Iterator[str]:
        for sample in data:
            for word in sample['words']:
                yield word
    
    def text_length(self, sample):
        return len(sample['tokens'])
    
    def numerize(self, inputs: dict):
        tokens = self.numerize_tokens(inputs['tokens'])
        words = self.word_dictionary.encode(inputs['words'])
        mask = inputs['word_begin_mask']
        max_len = self.args.max_len - 2
        if len(tokens) > max_len:
            # only keep whole words
            while not mask[max_len] and max_len > 0:
                max_len -= 1
            if not max_len:
                raise RuntimeError(f'Forget to do tokenization? A single word exceeds max_len: {inputs["words"]}')
            tokens = tokens[:max_len]
            mask = mask[:max_len]
            words = words[:sum(mask)]
        tokens = [self.args.cls_index] + tokens + [self.args.sep_index]

        results = {
            'id': inputs['id'], 
            'tokens': tokens, 
            'words': words,
            'word_begin_mask': mask,
        }
        return results

    def collate(self, samples):
        tokens = torch.LongTensor(self.pad([sample['tokens'] for sample in samples]))
        token_mask = torch.ne(tokens, self.args.pad_index)
        words = torch.LongTensor(self.pad([sample['words'] for sample in samples]))
        word_mask = torch.ne(words, self.args.pad_index)
        word_begin_mask = torch.LongTensor(self.pad([sample['word_begin_mask'] for sample in samples]))

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': tokens,
                'token_mask': token_mask,
                'words': words,
                'word_mask': word_mask,
                'word_begin_mask': word_begin_mask,
            },
            'ntokens': tokens.numel(),
        }
        try:
            target = torch.LongTensor([sample['target'] for sample in samples])
            batch.update({'target': target})
        except KeyError:
            ...
        return batch

    def sanity_check(self, inputs):
        batch = self.collate(inputs)

        def decode_char(ids):
            return ' '.join((self.decode(ids))).replace(self.args.pad_word, '_')
        
        def decode_word(ids):
            return ' '.join(self.word_dictionary.decode(ids)).replace(self.args.pad_word, '_')

        outputs = []
        for i in range(len(inputs)):
            tokens = batch['net_input']['input_ids'][i].tolist()
            token_mask = batch['net_input']['token_mask'][i].tolist()
            words = batch['net_input']['words'][i].tolist()
            word_mask = batch['net_input']['word_mask'][i].tolist()
            word_begin_mask = batch['net_input']['word_begin_mask'][i].tolist()

            sample = {
                'id': batch['id'][i],
                'tokens': decode_char(tokens),
                'token_mask': token_mask,
                'words': decode_word(words),
                'word_mask': word_mask,
                'word_begin_mask': word_begin_mask,
            }
            if 'target' in batch:
                target = batch['target'][i].tolist()
                sample['target'] = target
            outputs.append(sample)
        return outputs
