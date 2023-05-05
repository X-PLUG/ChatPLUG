import torch
from . import register, Processor


@register('string')
class StringProcessor(Processor):
    def numerize(self, inputs: dict):
        results = {
            'id': inputs['id'],
            'tokens': inputs['tokens'][:self.args.max_len],
        }
        return results

    def text_length(self, sample):
        return len(sample['tokens'])

    def _autoset_meta(self):
        meta = super()._autoset_meta()
        meta['pad_word'] = self.dictionary.pad_word
        return meta

    def collate(self, samples):
        tokens = self.pad([sample['tokens'] for sample in samples], pad_index=self.dictionary.pad_word)
        batch = {
            'net_input': {
                'tokens': tokens,
            },
            'ntokens': len(tokens) * len(tokens[0]),
        }
        try:
            target = torch.tensor([sample['target'] for sample in samples], dtype=torch.long)
            batch.update({'target': target})
        except KeyError:
            ...
        return batch

    @staticmethod
    def token_stream(data):
        for sample in data:
            for token in sample['tokens']:
                yield token

    @staticmethod
    def target_stream(data):
        for sample in data:
            if 'target' in sample:
                yield sample['target']


@register('string_mix')
class StringMixProcessor(StringProcessor):
    def numerize(self, inputs: dict):
        words = inputs['tokens'][:self.args.max_len]
        results = {
            'id': inputs['id'],
            'words': words,
            'tokens': [list(word) for word in words],
        }
        return results

    def collate(self, samples):
        words = self.pad([sample['words'] for sample in samples], pad_index=self.args.pad_word)
        tokens = self.pad3d([sample['tokens'] for sample in samples], pad_index=self.args.pad_word)
        batch = {
            'net_input': {
                'words': words,
                'tokens': tokens,
            },
            'ntokens': len(words) * len(words[0]),
        }
        try:
            target = torch.tensor([sample['target'] for sample in samples], dtype=torch.long)
            batch.update({'target': target})
        except KeyError:
            ...
        return batch
