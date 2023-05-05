import math
import torch
import numpy as np
from typing import List, Optional, Union
from xdpx.options import Argument
from xdpx.utils import numpy_seed
from . import register
from .bert_lm import BertMaskedLMProcessor


@register('bert_prompt_single')
class BertPromptSingleProcessor(BertMaskedLMProcessor):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('label_length', default=2),
            Argument('hard_templates', default=None, type=list,
                     validate=lambda value: value is None or all(['<unk>' in v for v in value])),
            Argument('soft_prompt_length', default=None),
            Argument('labels_ext', default=None),
            domain="pet/ptuning"
        )

    def __init__(self, args):
        super().__init__(args)
        self._label_ids = None

    def numerize(self, inputs: dict):

        prefix_len = max([len(p) for p in inputs['prefix']])
        content_ids = self.numerize_tokens(
            inputs['content'][:self.args.max_len - self.special_tokens - prefix_len])

        results = {
            'id': inputs['id'],
            'content_ids': content_ids,
            'prefix_ids': [self.numerize_tokens(p) for p in inputs['prefix']],
            'word_begin_mask': inputs['word_begin_mask'][:self.args.max_len - self.special_tokens]
            if self.args.mask_whole_words else None
        }
        return results

    def numerize_target(self, sample):
        if 'target' not in sample:
            return {}
        return {
            'target': self.target_map.encode(sample['target']),
            'target_tokens': self.numerize_tokens(sample['target_tokens']),
            'target_symbol': sample['target']
        }

    @property
    def label_ids(self):
        if self._label_ids is None:
            labels = []
            if self.args.labels_ext is None:
                labels = self.target_map.symbols
                max_ext_size = 0
            else:
                labels_ext = {}
                max_ext_size = 0
                for t in self.args.labels_ext.strip().split(';'):
                    if len(t.split('=')) == 2:
                        key, value = t.split('=')
                        ext_words = value.split('/')
                        labels_ext[key] = ext_words
                        if len(ext_words) > max_ext_size:
                            max_ext_size = len(ext_words)

                for index, label in enumerate(self.target_map.symbols):
                    labels.append(label)
                    if label in labels_ext:
                        for w in labels_ext[label]:
                            labels.append(w)
                        labels.extend([label] * (max_ext_size - len(labels_ext[label])))
                    else:
                        labels.extend([label] * max_ext_size)

            labels = torch.tensor([self.numerize_tokens(list(label)) for label in labels], dtype=torch.long)
            ext_count = max_ext_size + 1
            self._label_ids = (
                labels,
                ext_count
            )

        return self._label_ids

    def collate(self, samples):
        input_ids = []
        for sample in samples:
            for prefix_ids in sample['prefix_ids']:
                masked_inputs = [self.args.cls_index] + prefix_ids + [self.args.sep_index] + sample['content_ids'] + [self.args.sep_index]
                input_ids.append(masked_inputs)

        input_ids = torch.tensor(self.pad(input_ids), dtype=torch.long)
        prefix_size = len(samples[0]['prefix_ids'])

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,
            },
            'prefix_size': prefix_size,
            'label_ids': self.label_ids,
            'label_symbols': self.target_map.symbols,
            'ntokens': input_ids.numel(),
        }
        try:
            label_length = self.args.label_length
            targets = []
            target_ids = []
            target_symbols = []
            for sample in samples:
                for prefix_ids in sample['prefix_ids']:
                    start_mask_index = prefix_ids.index(self.args.mask_index)
                    masked_tokens = [self.args.pad_index] * (len(sample['content_ids']) + len(prefix_ids) + 1)

                    for i in range(label_length):
                        masked_tokens[start_mask_index + i] = sample['target_tokens'][i]
                    masked_tokens = [self.args.pad_index] + masked_tokens + [self.args.pad_index]
                    targets.append(masked_tokens)
                    target_ids.append(sample['target'])
                    target_symbols.append(sample['target_symbol'])

            target = torch.tensor(self.pad(targets), dtype=torch.long)
            target_id = torch.tensor(target_ids, dtype=torch.long)

            batch.update({'target': target, 'target_id': target_id, 'target_symbol': target_symbols})
        except KeyError:
            ...
        return batch

    def sanity_check(self, inputs: List[dict]):
        batch = self.collate(inputs)

        def decode(ids):
            return ' '.join(self.decode(ids)).replace(self.args.pad_word, '_')

        outputs = []
        for i in range(len(inputs)):
            input_ids = batch['net_input']['input_ids'][i].tolist()

            sample = {
                'id': batch['id'][i],
                'input_tokens': decode(input_ids)
            }
            if 'target' in batch:
                targets = batch['target'][i].tolist()
                sample['target'] = decode(targets)
            if 'target_id' in batch:
                sample['target_id'] = batch['target_id'][i]
            if 'target_symbol' in batch:
                sample['target_symbol'] = batch['target_symbol'][i]

            outputs.append(sample)
        return outputs

    @staticmethod
    def target_stream(data):
        for sample in data:
            if 'target' in sample:
                yield sample['target']
