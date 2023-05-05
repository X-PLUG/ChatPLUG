import math
import torch
import numpy as np
from typing import List, Optional, Union
from xdpx.options import Argument
from xdpx.utils import numpy_seed
from . import register
from .bert_lm import BertMaskedLMProcessor


@register('bert_cl_pair')
class BertCLPairProcessor(BertMaskedLMProcessor):
    def __init__(self, args):
        super().__init__(args)

    def numerize(self, inputs: dict):
        tokens1 = self.numerize_tokens(inputs['tokens1'])
        tokens2 = self.numerize_tokens(inputs['tokens2'])
        max_len = self.args.max_len - 2
        tokens1 = tokens1[:max_len]
        tokens2 = tokens2[:max_len]
        results = {
            'id': inputs['id'],
            'tokens1': tokens1,
            'tokens2': tokens2,
            'masks1': inputs['mask1'][:max_len],
            'masks2': inputs['mask2'][:max_len],
        }
        return results

    def mask_sent(self, sid, input_ids, word_begin_mask=None):
        with numpy_seed(self.args.seed, self.epoch, sid):
            masked_inputs, masked_tokens = self.generate_mask(
                input_ids, word_begin_mask,
            )
        masked_inputs = [self.args.cls_index] + masked_inputs + [self.args.sep_index]
        masked_tokens = [self.args.pad_index] + masked_tokens + [self.args.pad_index]
        return masked_inputs, masked_tokens

    def collate(self, samples):
        input_ids1, input_ids2 = [], []
        targets1, targets2 = [], []
        for sample in samples:
            masked_inputs1, masked_tokens1 = self.mask_sent(sample['id'], sample['tokens1'],
                                                            sample['masks1'] if self.args.mask_whole_words else None)
            masked_inputs2, masked_tokens2 = self.mask_sent(sample['id'], sample['tokens2'],
                                                            sample['masks2'] if self.args.mask_whole_words else None)
            input_ids1.append(masked_inputs1)
            input_ids2.append(masked_inputs2)
            targets1.append(masked_tokens1)
            targets2.append(masked_tokens2)

        input_ids = torch.tensor(self.pad(np.concatenate([input_ids1, input_ids2], axis=0)), dtype=torch.long)
        targets = torch.tensor(self.pad(np.concatenate([targets1, targets2], axis=0)), dtype=torch.long)
        orig_input_ids = torch.tensor(self.pad(np.concatenate(
            [[[self.args.cls_index] + sample['tokens1'] + [self.args.sep_index] for sample in samples],
             [[self.args.cls_index] + sample['tokens2'] + [self.args.sep_index] for sample in samples]], axis=0)),
            dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,
            },
            'orig_input_ids': orig_input_ids,
            'target': targets,
            'ntokens': input_ids.numel(),
        }
        return batch

    def sanity_check(self, inputs: List[dict]):
        batch = self.collate(inputs)

        def decode(ids):
            return ' '.join(self.decode(ids)).replace(self.args.pad_word, '_')

        outputs = []
        bsz = int(batch['orig_input_ids'].size(0) / 2)
        for i in range(len(inputs)):
            input_ids1 = batch['net_input']['input_ids'][i].tolist()
            input_ids2 = batch['net_input']['input_ids'][i + bsz].tolist()
            orig_input_ids1 = batch['orig_input_ids'][i].tolist()
            orig_input_ids2 = batch['orig_input_ids'][i + bsz].tolist()
            targets1 = batch['target'][i].tolist()
            targets2 = batch['target'][i + bsz].tolist()

            sample = {
                'id': batch['id'][i],
                'input_tokens1': decode(input_ids1),
                'input_tokens2': decode(input_ids2),
                'mlm_labels1': decode(targets1),
                'mlm_labels2': decode(targets2),
                'orig_input_tokens1': decode(orig_input_ids1),
                'orig_input_tokens2': decode(orig_input_ids2),
            }

            outputs.append(sample)
        return outputs


@register('bert_cl_single')
class BertCLSingleProcessor(BertMaskedLMProcessor):

    def collate(self, samples):
        input_ids = []
        targets = []
        for sample in samples:
            with numpy_seed(self.args.seed, self.epoch, sample['id']):
                masked_inputs, masked_tokens = self.generate_mask(
                    sample['input_ids'], sample['word_begin_mask'],
                )
            masked_inputs = [self.args.cls_index] + masked_inputs + [self.args.sep_index]
            masked_tokens = [self.args.pad_index] + masked_tokens + [self.args.pad_index]
            input_ids.append(masked_inputs)
            targets.append(masked_tokens)

        input_ids = torch.tensor(self.pad(input_ids), dtype=torch.long)
        targets = torch.tensor(self.pad(targets), dtype=torch.long)

        orig_input_ids = torch.tensor(
            self.pad([[self.args.cls_index] + sample['input_ids'] + [self.args.sep_index] for sample in samples]),
            dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,
            },
            'orig_input_ids': orig_input_ids,
            'target': targets,
            'ntokens': input_ids.numel(),
        }
        return batch

    def sanity_check(self, inputs: List[dict]):
        batch = self.collate(inputs)

        def decode(ids):
            return ' '.join(self.decode(ids)).replace(self.args.pad_word, '_')

        outputs = []
        for i in range(len(inputs)):
            input_ids = batch['net_input']['input_ids'][i].tolist()
            orig_input_ids = batch['orig_input_ids'][i].tolist()
            targets = batch['target'][i].tolist()

            sample = {
                'id': batch['id'][i],
                'input_tokens': decode(input_ids),
                'target': decode(targets),
                'orig_input_tokens': decode(orig_input_ids),
            }

            outputs.append(sample)
        return outputs


@register('bert_cl_single_prompt')
class BertCLSinglePromptProcessor(BertMaskedLMProcessor):

    def numerize(self, inputs: dict):
        results = {
            'id': inputs['id'],
            'input_ids': self.numerize_tokens(inputs['content'][:self.args.max_len - self.special_tokens]),
            'prompt_ids': self.numerize_tokens(inputs['prompt_tokens']),
            'word_begin_mask': inputs['word_begin_mask'][:self.args.max_len - self.special_tokens]
            if self.args.mask_whole_words else None
        }
        return results

    def collate(self, samples):
        input_ids = []
        targets = []
        for sample in samples:
            with numpy_seed(self.args.seed, self.epoch, sample['id']):
                masked_inputs, masked_tokens = self.generate_mask(
                    sample['input_ids'], sample['word_begin_mask'],
                )
            masked_inputs = [self.args.cls_index] + sample['prompt_ids'] + masked_inputs + [self.args.sep_index]
            masked_tokens = [self.args.pad_index] * (len(sample['prompt_ids']) + 1) + masked_tokens + [
                self.args.pad_index]
            input_ids.append(masked_inputs)
            targets.append(masked_tokens)

        input_ids = torch.tensor(self.pad(input_ids), dtype=torch.long)
        targets = torch.tensor(self.pad(targets), dtype=torch.long)

        orig_input_ids = torch.tensor(
            self.pad([[self.args.cls_index] + sample['input_ids'] + [self.args.sep_index] for sample in samples]),
            dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,
            },
            'orig_input_ids': orig_input_ids,
            'target': targets,
            'ntokens': input_ids.numel(),
        }
        return batch
