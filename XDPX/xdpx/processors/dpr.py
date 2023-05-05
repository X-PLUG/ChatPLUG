import math
import torch
import numpy as np
from typing import List, Optional, Union
from xdpx.options import Argument
from xdpx.utils import numpy_seed
from typing import List, Iterator
from . import register
from .bert_lm import BertMaskedLMProcessor

@register('bert_dpr')
class BertDPRProcessor(BertMaskedLMProcessor):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('max_question_length', default=30),
            Argument('max_passage_length', default=300),
            domain='bert_dpr',
        )

    def __init__(self, args):
        super().__init__(args)

    def numerize(self, inputs: dict):
        question = self.numerize_tokens(inputs['question'])
        passage = self.numerize_tokens(inputs['passage'])
        max_question_len = self.args.max_question_length - 2
        max_passage_len = self.args.max_passage_length - 2
        question = question[:max_question_len]
        passage = passage[:max_passage_len]
        results = {
            'id': inputs['id'],
            'question': question,
            'passage': passage
        }
        return results

    def text_length(self, sample):
        return len(sample['input_ids']) + self.special_tokens

    def mask_sent(self, sid, input_ids, word_begin_mask=None):
        with numpy_seed(self.args.seed, self.epoch, sid):
            masked_inputs, masked_tokens = self.generate_mask(
                input_ids, word_begin_mask,
            )
        masked_inputs = [self.args.cls_index] + masked_inputs + [self.args.sep_index]
        masked_tokens = [self.args.pad_index] + masked_tokens + [self.args.pad_index]
        return masked_inputs, masked_tokens

    def collate(self, samples):
        question_input_ids = torch.tensor(
            self.pad([[self.args.cls_index] + sample['question'] + [self.args.sep_index] for sample in samples]),
            dtype=torch.long)
        passage_input_ids = torch.tensor(
            self.pad([[self.args.cls_index] + sample['passage'] + [self.args.sep_index] for sample in samples]),
            dtype=torch.long)
        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'question_input_ids': question_input_ids,
                'passage_input_ids': passage_input_ids
            },
            'ntokens': question_input_ids.numel() + passage_input_ids.numel(),
        }
        return batch

    def sanity_check(self, inputs: List[dict]):
        batch = self.collate(inputs)

        def decode(ids):
            return ' '.join(self.decode(ids)).replace(self.args.pad_word, '_')

        outputs = []

        for i in range(len(inputs)):
            question_input_ids = batch['net_input']['question_input_ids'][i].tolist()
            passage_input_ids = batch['net_input']['passage_input_ids'][i].tolist()

            sample = {
                'id': batch['id'][i],
                'question_input_ids': decode(question_input_ids),
                'passage_input_ids': decode(passage_input_ids)
            }

            outputs.append(sample)
        return outputs

    @staticmethod
    def target_stream(data: List[dict]) -> Iterator[str]:
        yield 'NotUsedInFewShotTraining'

    def numerize_target(self, sample):
        return {}
