import torch
from typing import List
from . import register, Processor
from .single import SingleProcessor
from .pair import PairProcessor
from xdpx.options import Argument
from xdpx.tokenizers import tokenizers
from xdpx.tokenizers.bert import BertTokenizer
from .bert import BertProcessor, BertPairProcessor


@register('sbert_nli')
class SBertNLIProcessor(BertProcessor, PairProcessor):

    def numerize(self, inputs: dict):
        results = {
            'id': inputs['id'],
            'tokens1': self.numerize_tokens(inputs['tokens1'])[:self.args.max_len],
            'tokens2': self.numerize_tokens(inputs['tokens2'])[:self.args.max_len],
        }
        return results


    def clip(self, sent):
        return [self.args.cls_index, *sent[:min(len(sent), self.args.max_len - 2)], self.args.sep_index]

    def text_length(self, sample):
        return max(len(sample['tokens1']), len(sample['tokens2']))

    def collate(self, samples: List[dict]):
        tokens1 = torch.tensor(self.pad([self.clip(sample['tokens1']) for sample in samples]), dtype=torch.long)
        tokens2 = torch.tensor(self.pad([self.clip(sample['tokens2']) for sample in samples]), dtype=torch.long)
        mask1 = torch.ne(tokens1, self.args.pad_index)
        mask2 = torch.ne(tokens2, self.args.pad_index)
        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'tokens1': tokens1,
                'tokens2': tokens2,
                'mask1': mask1,
                'mask2': mask2,
            },
            'ntokens': tokens1.numel() + tokens2.numel(),
        }
        try:
            target = torch.tensor([sample['target'] for sample in samples], dtype=torch.long)
            batch.update({'target': target})
        except KeyError:
            ...
        return batch

    def sanity_check(self, inputs: List[dict]):
        batch = self.collate(inputs)

        def get_net_input(name, i):
            return ' '.join(self.decode(batch['net_input'][name][i].tolist())).replace(self.args.pad_word, '_')

        outputs = []
        for i in range(len(inputs)):
            output = {
                'id': batch['id'][i],
                'tokens1': get_net_input('tokens1', i),
                'tokens2': get_net_input('tokens2', i),
                'mask1': batch['net_input']['mask1'][i].tolist(),
                'mask2': batch['net_input']['mask2'][i].tolist(),
            }
            if 'target' in batch:
                output['target'] = str(batch['target'][i].item())
            outputs.append(output)
        return outputs
