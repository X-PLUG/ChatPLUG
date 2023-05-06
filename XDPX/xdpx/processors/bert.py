import torch
from typing import List
from . import register, Processor
from .single import SingleProcessor
from .pair import PairProcessor
from xdpx.options import Argument
from xdpx.tokenizers import tokenizers
from xdpx.tokenizers.bert import BertTokenizer


class BertProcessor(Processor):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('cls_index', type=int),
            Argument('sep_index', type=int),
        )
        options.add_global_constraint(lambda args: not hasattr(args, 'tokenizer') or issubclass(tokenizers[args.tokenizer], BertTokenizer))
        options.add_global_constraint(lambda args: args.max_len >= 3)
    
    def text_length(self, sample):
        return len(sample['input_ids'])

    def sanity_check(self, inputs: List[dict]):
        batch = self.collate(inputs)
        
        outputs = []
        for i in range(len(inputs)):
            input_tokens = ' '.join(self.decode(batch['net_input']['input_ids'][i].tolist()))
            input_tokens = input_tokens.replace(self.args.pad_word, '_')
            outputs.append({
                'id': batch['id'][i],
                'input_tokens': input_tokens,
                'target': str(batch['target'][i].item()),
            })
        return outputs


    @property
    def special_tokens(self):
        """number of special tokens added to the sequence, like "[CLS]", "[SEP]" """
        return 2

    def collate(self, samples):
        input_ids = torch.tensor(self.pad([sample['input_ids'] for sample in samples]), dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,
            },
            'ntokens': input_ids.numel(),
        }
        try:
            token_type_ids = torch.tensor(self.pad([sample['token_type_ids'] for sample in samples]), dtype=torch.long)
            batch['net_input']['token_type_ids'] = token_type_ids
        except KeyError:
            ...

        try:
            target = torch.tensor([sample['target'] for sample in samples], dtype=torch.long)
            batch.update({'target': target})
        except KeyError:
            ...
        return batch


@register('bert_single')
class BertSingleProcessor(BertProcessor, SingleProcessor):
    def numerize(self, inputs: dict):
        text = self.numerize_tokens(inputs['tokens'])
        results = {
            'id': inputs['id'], 
            'input_ids': [self.args.cls_index, *text[:self.args.max_len - 2], self.args.sep_index],
        }
        return results


@register('bert_pair')
class BertPairProcessor(BertProcessor, PairProcessor):
    def numerize(self, inputs: dict):
        tokens1 = self.numerize_tokens(inputs['tokens1'])
        tokens2 = self.numerize_tokens(inputs['tokens2'])
        max_len = self.args.max_len - 3
        half_max_len, extra = divmod(max_len, 2)
        if len(tokens1) + len(tokens2) > max_len:
            if len(tokens1) > half_max_len and len(tokens2) > half_max_len:
                tokens1 = tokens1[:half_max_len + extra]
                tokens2 = tokens2[:half_max_len]
            elif len(tokens1) > half_max_len:
                tokens1 = tokens1[:max_len - len(tokens2)]
            else:
                tokens2 = tokens2[:max_len - len(tokens1)]
        results = {
            'id': inputs['id'], 
            'input_ids': [self.args.cls_index, *tokens1, self.args.sep_index, *tokens2, self.args.sep_index],
            'token_type_ids': [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)
        }
        return results


@register('bert_pair_siamese')
class BertPairSiameseProcessor(BertPairProcessor):
    def numerize(self, inputs: dict):
        tokens1 = self.numerize_tokens(inputs['tokens1'])
        tokens2 = self.numerize_tokens(inputs['tokens2'])
        max_len = self.args.max_len - 2
        tokens1 = tokens1[:max_len]
        tokens2 = tokens2[:max_len]
        results = {
            'id': inputs['id'],
            'tokens1': [self.args.cls_index, *tokens1, self.args.sep_index],
            'tokens2': [self.args.cls_index, *tokens2, self.args.sep_index],
        }
        return results

    def collate(self, samples):
        input_ids_1 = torch.tensor(self.pad([sample['tokens1'] for sample in samples]), dtype=torch.long)
        input_ids_2 = torch.tensor(self.pad([sample['tokens2'] for sample in samples]), dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids_1': input_ids_1,
                'input_ids_2': input_ids_2,
            },
            'ntokens': input_ids_1.numel() + input_ids_2.numel(),
        }
        try:
            target = torch.tensor([sample['target'] for sample in samples], dtype=torch.long)
            batch.update({'target': target})
        except KeyError:
            ...
        return batch

    def text_length(self, sample):
        return max(len(sample['tokens1']), len(sample['tokens2']))

    def sanity_check(self, inputs: List[dict]):
        batch = self.collate(inputs)

        outputs = []
        for i in range(len(inputs)):
            input_tokens_1 = ' '.join(self.decode(batch['net_input']['input_ids_1'][i].tolist()))
            input_tokens_1 = input_tokens_1.replace(self.args.pad_word, '_')
            input_tokens_2 = ' '.join(self.decode(batch['net_input']['input_ids_2'][i].tolist()))
            input_tokens_2 = input_tokens_2.replace(self.args.pad_word, '_')
            outputs.append({
                'id': batch['id'][i],
                'input_tokens_1': input_tokens_1,
                'input_tokens_2': input_tokens_2,
                'target': str(batch['target'][i].item()),
            })
        return outputs