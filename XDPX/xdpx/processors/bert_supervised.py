import torch
from typing import List
from xdpx.utils import numpy_seed
from . import register
from .bert_lm import BertMaskedLMProcessor


@register('bert_supervised')
class BertSupervisedProcessor(BertMaskedLMProcessor):

    def numerize(self, inputs: dict):
        if self.text_length(inputs) > self.max_len:
            return None
        
        id1 = self.numerize_tokens(inputs['tokens1'])
        id2 = self.numerize_tokens(inputs['tokens2'])
        
        results = {
            'id': inputs['id'],
            'tokens1': id1,
            'tokens2': id2,
            'mask1': inputs['mask1'] if self.args.mask_whole_words else None,
            'mask2': inputs['mask2'] if self.args.mask_whole_words else None,
        }
        return results
    
    def text_length(self, sample):
        return len(sample['tokens1']) + len(sample['tokens2']) + 3
    
    def collate(self, samples):
        input_ids = [] 
        token_type_ids = []
        targets = []
        for sample in samples:
            with numpy_seed(self.args.seed, self.epoch, sample['id']):
                masked_inputs1, masked_tokens1 = self.generate_mask(sample['tokens1'], sample['mask1'])
                masked_inputs2, masked_tokens2 = self.generate_mask(sample['tokens2'], sample['mask2'])
            
            masked_inputs = [self.args.cls_index] + masked_inputs1 + [self.args.sep_index] + masked_inputs2 + [self.args.sep_index]
            masked_tokens = [self.args.pad_index] + masked_tokens1 + [self.args.pad_index] + masked_tokens2 + [self.args.pad_index]
            token_type_id = [0] * (len(masked_inputs1) + 2) + [1] * (len(masked_inputs2) + 1)

            input_ids.append(masked_inputs)
            token_type_ids.append(token_type_id)
            targets.append(masked_tokens)

        input_ids = torch.tensor(self.pad(input_ids), dtype=torch.long)
        targets = torch.tensor(self.pad(targets), dtype=torch.long)
        token_type_ids = torch.tensor(self.pad(token_type_ids), dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
            },
            'orig_input_ids': [sample['tokens1'] + [self.args.sep_index] + sample['tokens2'] for sample in samples],
            'ntokens': input_ids.numel(),
            'target': targets,
        }
        try:
            cls_targets = torch.tensor([sample['target'] for sample in samples], dtype=torch.long)
            batch.update({'cls_target': cls_targets})
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
            token_type_ids = batch['net_input']['token_type_ids'][i].tolist()
            targets = batch['target'][i].tolist()
            cls_target = batch['cls_target'][i].item()

            sample = {
                'id': batch['id'][i],
                'input_tokens': decode(input_ids),
                'token_type_ids': token_type_ids,
                'target': decode(targets),
                'cls_target': cls_target,
            }
            if inputs[i]['mask1']:
                mask1 = ' '.join((map(lambda x: str(int(x)), inputs[i]['mask1'])))
                mask2 = ' '.join((map(lambda x: str(int(x)), inputs[i]['mask2'])))
                sample.update({
                    'word_begin_mask_1': mask1,
                    'word_begin_mask_2': mask2,
                })
            outputs.append(sample)
        return outputs
