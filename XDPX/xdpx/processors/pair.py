import torch
from typing import List
from . import register, Processor


@register('pair')
class PairProcessor(Processor):
    def numerize(self, inputs: dict):
        results = {
            'id': inputs['id'], 
            'tokens1': self.numerize_tokens(inputs['tokens1'])[:self.args.max_len], 
            'tokens2': self.numerize_tokens(inputs['tokens2'])[:self.args.max_len],
        }
        return results

    def text_length(self, sample):
        return max(len(sample['tokens1']), len(sample['tokens2']))

    def collate(self, samples: List[dict]):
        tokens1 = torch.tensor(self.pad([sample['tokens1'] for sample in samples]), dtype=torch.long)
        tokens2 = torch.tensor(self.pad([sample['tokens2'] for sample in samples]), dtype=torch.long)
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

    @staticmethod
    def token_stream(data):
        for sample in data:
            for tokens in (sample['tokens1'], sample['tokens2']):
                for token in tokens:
                    yield token
    
    @staticmethod
    def target_stream(data):
        for sample in data:
            if 'target' in sample:
                yield sample['target']

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


@register('pair_with_logits')
class PairWithLogitsProcessor(PairProcessor):
    def numerize(self, inputs: dict):
        results = super().numerize(inputs)
        if 'logits' in inputs:
            results['logits'] = inputs['logits']
        return results

    def collate(self, samples):
        batch = super().collate(samples)
        if 'logits' in samples[0]:
            logits = torch.tensor([sample['logits'] for sample in samples], dtype=torch.float)
            batch.update({'logits': logits})
        return batch
