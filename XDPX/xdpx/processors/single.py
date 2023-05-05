import torch
from . import register, Processor


@register('single')
class SingleProcessor(Processor):
    def numerize(self, inputs: dict):
        results = {
            'id': inputs['id'],
            'tokens': self.numerize_tokens(inputs['tokens'])[:self.args.max_len],
        }
        return results

    def text_length(self, sample):
        return len(sample['tokens'])
    
    def collate(self, samples):
        tokens = torch.tensor(self.pad([sample['tokens'] for sample in samples]), dtype=torch.long)
        mask = torch.ne(tokens, self.args.pad_index)
        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'tokens': tokens,
                'mask': mask,
            },
            'ntokens': tokens.numel(),
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

    def sanity_check(self, inputs):
        batch = self.collate(inputs)

        outputs = []
        for i in range(len(inputs)):
            tokens = ' '.join(self.decode(batch['net_input']['tokens'][i].tolist()))
            tokens = tokens.replace(self.args.pad_word, '_')
            output = {
                'id': batch['id'][i],
                'tokens': tokens,
                'mask': batch['net_input']['mask'][i].tolist(),
            }
            if 'target' in batch:
                output['target'] = str(batch['target'][i].item())
            outputs.append(output)
        return outputs


@register('single_with_logits')
class SingleWithLogitsProcessor(SingleProcessor):
    def numerize(self, inputs: dict):
        if not inputs:
            return None
        results = super(SingleWithLogitsProcessor, self).numerize(inputs)
        if 'logits' in inputs:
            results['logits'] = inputs['logits']
        return results

    def collate(self, samples):
        batch = super(SingleWithLogitsProcessor, self).collate(samples)
        if 'logits' in samples[0]:
            logits = torch.FloatTensor([sample['logits'] for sample in samples])
            batch.update({'logits': logits})
        return batch

