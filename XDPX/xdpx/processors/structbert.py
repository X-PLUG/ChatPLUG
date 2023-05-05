import torch
from collections import deque
from typing import List
from xdpx.utils import numpy_seed
from . import register
from .bert_lm import BertMaskedLMProcessor


@register('structbert')
class StructBertProcessor(BertMaskedLMProcessor):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.add_global_constraint(lambda args: not hasattr(args, 'min_word_len') or args.min_word_len > 2)

    def numerize(self, inputs: dict):
        results = super().numerize(inputs)
        if results['word_begin_mask']:
            half_words = sum(results['word_begin_mask']) // 2
            word_cnt = i = 0
            for i, mask in enumerate(results['word_begin_mask']):
                word_cnt += mask
                if word_cnt > half_words:
                    break
            assert i
            results['center'] = i
        else:
            results['center'] = len(results['input_ids']) // 2
        return results

    @property
    def special_tokens(self):
        return 3

    def collate(self, samples):
        input_ids = []
        targets = []
        masked_samples = []

        pattern = [0, 1, 2]
        actions = [pattern[i % 3] for i in range(len(samples))]
        for sample in samples:
            with numpy_seed(self.args.seed, self.epoch, sample['id']):
                masked_inputs, masked_tokens = self.generate_mask(
                    sample['input_ids'], sample['word_begin_mask'],
                )
            masked_samples.append((masked_inputs, masked_tokens, sample['center']))
        nsp = deque(i for i, x in enumerate(actions) if x == 0)
        nsp.rotate(1)
        if len(nsp) == 1:
            actions[nsp[0]] = 1

        for action, sample in zip(actions, masked_samples):
            masked_inputs, masked_tokens, center = sample
            if action == 1:
                masked_inputs = [*masked_inputs[:center], self.args.sep_index, *masked_inputs[center:]]
                masked_tokens = [*masked_tokens[:center], self.args.pad_index, *masked_tokens[center:]]
            elif action == 2:
                masked_inputs = [*masked_inputs[center:], self.args.sep_index, *masked_inputs[:center]]
                masked_tokens = [*masked_tokens[center:], self.args.pad_index, *masked_tokens[:center]]
            else:
                candidate = nsp.popleft()
                cand_masked_inputs, cand_masked_tokens, cand_center = masked_samples[candidate]
                masked_inputs = [*masked_inputs[:center], self.args.sep_index, *cand_masked_inputs[cand_center:]]
                masked_tokens = [*masked_tokens[:center], self.args.pad_index, *cand_masked_tokens[cand_center:]]
            masked_inputs = masked_inputs[:self.args.max_len - 2]
            masked_tokens = masked_tokens[:self.args.max_len - 2]
            masked_inputs = [self.args.cls_index, *masked_inputs, self.args.sep_index]
            masked_tokens = [self.args.pad_index, *masked_tokens, self.args.pad_index]
            input_ids.append(masked_inputs)
            targets.append(masked_tokens)
        input_ids = torch.tensor(self.pad(input_ids), dtype=torch.long)
        targets = torch.tensor(self.pad(targets), dtype=torch.long)
        cls_targets = torch.tensor(actions, dtype=torch.long)

        # no token_type_ids here to save memory transferred to GPU. Almost no difference to cls_acc.
        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,
            },
            'orig_input_ids': [sample['input_ids'] for sample in samples],
            'ntokens': input_ids.numel(),
            'target': targets,
            'cls_target': cls_targets,
        }
        return batch

    def sanity_check(self, inputs: List[dict]):
        self.epoch = 2
        batch = self.collate(inputs)

        def decode(ids):
            return ' '.join(self.decode(ids)).replace(self.args.pad_word, '_')

        outputs = []
        for i in range(len(inputs)):
            input_ids = batch['net_input']['input_ids'][i].tolist()
            targets = batch['target'][i].tolist()
            cls_target = batch['cls_target'][i].tolist()

            sample = {
                'id': batch['id'][i],
                'input_tokens': decode(input_ids),
                'orig_input_tokens': decode(batch['orig_input_ids'][i]),
                'target': decode(targets),
                'cls_target': cls_target,
            }
            if inputs[i]['word_begin_mask']:
                word_begin_mask = ' '.join(map(lambda x: str(int(x)), inputs[i]['word_begin_mask']))
                sample['word_begin_mask'] = word_begin_mask

            outputs.append(sample)
        return outputs
