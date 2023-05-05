import torch
import numpy as np
from collections import deque
from xdpx.utils import numpy_seed
from xdpx.options import Argument
from . import register
from .bert_lm import BertMaskedLMProcessor


@register('session_corpus')
class SessionCorpusProcessor(BertMaskedLMProcessor):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('max_messages', default=8, validate=lambda val: val >= 2),
        )
        options.add_global_constraint(
            lambda args: not hasattr(args, 'batch_size') or args.batch_size // args.distributed_world_size > 1
        )
        options.set_default('mask_whole_words', False)
        options.add_global_constraint(
            lambda args: (args.mask_whole_words is False, 'WWM not implemented.')
        )

    def numerize(self, inputs: dict):
        results = {
            'id': inputs['id'],
            'sender': inputs['sender'],
            'content': [self.numerize_tokens(tokens)[:self.args.max_len - self.special_tokens]
                        for tokens in inputs['content']],
        }
        return results

    def text_length(self, sample):
        return max(map(len, sample['content'])) + self.special_tokens

    def collate(self, samples):
        indices = []  # indices of shuffled messages
        candidates = deque()  # messages to shuffle
        all_contents = []
        sender_mask = []
        for sample in samples:
            m = self.args.max_messages
            sample_id = sample['id']
            session_size = len(sample['content'])

            begin = 0
            if session_size > m:
                with numpy_seed(self.args.seed, self.epoch, sample_id):
                    begin = np.random.randint(session_size - m)
            contents = sample['content'][begin: begin + m]
            senders = sample['sender'][begin: begin + m]
            sender_idx = [i for i, sender in enumerate(senders) if sender == 1]
            with numpy_seed(self.args.seed, self.epoch, sample_id):
                index = np.random.choice(sender_idx)

            masked_contents = []
            for i, content in enumerate(contents):
                with numpy_seed(self.args.seed, self.epoch, sample_id, i):
                    masked_inputs, masked_tokens = self.generate_mask(content)
                masked_inputs = [self.args.cls_index] + masked_inputs + [self.args.sep_index]
                masked_tokens = [self.args.pad_index] + masked_tokens + [self.args.pad_index]
                masked_contents.append((masked_inputs, masked_tokens))

            indices.append(index)
            candidates.append(masked_contents[index])
            all_contents.append(masked_contents)
            sender_mask.append(senders)
        candidates.rotate(1)
        for contents, index in zip(all_contents, indices):
            contents[index] = candidates.popleft()
        assert not candidates

        tokens = torch.tensor(self.pad([content[0] for contents in all_contents for content in contents]),
                              dtype=torch.long)
        targets = torch.tensor(self.pad([content[1] for contents in all_contents for content in contents]),
                               dtype=torch.long)
        session_sizes = torch.tensor(list(map(len, all_contents)), dtype=torch.long)
        sender_mask = torch.tensor(self.pad(sender_mask)).eq(1)
        cls_targets = torch.tensor(indices, dtype=torch.long)
        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': tokens,
            },
            'session_size': session_sizes,
            'sender_mask': sender_mask,
            'ntokens': tokens.numel(),
            'target': targets,
            'cls_target': cls_targets,
        }
        return batch

    def sanity_check(self, inputs):
        batch = self.collate(inputs)

        outputs = []
        sample_ids = batch['id']
        session_sizes = batch['session_size'].tolist()
        cls_targets = batch['cls_target'].tolist()
        batch_targets = batch['target'].tolist()
        batch_tokens = batch['net_input']['input_ids'].tolist()
        buffer = []
        for tokens, targets in zip(batch_tokens, batch_targets):
            tokens = ' '.join(self.decode(tokens))
            tokens = tokens.replace(' ' + self.args.pad_word, '')
            targets = ' '.join(self.decode(targets)).replace(self.args.pad_word, '_')
            buffer.append((tokens, targets))
            if len(buffer) == session_sizes[0]:
                outputs.append({
                    'id': sample_ids.pop(0),
                    'cls_target': str(cls_targets.pop(0)),
                    'tokens': '\n' + '\n'.join(buf[0] + ' | ' + buf[1] for buf in buffer),
                })
                session_sizes.pop(0)
                buffer.clear()
        assert not cls_targets and not session_sizes and not sample_ids
        return outputs
