import math
import torch
import numpy as np
from functools import lru_cache
from typing import List, Optional, Union
from xdpx.options import Argument
from xdpx.utils import numpy_seed, cache_file
from . import register
from .bert import BertProcessor


@register('bert_lm')
class BertMaskedLMProcessor(BertProcessor):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('mask_prob', default=0.15, doc='probability of replacing a token with mask'),
            Argument('leave_unmasked_prob', default=0.1, doc='probability that a masked token is unmasked'),
            Argument('random_token_prob', default=0.1, doc='probability of replacing a token with a random token'),
            Argument('max_predictions', default=0.33, type=Union[int, float],
                     doc='max number of targets in each sample, use float nunberto set a relative value'),
            Argument('mask_whole_words', default=True),
            Argument('mask_index', type=int),
            Argument('freq_weighted_replacement', doc='path to token count file if not None', children=[
                Argument('token_default_count', required=True, type=int),
                Argument('vocab_sample_alpha', default=0.5, doc='Exponent for transforming word counts to probabilities (~word2vec sampling)'),
            ]),
            domain='mask_lm_task',
        )
    
    def __init__(self, args):
        super().__init__(args)
        self._vocab_weights = None

    @lru_cache(maxsize=8)
    def transform_weight(self, x, alpha):
        return float(x) ** alpha

    @property
    def vocab_weights(self):
        if self._vocab_weights is None:
            if self.args.freq_weighted_replacement:
                token_weight_file = cache_file(self.args.freq_weighted_replacement)
                from xdpx.utils import io
                default_value = self.transform_weight(self.args.token_default_count, self.args.vocab_sample_alpha)
                self._vocab_weights = [default_value] * self.args.vocab_size
                with io.open(token_weight_file) as f:
                    for line in f:
                        token, weight = line.split()
                        token_id = self.numerize_tokens(token)
                        weight = self.transform_weight(weight, self.args.vocab_sample_alpha)
                        self._vocab_weights[token_id] = weight
                self._vocab_weights[:106] = [0.] * 106  # don't sample special tokens
                self._vocab_weights = self.normalize_mask_weights(self._vocab_weights)
        return self._vocab_weights

    def numerize(self, inputs: dict):
        results = {
            'id': inputs['id'], 
            'input_ids': self.numerize_tokens(inputs['content'][:self.args.max_len - self.special_tokens]),
            'word_begin_mask': inputs['word_begin_mask'][:self.args.max_len - self.special_tokens]
            if self.args.mask_whole_words else None
        }
        return results

    @staticmethod
    def normalize_mask_weights(mask_weights):
        total = sum(mask_weights)
        return [w / total for w in mask_weights]

    def text_length(self, sample):
        return len(sample['input_ids']) + self.special_tokens

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
        
        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,
            },
            'orig_input_ids': [sample['input_ids'] for sample in samples],
            'ntokens': input_ids.numel(),
            'target': targets,
        }
        return batch
        
    def sanity_check(self, inputs: List[dict]):
        batch = self.collate(inputs)

        def decode(ids):
            return ' '.join(self.decode(ids)).replace(self.args.pad_word, '_')
        
        outputs = []
        for i in range(len(inputs)):
            input_ids = batch['net_input']['input_ids'][i].tolist()
            targets = batch['target'][i].tolist()

            sample = {
                'id': batch['id'][i],
                'input_tokens': decode(input_ids),
                'target': decode(targets),
            }
            if inputs[i]['word_begin_mask']:
                word_begin_mask = ' '.join(map(lambda x: str(int(x)), inputs[i]['word_begin_mask']))
                sample['word_begin_mask'] = word_begin_mask

            outputs.append(sample)
        return outputs

    def generate_mask(self, tokens: List[int], word_begin_mask: Optional[List[bool]] = None,
                      mask_weights: Optional[List[float]] = None):
        """
        Reference: https://github.com/pytorch/fairseq/blob/master/fairseq/data/mask_tokens_dataset.py#L98
        """
        sz = len(tokens)
        # assert self.args.mask_index not in tokens  # comment out this to speed up in runtime
        
        tokens = np.asarray(tokens)
        word_lens = None
        if self.args.mask_whole_words:
            word_begin_mask = np.asarray(word_begin_mask)
            word_begins_idx = np.argwhere(word_begin_mask).squeeze(1)
            words = np.split(word_begin_mask, word_begins_idx)[1:]
            word_lens = list(map(len, words))
            sz = len(words)
            assert sz == len(word_begins_idx)
        
        mask = np.full(sz, False)
        num_mask = int(
            # add a random number for probabilistic rounding
            self.args.mask_prob * sz + np.random.rand()
        )
        max_nmasks = self.args.max_predictions if isinstance(self.args.max_predictions, int) else \
            int(math.ceil(len(tokens) * self.args.max_predictions))
        num_mask = min(max_nmasks, num_mask, len(tokens) - 1)
        mask[np.random.choice(sz, num_mask, replace=False, p=mask_weights)] = True
    
        # get masked tokens as targets for LM before unmasking
        if self.args.mask_whole_words:
            target_mask = np.repeat(mask, word_lens)
        else:
            target_mask = mask
        masked_tokens = np.full(len(target_mask), self.args.pad_index)
        masked_tokens[target_mask] = tokens[target_mask]

        # decide unmasking and random replacement
        rand_or_unmask_prob = self.args.random_token_prob + self.args.leave_unmasked_prob
        if rand_or_unmask_prob > 0.0:
            rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
            if self.args.random_token_prob == 0.0:
                unmask = rand_or_unmask
                rand_mask = None
            elif self.args.leave_unmasked_prob == 0.0:
                unmask = None
                rand_mask = rand_or_unmask
            else:
                unmask_prob = self.args.leave_unmasked_prob / rand_or_unmask_prob
                decision = np.random.rand(sz) < unmask_prob
                unmask = rand_or_unmask & decision
                rand_mask = rand_or_unmask & (~decision)
        else:
            unmask = rand_mask = None

        if unmask is not None:
            mask = mask ^ unmask

        if self.args.mask_whole_words:
            mask = np.repeat(mask, word_lens)

        new_tokens = np.copy(tokens)
        new_tokens[mask] = self.args.mask_index
        if rand_mask is not None:
            num_rand = rand_mask.sum()
            if num_rand > 0:
                if self.args.mask_whole_words:
                    rand_mask = np.repeat(rand_mask, word_lens)
                    num_rand = rand_mask.sum()

                new_tokens[rand_mask] = np.random.choice(
                    self.args.vocab_size,
                    num_rand,
                    p=self.vocab_weights,
                )
    
        return new_tokens.tolist(), masked_tokens.tolist()


