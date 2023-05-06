import math
import json
import torch
import numpy as np
from . import register, Profile


@register('bert_lm')
class ProfileBertLM(Profile):
    def model_config(self):
        return dict(
            model='bert_lm',
            processor='bert_lm',
            pad_index=0,
            cls_index=101,
            sep_index=102,
            mask_index=103,
            num_classes=21128,
            vocab_size=21128,
            max_len=128,
            pretrained_model=None,
            **json.load(open('tests/sample_data/config.json')),
        )

    def create_fake_data(self):
        batch_size = 32
        seq_len = 128
        input_ids = torch.randint(107, 21128, (batch_size, seq_len))
        num_masks = math.ceil(0.15 * seq_len)
        for i in range(batch_size):
            input_ids[i, torch.tensor(np.random.choice(seq_len, num_masks, replace=False))] = 103
        return dict(
            input_ids=input_ids,
        )
