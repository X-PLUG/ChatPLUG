import json
import torch
from . import register, Profile


@register('bert_classification')
class ProfileBertClassification(Profile):
    def model_config(self):
        return dict(
            model='bert_classification',
            processor='bert_single',
            pad_index=0,
            cls_index=101,
            sep_index=102,
            num_classes=2,
            vocab_size=21128,
            max_len=40,
            pretrained_model=None,
            **json.load(open('tests/sample_data/config.json')),
        )

    def create_fake_data(self):
        batch_size = 32
        seq_len = 30
        return dict(
            input_ids=torch.randint(107, 21128, (batch_size, seq_len)),
    )
