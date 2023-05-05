import json
import math
import torch
import numpy as np
from . import register, Profile


@register('mgimn')
class ProfileMGIMN(Profile):
    def model_config(self):
        return dict(
            model='mgimn',
            processor='fewshot',
            pad_index=0,
            cls_index=101,
            sep_index=102,
            num_classes=21128,
            vocab_size=21128,
            max_len=30,
            pretrained_model=None,
            attention_probs_dropout_prob=0.1,
            directionality="bidi",
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            hidden_size=256,
            initializer_range=0.02,
            intermediate_size=1024,
            max_position_embeddings=512,
            num_attention_heads=4,
            num_hidden_layers=4,
            type_vocab_size=2,
            graident_checkpointing=False,
            gc_skip_interval=6,
            multi_level_matching=True,
            instance_level=False,
            class_level=True,
            episode_level=False,
            pooling="self_attention",
            metrics="cosine"

        )

    def create_fake_data(self):
        q_batch_size = 1
        max_len = 30
        N = 10
        K = 3

        query_batch = torch.randint(107, 21128, (1, q_batch_size, max_len))
        support_batch = torch.randint(107, 21128, (1, N * K, max_len))
        slabel_batch = torch.randint(0, N, (1, N * K))

        return dict(
            query=query_batch,
            support=support_batch,
            support_labels=slabel_batch
        )
