import torch
import torch.nn as nn
import torch.nn.functional as F
from xdpx.models import register
from xdpx.options import Argument
from xdpx.models.fewshot import BertFewshotBase
from xdpx.modules.projections import LinearProjection

'''
this file includes prototypical-based few-shot models 
these models have no token-level interaction between query and support instance. 
'''


@register('protonet')
class BertProtoNet(BertFewshotBase):
    @staticmethod
    def register(options):
        options.register(
            Argument('temperature', default=0.05, doc='Temperature for the contrastive loss'),
            Argument('proj_hidden_size', default=768),
        )

    def __init__(self, args):
        super().__init__(args)
        self.similarity = lambda x, y: F.cosine_similarity(x, y, dim=-1, eps=1e-4) / self.args.temperature
        self.emb_proj = None
        if args.proj_hidden_size != args.hidden_size:
            self.emb_proj = LinearProjection(args.hidden_size, args.proj_hidden_size, activation='relu')

    def forward(self, query_input_ids, support_input_ids,
                query_input_ids_with_prompt=None,
                support_input_ids_with_prompt=None,
                masked_input_ids=None, label_ids=None, masked_tokens=None,
                query_emb=None, support_emb=None, **kwargs
                ):
        is_training = support_emb is None
        prompt_masked_logits = None
        prompt_logits = None
        masked_logits = None
        N, K = support_input_ids.shape[:2]
        support_shot_masks = torch.any(support_input_ids.ne(self.args.pad_index), dim=-1)

        if is_training:
            support_emb = self.bert_forward(support_input_ids.reshape(N * K, -1), **kwargs)[1]  # NK * D
            support_emb = support_emb.reshape(N, K, self.args.hidden_size)  # N * K * D
        else:  # when in inference phrase, support way and shot are infered dynamiclly
            N = support_emb.shape[0]
            K = support_emb.shape[1]
            support_shot_masks = torch.ones(N, K, dtype=torch.bool,
                                            device=support_emb.device)

        if self.emb_proj is not None:
            support_emb = self.emb_proj(support_emb)

        sum_prototypes = torch.sum(support_emb * support_shot_masks.unsqueeze(2), dim=1)  # N * D

        query_emb = None
        if query_input_ids is None:
            z1, z2 = self.split_support_set(support_emb, sum_prototypes,
                                            support_shot_masks, N, K)

            logits = proto_logits = self.similarity(z1, z2).reshape(-1, N)  # NK * N

            if label_ids is not None and support_input_ids_with_prompt is not None:
                prompt_logits, prompt_masked_logits = self.compute_prompt_logits(support_input_ids_with_prompt,
                                                                                 label_ids)
                logits += prompt_logits

        else:
            Q = query_input_ids.shape[0]
            query_emb = self.bert_forward(query_input_ids, **kwargs)[1]  # Q * D
            if self.emb_proj is not None:
                query_emb = self.emb_proj(query_emb)

            prototypes = sum_prototypes / torch.sum(support_shot_masks, dim=1, keepdim=True)

            z1 = query_emb.unsqueeze(1).repeat(1, N, 1)  # Q * N * D
            z2 = prototypes.unsqueeze(0).repeat(Q, 1, 1)  # Q * N * D
            logits = proto_logits = self.similarity(z1, z2).reshape(-1, N)  # Q * N

            if label_ids is not None and query_input_ids_with_prompt is not None:
                prompt_logits, prompt_masked_logits = self.compute_prompt_logits(query_input_ids_with_prompt, label_ids)
                logits += prompt_logits

        if masked_input_ids is not None and masked_tokens is not None:
            seq_features = self.bert_forward(masked_input_ids)[0]
            masked_logits = self.cls['predictions'](seq_features, masked_tokens=masked_tokens)

        return logits, support_emb, query_emb, proto_logits, prompt_logits, prompt_masked_logits, masked_logits

    def compute_prompt_logits(self, prompt_input_ids, label_ids):
        seq_features = self.bert_forward(prompt_input_ids)[0]
        masked_tokens = prompt_input_ids.eq(self.args.mask_index)
        seq_logits = prompt_seq_logits = self.cls['predictions'](seq_features, masked_tokens=masked_tokens)
        seq_logits = seq_logits.view(-1, seq_logits.size(-1))
        batch_size = prompt_input_ids.shape[0]
        vocab_size = seq_logits.shape[-1]
        label_count = label_ids.shape[0]

        prediction_probs = F.softmax(seq_logits, dim=-1)
        prediction_probs = torch.reshape(
            prediction_probs, shape=[batch_size, -1, vocab_size])

        probs = torch.ones(size=[batch_size, label_count], device=prompt_input_ids.device)
        # Calculate joint distribution of candidate labels
        for index in range(self.args.label_length):
            probs *= prediction_probs[:, index, label_ids[:, index]]

        prompt_logits = probs / self.args.temperature
        return prompt_logits, prompt_seq_logits

    def split_support_set(self, support_emb, sum_prototypes, support_shot_masks, N, K):
        expanded_support_emb = support_emb.unsqueeze(2).repeat(1, 1, N, 1)  # N * K * N * D
        expanded_sum_prototypes = sum_prototypes.repeat(N, K, 1, 1)  # N * K * N * D
        embedding_masks = torch.eye(N, dtype=torch.bool, device=support_emb.device) \
            .unsqueeze(1).unsqueeze(-1).repeat(1, K, 1, 1)  # N * K * N * 1

        denominator_masks = 1 - torch.eye(N * K, device=support_emb.device)
        denominator_masks = denominator_masks.reshape(N, K, N, K)
        shot_masks = support_shot_masks.flatten()
        shot_masks = shot_masks.unsqueeze(1) * shot_masks.unsqueeze(0)  # NK * NK
        shot_masks = shot_masks.reshape(N, K, N, K)
        denominator = torch.sum(denominator_masks * shot_masks, dim=-1, keepdim=True) + 1e-7  # N x K x N x 1

        z1 = expanded_support_emb
        z2 = (expanded_sum_prototypes - embedding_masks * expanded_support_emb) / denominator
        return z1, z2
