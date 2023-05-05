from xdpx.models.fewshot.proto import BertFewshotBase
import torch
import torch.nn as nn
import torch.nn.functional as F
from xdpx.models import register
from xdpx.options import Argument
from xdpx.modules.projections import LinearProjection
from xdpx.modules.alignments import Alignment


class BertMatchingNetBase(BertFewshotBase):
    @staticmethod
    def register(options):
        options.register(
            Argument('output_dropout_prob', default=0.1),
            Argument('proj_hidden_size', default=768)
        )

    def __init__(self, args):
        super().__init__(args)
        self.emb_proj = None
        if args.proj_hidden_size != args.hidden_size:
            self.emb_proj = LinearProjection(args.hidden_size, args.proj_hidden_size, activation='relu')

    def forward(self, query_input_ids, support_input_ids,
                query_input_ids_with_prompt=None,
                support_input_ids_with_prompt=None,
                masked_input_ids=None, masked_tokens=None,
                query_emb=None, support_emb=None, **kwargs
                ):
        assert query_input_ids is not None, "for BertMatchingNetBase, query_input_ids CANT be None, " \
                                            "if dataset mode is episode, please set auto_sample_query_cnt a positive number"
        is_training = support_emb is None
        N, K = support_input_ids.shape[:2]
        if is_training:
            support_emb = self.bert_forward(support_input_ids.reshape(N * K, -1), **kwargs)[0]  # (NK, max_s_len, D)
            dim = support_emb.shape[-1]
            support_emb = support_emb.reshape(N, K, -1, dim)
            support_shot_masks = torch.any(support_input_ids.ne(self.args.pad_index), dim=-1)  # (N, K)
        else:  # when in inference phrase, support way and shot are infered dynamiclly
            N = support_emb.shape[0]
            K = support_emb.shape[1]
            support_shot_masks = torch.ones(N, K, dtype=torch.bool,
                                            device=support_emb.device)

        if self.emb_proj is not None:
            support_emb = self.emb_proj(support_emb)

        query_emb = self.bert_forward(query_input_ids, **kwargs)[0]  # Q * max_q_len * D
        if self.emb_proj is not None:
            query_emb = self.emb_proj(query_emb)

        Q = query_input_ids.shape[0]
        query_att_masks = query_input_ids.ne(self.args.pad_index)  # (Q, max_q_len)
        support_att_masks = support_input_ids.ne(self.args.pad_index)  # (N, K, max_s_len)
        instance_matching_vectors, pooled_query, pooled_support = self.instance_compare(query_emb,
                                                                                        support_emb,
                                                                                        query_att_masks,
                                                                                        support_att_masks)
        matching_masks = support_shot_masks.unsqueeze(0).repeat(Q, 1, 1)  # Q, N, K

        logits = self.class_aggregate(instance_matching_vectors, matching_masks, pooled_query,
                                      pooled_support)

        base_logits, prompt_logits, prompt_masked_logits, masked_logits = None, None, None, None

        return logits, support_emb, query_emb, base_logits, prompt_logits, prompt_masked_logits, masked_logits

    def instance_compare(self, query_emb, support_emb, query_att_masks, support_att_masks):
        '''
        return matching feature vectors between each query and each support instance
        :param query_emb: (Q, max_len_a, D)
        :param support_emb: (N, K, max_len_b, D)
        :param query_att_masks: (Q, max_len_a)
        :param support_att_masks: (N, K, max_len_b)
        :return:
          matching_vectors: (Q, N, K, D)
          pooled_query:  (Q, N, K, D), optional
          pooled_support: (Q, N, K, D), optional
        '''
        raise NotImplementedError

    def class_aggregate(self, matching_vectors, matching_masks=None, pooled_query=None, pooled_support=None):
        '''
        aggregate instance-wise matching vectors into class-wise matching vectors for final prediction
        :param matching_vectors: shape=(Q, N, K, D)
        :param matching_masks: shape=(Q, N, K)
        :param pooled_query: shape=(Q, N, K, D)
        :param pooled_support: shape=(Q, N, K, D)
        :return: (Q,N)
        '''
        raise NotImplementedError


@register('mgimn-simple')
class BertMgimnSimple(BertMatchingNetBase):
    '''
       use only instance-level interaction
    '''

    def __init__(self, args):
        super().__init__(args)
        self.fuse_proj = nn.Sequential(
            nn.Dropout(args.output_dropout_prob),
            nn.Linear(args.proj_hidden_size * 4, args.proj_hidden_size),
            nn.ReLU()
        )

        self.instance_matching_predict = nn.Sequential(
            nn.Dropout(args.output_dropout_prob),
            nn.Linear(args.proj_hidden_size * 4, args.proj_hidden_size),
            nn.ReLU()
        )
        self.prediction = torch.nn.Sequential(
            LinearProjection(args.proj_hidden_size, 1))

        self.alignment = Alignment(args.proj_hidden_size)

    def instance_compare(self, query_emb, support_emb, query_att_masks, support_att_masks):
        '''
        return matching feature vectors between each query and each support instance

        :param query_emb: (Q, max_len_a, D)
        :param support_emb: (N, K, max_len_b, D)
        :param query_att_masks: (Q, max_len_a)
        :param support_att_masks: (N, K, max_len_b)
        :return:
          matching_vectors: (Q, N, K, D)
          pooled_query:  (Q, N, K, D), optional
          pooled_support: (Q, N, K, D), optional
        '''
        Q, len_q, D = query_emb.shape
        N, K, len_s, D = support_emb.shape

        z_query = query_emb.view(Q, 1, 1, len_q, D).repeat(1, N, K, 1, 1).view(Q * N * K, len_q, D)
        z_support = support_emb.view(1, N, K, len_s, D).repeat(Q, 1, 1, 1, 1).view(N * K * Q, len_s, D)

        z_q_masks = query_att_masks.view(Q, 1, 1, len_q).repeat(1, 1, N, K, 1).view(Q * N * K, len_q, 1)
        z_s_masks = support_att_masks.view(1, N, K, len_s).repeat(Q, 1, 1, 1).view(N * K * Q, len_s, 1)
        aligned_query, aligned_support = self.alignment(z_query, z_support, z_q_masks, z_s_masks)

        enhanced_query = self.fusion(z_query, aligned_query)
        enhanced_support = self.fusion(z_support, aligned_support)

        enhanced_query.masked_fill_(~z_q_masks, -1e4)
        enhanced_support.masked_fill_(~z_s_masks, -1e4)
        pooled_query = torch.max(enhanced_query, dim=-2).values
        pooled_support = torch.max(enhanced_support, dim=-2).values

        features = torch.cat(
            [pooled_query, pooled_support, pooled_query - pooled_support, pooled_query * pooled_support], dim=-1)
        instance_matching_vectors = self.instance_matching_predict(features)

        return instance_matching_vectors.view(Q, N, K, D), None, None

    def class_aggregate(self, matching_vectors, matching_masks=None, pooled_query=None, pooled_support=None):
        '''
        aggregate instance-wise matching vectors into class-wise matching vectors for final prediction
        :param matching_vectors: shape=(Q, N, K, D)
        :param matching_masks: shape=(Q, N, K)
        :param pooled_query: shape=(Q, N, K, D)
        :param pooled_support: shape=(Q, N, K, D)
        :return: (Q,N)
        '''
        matching_masks = matching_masks.unsqueeze(-1)
        matching_vectors.masked_fill_(~matching_masks, -1e4)
        pooled = torch.max(matching_vectors, dim=-2).values
        logits = self.prediction(pooled).squeeze(-1)
        return logits

    def fusion(self, z, aligned_z):
        '''
        z.shape=aligned_z.shape=(B,len, D)
        return (B, len, D)
        '''
        features = torch.cat([z, aligned_z, z - aligned_z, z * aligned_z], dim=-1)
        return self.fuse_proj(features)


@register('mgimn-class')
class BertMgimnClass(BertMgimnSimple):
    '''
       use only class-level interaction
    '''

    def instance_compare(self, query_emb, support_emb, query_att_masks, support_att_masks):
        ...


@register('mgimn-all')
class BertMgimnAll(BertMgimnSimple):
    def instance_compare(self, query_emb, support_emb, query_att_masks, support_att_masks):
        ...


@register('mlman')
class BertMlman(BertMgimnSimple):
    '''
    https://aclanthology.org/P19-1277.pdf
    '''

    def __init__(self, args):
        super().__init__(args)

        self.instance_matching_predict = nn.Sequential(
            nn.Dropout(args.output_dropout_prob),
            nn.Linear(args.proj_hidden_size * 4, args.proj_hidden_size, bias=False),
            nn.ReLU(),

        )
        self.aggregate_proj = nn.Sequential(
            nn.Linear(args.proj_hidden_size, 1, bias=False)
        )
        self.fuse_proj = nn.Sequential(
            nn.Dropout(args.output_dropout_prob),
            nn.Linear(args.proj_hidden_size * 4, args.proj_hidden_size, bias=False),
            nn.ReLU(),
        )

    def instance_compare(self, query_emb, support_emb, query_att_masks, support_att_masks):
        '''
        return matching feature vectors between each query and each support instance
        :param query_emb: (Q, len_q, D)
        :param support_emb: (N, K, len_s, D)
        :param query_att_masks: (Q, len_q)
        :param support_att_masks: (N, K, len_s)
        :return: (Q, N, K, D)
        '''
        Q, len_q, D = query_emb.shape
        N, K, len_s, D = support_emb.shape

        z_query = query_emb.reshape(Q, 1, len_q, D) \
            .repeat(1, N, 1, 1) \
            .reshape(Q * N, len_q, D)
        z_support = support_emb.reshape(1, N, K * len_s, D) \
            .repeat(Q, 1, 1, 1) \
            .reshape(N * Q, K * len_s, D)

        z_q_masks = query_att_masks.reshape(Q, 1, len_q) \
            .repeat(1, N, 1) \
            .reshape(Q * N, len_q, 1)
        z_s_masks = support_att_masks.reshape(1, N, K * len_s) \
            .repeat(Q, 1, 1) \
            .reshape(N * Q, K * len_s, 1)

        aligned_query, aligned_support = self.alignment(z_query, z_support, z_q_masks, z_s_masks)

        enhanced_query = self.fusion(z_query, aligned_query)
        enhanced_support = self.fusion(z_support, aligned_support)

        enhanced_query.masked_fill_(~z_q_masks, -1e4)
        max_pooled_query = torch.max(enhanced_query, dim=-2).values  # (Q*N,D)
        enhanced_query_float = enhanced_query.float()
        z_q_masks_float = z_q_masks.float()
        mean_pooled_query = torch.sum(enhanced_query_float * z_q_masks_float, dim=-2) / (
                torch.sum(z_q_masks_float, dim=-2) + 1e-5)  # reduce the risk of overflow

        mean_pooled_query = mean_pooled_query.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        pooled_query = torch.cat([max_pooled_query, mean_pooled_query], dim=-1)

        enhanced_support_1 = enhanced_support.masked_fill(~z_s_masks, -1e4)
        enhanced_support_1 = enhanced_support_1.reshape(N * Q, K, len_s, D)
        max_pooled_support = torch.max(enhanced_support_1, dim=-2).values  # (Q * N, K, D)

        enhanced_support_2 = enhanced_support * z_s_masks
        enhanced_support_2 = enhanced_support_2.reshape(N * Q, K, len_s, D)
        z_s_masks = z_s_masks.reshape(N * Q, K, len_s, 1)
        enhanced_support_2_float = enhanced_support_2.float()
        z_s_masks_float = z_s_masks.float()
        mean_pooled_support = torch.sum(enhanced_support_2_float, dim=-2) / (torch.sum(z_s_masks_float, dim=-2) + 1e-5)

        mean_pooled_support = mean_pooled_support.to(dtype=next(self.parameters()).dtype)

        pooled_support = torch.cat([max_pooled_support, mean_pooled_support], dim=-1)  # (Q * N, K, 2D)
        pooled_query = pooled_query.unsqueeze(1).repeat(1, K, 1)  # (Q * N, K, 2D)

        features = torch.cat(
            [pooled_query, pooled_support], dim=-1)

        instance_matching_vectors = self.instance_matching_predict(features)  # (Q * N, K, D)

        return instance_matching_vectors.view(Q, N, K, D), \
               pooled_query.view(Q, N, K, 2 * D), \
               pooled_support.view(Q, N, K, 2 * D)

    def class_aggregate(self, matching_vectors, matching_masks=None, pooled_query=None, pooled_support=None):
        '''
        aggregate instance-wise matching vectors into class-wise matching vectors for final prediction
        :param matching_vectors: shape=(Q, N, K, D)
        :param matching_masks: shape=(Q, N, K)
        :param pooled_query: shape=(Q, N, K, 2D)
        :param pooled_support: shape=(Q, N, K, 2D)
        :return: (Q,N)
        '''
        matching_masks = matching_masks.unsqueeze(-1)
        beta = self.aggregate_proj(matching_vectors)  # Q, N, K, 1
        beta.masked_fill_(~matching_masks, -1e4)
        attn = F.softmax(beta, dim=-2)  # Q, N, 1, K
        prototypes = torch.matmul(attn.transpose(-2, -1), pooled_support).squeeze(-2)  # Q, N, 2D

        pooled_query = pooled_query[:, :, 0, :].squeeze(-2)  # query was repeated for k times before
        features = torch.cat(
            [pooled_query, prototypes], dim=-1)

        features = self.instance_matching_predict(features)
        results = self.aggregate_proj(features)
        return results.squeeze(-1)
