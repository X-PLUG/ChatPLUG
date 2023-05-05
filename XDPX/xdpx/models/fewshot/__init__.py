import torch
from xdpx.models.bert import BertForLanguageModeling


class BertFewshotBase(BertForLanguageModeling):
    def forward(self, query_input_ids, support_input_ids, query_input_ids_with_prompt=None,
                support_input_ids_with_prompt=None,
                masked_input_ids=None, masked_tokens=None,
                query_emb=None, support_emb=None, **kwargs
                ):
        '''
        compute matching scores between query set and support set, embeddings of all instances

        In training phrase (support_emb=None),
        if query set is empty, split original support set into new support set and new query set
        this can speed training process;

        In inference phrase, support_emb are fed as input (*this can speed inference process),
        query set should not be empty;

        :param query_input_ids: shape=(Q, max_q_len)
        :param support_input_ids: shape=(N, K, max_seq_len)
        :param masked_input_ids: shape=(Q+NK, max_seq_len)
        :param masked_tokens: shape=(Q+NK, max_seq_len)
        :param query_emb
        :param support_emb: shape=(N, K, dim) or (N, K, max_seq_len, dim)
        :param kwargs:
        :return: (logits, support_emb, query_emb, seq_logits)

        in training phrase,
            if query_input_ids is not None
                logits.shape=(Q, N)
            else:
                logits.shape=(NK, N) #BertProtoNet
                query_set and support_set are splitted automatically
        in inference phrase,  logits=(Q, N)

        support_emb: (N, K, dim) or (N, K, max_seq_len, dim)
        query_emb: (Q, dim) or (Q, max_query_len, dim) or None when query set is empty
        seq_logits (optional): for auxiliary mlm task
        '''
        raise NotImplementedError

    @property
    def dummy_inputs(self):
        query_input_ids = torch.randint(1, self.args.vocab_size, (4, 10))
        support_input_ids = torch.randint(1, self.args.vocab_size, (8, 4, 10))  # 4-way 4-shot , max-len: 10
        return query_input_ids, support_input_ids
