import torch
import torch.nn as nn
import torch.nn.functional as f
from .positional_embedding import PositionalEmbedding


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, fix_embeddings, dropout, padding_index=0):
        super().__init__()
        self.fix_embeddings = fix_embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_index)
        self.dropout = dropout

    def set_(self, value):
        self.embedding.weight.requires_grad = not self.fix_embeddings
        self.embedding.load_state_dict({'weight': torch.as_tensor(value)})

    def forward(self, x):
        x = self.embedding(x)
        x = f.dropout(x, self.dropout, self.training)
        return x
