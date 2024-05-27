import math


import torch
from torch import nn
import torch.nn.functional as F


class DenseAttentionMultiHead(nn.Module):
    """ DenseAttention with one dense projection operation and multiple heads"""

    def __init__(self, config, layer_number=1):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = int(self.hidden_size / self.n_heads)
        num_layers = config.num_hidden_layers
        std = config.initializer_range
        self.layer_number = layer_number
        self.queries = nn.Parameter(
            torch.zeros(self.hidden_size, self.hidden_size)
        )
        torch.nn.init.normal_(self.queries, mean=0,
                              std=std / math.sqrt(2.0 * num_layers * self.hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: Batch, SeqLen, EmbedDim
        queries = F.linear(hidden_states, self.queries)  # * self.norm_ratio_queries)
        # queries: Batch, SeqLen, EmbedDim
        size = hidden_states.size()
        new_size = hidden_states.size()[:-1] + (self.n_heads, self.head_size)
        queries = queries.view(new_size)
        # queries: Batch, SeqLen, Head, HeadDim
        queries = queries.permute(0, 2, 1, 3)
        # queries: Batch, Head, SeqLen, HeadDim

        hidden_states = hidden_states.view(new_size)
        # hidden_states: Batch, SeqLen, Head, HeadDim
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        # hidden_states: Batch, Head, SeqLen, HeadDim
        keys = hidden_states.transpose(2, 3)
        # keys: Batch, Head, HeadDim, SeqLen
        pre_attn = torch.matmul(queries, keys)
        # pre_attn: Batch, Head, SeqLen, Seqlen
        attention = torch.matmul(pre_attn, hidden_states)
        # attention: Batch, Head, SeqLen, HeadDim

        output = attention.permute(0, 2, 1, 3)
        # output: Batch, SeqLen, Head, HeadDim
        output = output.reshape(*size)
        # output: Batch, SeqLen, EmbedDim
        return output


class DenseAttentionMultiHeadLinComplexity(DenseAttentionMultiHead):
    """ DenseAttention with one dense projection operation and multiple heads
    with O(d^2*N) time complexity"""

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: Batch, SeqLen, EmbedDim
        # queries: Batch, SeqLen, EmbedDim
        size = hidden_states.size()
        new_size = size[:-1] + (self.n_heads, self.head_size)
        queries = F.linear(hidden_states, self.queries)  # * self.norm_ratio_queries)
        queries = queries.view(new_size)
        # queries: Batch, SeqLen, Head, HeadDim
        queries = queries.permute(0, 2, 1, 3)
        # queries: Batch, Head, SeqLen, HeadDim

        hidden_states = hidden_states.view(new_size)
        # hidden_states: Batch, SeqLen, Head, HeadDim
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        # hidden_states: Batch, Head, SeqLen, HeadDim
        keys = hidden_states.transpose(2, 3)
        # keys: Batch, Head, HeadDim, SeqLen

        pre_attn = torch.matmul(keys, hidden_states)
        # pre_attn: Batch, Head, HeadDim, HeadDim
        attention = torch.matmul(queries, pre_attn)
        # attention: Batch, Head, SeqLen, HeadDim

        output = attention.permute(0, 2, 1, 3)
        # output: Batch, SeqLen, Head, HeadDim
        output = output.reshape(*size)
        # output: Batch, SeqLen, EmbedDim
        return output


class DenseAttentionOneHead(nn.Module):
    """ Simplified implementation of DenseAttention equivalent to it in case
     there is only one head"""

    def __init__(self, config, layer_number=1):
        super().__init__()
        self.hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        std = config.initializer_range
        self.layer_number = layer_number
        self.queries = nn.Parameter(
            torch.zeros(self.hidden_size, self.hidden_size)
        )
        torch.nn.init.normal_(self.queries, mean=0,
                              std=std / math.sqrt(2.0 * num_layers * self.hidden_size))

        #self.default_norm_queries = 1. / math.sqrt(self.hidden_size)  # * correction_ratio ** 2
        #self.norm_ratio_queries = self.default_norm_queries / self.queries.abs().max().item()

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: Batch, SeqLen, EmbedDim
        queries = F.linear(hidden_states, self.queries)  # * self.norm_ratio_queries)
        # queries = hidden_states
        # queries: Batch, SeqLen, EmbedDim
        keys = hidden_states.transpose(1, 2)
        # keys: Batch, EmbedDim, SeqLen
        pre_attn = torch.matmul(queries, keys)
        # pre_attn: Batch, SeqLen, Seqlen
        attention = torch.matmul(pre_attn, hidden_states)
        # attention: Batch, SeqLen, EmbedDim
        return attention


class DenseAttentionOneHeadLinComplexity(DenseAttentionOneHead):
    """ Simplified implementation of DenseAttention equivalent to it in case
     there is only one head but with altered order of operations which yields
     linear complexity"""

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: Batch, SeqLen, EmbedDim
        queries = F.linear(hidden_states, self.queries)  # * self.norm_ratio_queries)
        # queries: Batch, SeqLen, EmbedDim
        keys = hidden_states.transpose(-1, -2)
        # keys: Batch, EmbedDim, SeqLen
        pre_attn = torch.matmul(keys, hidden_states)
        # pre_attn: Batch, EmbedDim, EmbedDim
        attention = torch.matmul(queries, pre_attn)
        # attention: Batch, SeqLen, EmbedDim
        return attention


