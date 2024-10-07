import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_config import ModelConfig
from src.positional_embeddings import RelPEBase


class DenseAttention(nn.Module):
    """Efficient implementation of DenseAttention module"""

    def __init__(self, config: ModelConfig, layer_number=1):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = int(self.hidden_size / self.n_heads)
        self.layer_number = layer_number
        self.queries = nn.Parameter(
            torch.zeros(self.hidden_size, self.hidden_size)
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Initialization
        std = config.initializer_range
        num_layers = config.num_hidden_layers
        torch.nn.init.normal_(self.queries, mean=0,
                              std=std / math.sqrt(2.0 * num_layers * self.hidden_size))
        # Runtime complexity selection
        self.attention_complexity = config.attention_complexity
        if self.attention_complexity not in ['linear', 'quadratic', 'auto']:
            raise ValueError(
                f"Attention complexity should be one of these values: "
                f"'linear', 'quadratic', 'auto', but your provided value is "
                f"'{self.attention_complexity}'."
            )
        self._forward_linear = (self._multi_head_linear
                                if self.n_heads > 1
                                else self._single_head_linear)
        self._forward_quadratic = (self._multi_head_quadratic
                                   if self.n_heads > 1
                                   else self._single_head_quadratic)
        if self.attention_complexity == "linear":
            self.forward_ = self._forward_linear
        elif self.attention_complexity == "quadratic":
            self.forward_ = self._forward_quadratic
        else:  # "auto" case
            self.forward_ = self.forward_auto
        self.forward = self.forward_

    def forward_auto(self,
                     hidden_states: torch.Tensor,
                     rope_cache: RelPEBase = None):
        """DenseAttention forward which automatically selects method of
        computation based on optimal total number of operations."""
        # hidden_states: Batch, SeqLen, EmbedDim
        n = hidden_states.shape[1]
        # Same as comparison between (head_size**2 * n) and (head_size * n**2)
        if self.head_size <= n:
            return self._forward_linear(hidden_states, rope_cache)
        else:
            return self._forward_quadratic(hidden_states, rope_cache)

    def _multi_head_quadratic(self,
                              hidden_states: torch.Tensor,
                              rope_cache: RelPEBase = None):
        """DenseAttention forward with quadratic O(N^2*d) time complexity
        w.r.t sequence length suitable for arbitrary number of heads."""
        hidden_states = self.dropout(hidden_states)
        hidden_states = rope_cache.apply_relpe(hidden_states)
        # hidden_states: Batch, SeqLen, EmbedDim
        queries = F.linear(hidden_states, self.queries)
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

    def _multi_head_linear(self,
                           hidden_states: torch.Tensor,
                           rope_cache: RelPEBase = None):
        """DenseAttention forward with linear O(N*d^2) time complexity
        w.r.t sequence length suitable for arbitrary number of heads."""
        hidden_states = self.dropout(hidden_states)
        hidden_states = rope_cache.apply_relpe(hidden_states)
        # hidden_states: Batch, SeqLen, EmbedDim
        size = hidden_states.size()
        new_size = size[:-1] + (self.n_heads, self.head_size)
        queries = F.linear(hidden_states, self.queries)
        # queries: Batch, SeqLen, EmbedDim
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
        #keys = rope.apply_rope(keys, rope_cache.permute(0, 2, 3, 1))

        pre_attn = torch.matmul(keys, hidden_states)
        #pre_attn = self.dropout(pre_attn)
        # pre_attn: Batch, Head, HeadDim, HeadDim
        attention = torch.matmul(queries, pre_attn)
        # attention: Batch, Head, SeqLen, HeadDim

        output = attention.permute(0, 2, 1, 3)
        # output: Batch, SeqLen, Head, HeadDim
        output = output.reshape(*size)
        # output: Batch, SeqLen, EmbedDim
        return output

    def _single_head_quadratic(self,
                               hidden_states: torch.Tensor,
                               rope_cache: RelPEBase = None):
        """Simplified and more efficient DenseAttention forward implementation
        for single-head case use only with quadratic O(N^2*d) time complexity
        w.r.t sequence length."""
        hidden_states = self.dropout(hidden_states)
        hidden_states = rope_cache.apply_relpe(hidden_states)
        # hidden_states: Batch, SeqLen, EmbedDim
        queries = F.linear(hidden_states, self.queries)
        # queries = hidden_states
        # queries: Batch, SeqLen, EmbedDim
        keys = hidden_states.transpose(1, 2)
        # keys: Batch, EmbedDim, SeqLen
        pre_attn = torch.matmul(queries, keys)
        # pre_attn: Batch, SeqLen, Seqlen
        attention = torch.matmul(pre_attn, hidden_states)
        # attention: Batch, SeqLen, EmbedDim
        return attention

    def _single_head_linear(self,
                            hidden_states: torch.Tensor,
                            rope_cache: RelPEBase = None):
        """Simplified and more efficient DenseAttention forward implementation
        for single-head case use only with linear O(N*d^2) time complexity
        w.r.t sequence length."""
        hidden_states = self.dropout(hidden_states)
        hidden_states = rope_cache.apply_relpe(hidden_states)
        # hidden_states: Batch, SeqLen, EmbedDim
        queries = F.linear(hidden_states, self.queries)
        # queries: Batch, SeqLen, EmbedDim
        keys = hidden_states.transpose(-1, -2)
        # keys: Batch, EmbedDim, SeqLen
        pre_attn = torch.matmul(keys, hidden_states)
        # pre_attn: Batch, EmbedDim, EmbedDim
        attention = torch.matmul(queries, pre_attn)
        # attention: Batch, SeqLen, EmbedDim
        return attention

    def extra_repr(self) -> str:
        return (f"in_features={self.hidden_size}, n_heads={self.n_heads}, "
                f"complexity={self.attention_complexity}")


class DenseLocalAttention(DenseAttention):
    def __init__(self, config: ModelConfig, layer_number=1):
        """Implementation of Dense Local Attention for arbitrary number of
        heads with linear complexity."""
        super(DenseLocalAttention, self).__init__(config, layer_number)
        self.window_size = config.window_size
        assert config.max_position_embeddings % self.window_size == 0
        self.forward = self.forward_local

    #TODO: quadratic implementation

    def forward_local(self, hidden_states: torch.Tensor, rope_cache: RelPEBase = None):
        # hidden_states: Batch, SeqLen, EmbedDim
        seq_len = hidden_states.shape[1]
        if seq_len < self.window_size:
            return self.forward_(hidden_states, rope_cache)
        num_windows = seq_len // self.window_size
        last_window =  seq_len - self.window_size * num_windows
        # Handle the case when the seq len is not divisible by window size
        if last_window > 0:
            main_seq_len = seq_len - last_window
            main_part = self._mh_local_linear(
                hidden_states[:, :main_seq_len, :],
                num_windows, rope_cache
            )
            last_part = self.forward_(
                hidden_states[:, :last_window, :], rope_cache
            )
            return torch.cat([main_part, last_part], dim=1)

        return self._mh_local_linear(hidden_states, num_windows, rope_cache)

    def _mh_local_linear(self,
                         hidden_states: torch.Tensor,
                         num_windows: int,
                         rope_cache: RelPEBase = None):
        # hidden_states: Batch, SeqLen, EmbedDim
        size = hidden_states.size()
        hidden_states = rope_cache.apply_local_relpe(
            hidden_states, self.window_size, num_windows
        )
        hidden_states = self.dropout(hidden_states)
        new_size = (size[0], num_windows, self.window_size,
                    self.n_heads, self.head_size)
        queries = F.linear(hidden_states, self.queries)  # * self.norm_ratio_queries)
        # queries: Batch, SeqLen, EmbedDim
        queries = queries.view(new_size)
        # queries: Batch, Seq, SubSeqLen, Head, HeadDim
        queries = queries.permute(0, 1, 3, 2, 4)
        # queries: Batch, Seq, Head, SubSeqLen, HeadDim

        hidden_states = hidden_states.view(new_size)
        # hidden_states: Batch, Seq, SubSeqLen, Head, HeadDim
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4)
        # hidden_states: Batch, Seq, Head, SubSeqLen, HeadDim
        keys = hidden_states.transpose(-2, -1)
        # keys: Batch, Seq, Head, HeadDim, SubSeqLen

        pre_attn = torch.matmul(keys, hidden_states)
        # pre_attn: Batch, Seq, Head, HeadDim, HeadDim
        attention = torch.matmul(queries, pre_attn)
        # attention: Batch, Seq, Head, SubSeqLen, HeadDim

        output = attention.permute(0, 1, 3, 2, 4)
        # output: Batch, Seq, SeqLen, Head, HeadDim
        output = output.reshape(*size)
        # output: Batch, SeqLen, EmbedDim
        return output


class DenseShiftedLocalAttention(DenseLocalAttention):
    def __init__(self, config: ModelConfig, layer_number=1):
        super(DenseShiftedLocalAttention,
              self).__init__(config, layer_number)
        # self.window_size = config.window_size
        assert self.window_size % 2 == 0 and self.window_size > 0
        if config.max_position_embeddings < self.window_size:
            raise ValueError(
                f"max_position_embeddings ({config.max_position_embeddings}) "
                f"should be at least equal to window_size ({self.window_size})."
            )
        else:
            self.left_pad = self.window_size // 2
            self.right_pad = self.window_size // 2
        self.forward = self.forward_shifted_local

    def forward_shifted_local(self, hidden_states: torch.Tensor, rope_cache=None):
        # hidden_states: Batch, SeqLen, EmbedDim
        seq_len = hidden_states.shape[1]
        if seq_len <= self.window_size // 2:
            return self.forward_(hidden_states, rope_cache)
        hidden_states = F.pad(hidden_states,
                              pad=(0, 0, self.left_pad, self.right_pad))
        hidden_states = self.forward_local(hidden_states, rope_cache)
        return hidden_states[:, self.left_pad:-self.right_pad, :]
