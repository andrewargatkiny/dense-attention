import math
import warnings
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from src.model_config import ModelConfig
from src.positional_embeddings import RelPEBase


class DenseAttention(nn.Module):
    """Efficient implementation of DenseAttention module"""

    def __init__(self, config: ModelConfig, local=False,
                 inference=False, layer_number=0):
        super().__init__()
        self.n_heads = config.num_attention_heads
        if local == "softmax" and not config.hybrid: raise NotImplementedError(
            "Softmax sliding window mechanism is not well tested "
            "for DenseAttention and currently not supported."
        )
        self.hidden_size = config.hidden_size
        if self.hidden_size % self.n_heads != 0:
            raise ValueError(
                f"The hidden size {self.hidden_size} is not a multiple of the number of attention "
                f"heads {self.n_heads}."
            )
        self.head_size = int(self.hidden_size / self.n_heads)
        self.layer_number = layer_number
        self.queries = nn.Parameter(
            torch.zeros(self.hidden_size, self.hidden_size)
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Hyperparams for Causal Language Modeling
        self.causal = config.causal
        self.chunk_size = config.chunk_size

        # Hyperparams for local attention
        self.local = local
        if local == 'global':
            self.local = False
        if self.local:
            self.window_size = config.window_size
            assert config.max_position_embeddings % self.window_size == 0
            assert self.window_size % 2 == 0 and self.window_size > 0
            if config.max_position_embeddings < self.window_size:
                raise ValueError(
                    f"max_position_embeddings ({config.max_position_embeddings}) "
                    f"should be at least equal to window_size ({self.window_size})."
                )
            else:
                self.left_pad = self.window_size // 2
                self.right_pad = self.window_size // 2

        # Initialization
        std = config.initializer_range
        num_layers = config.num_hidden_layers
        torch.nn.init.normal_(self.queries, mean=0,
                              std=std / math.sqrt(2.0 * num_layers * self.hidden_size))
        # Runtime complexity selection (for full bidirectional attention only).
        self.attention_complexity = config.attention_complexity
        if self.attention_complexity not in ['linear', 'quadratic', 'auto']:
            raise ValueError(
                f"Attention complexity should be one of these values: "
                f"'linear', 'quadratic', 'auto', but your provided value is "
                f"'{self.attention_complexity}'."
            )
        # Select basic attention computation kernel.
        # For causal attention, only one computational path is available
        # which involves O(C^2) basic kernel for fixed length C causal
        # attention.
        if self.causal:
            self.attention_kernel = self._causal_attn
            if self.local == 'softmax':
                self.attention_kernel = self._softmax_sw_attn
        elif self.attention_complexity == 'linear':
            self.attention_kernel = self._full_linear_attn
        elif self.attention_complexity == 'quadratic':
            self.attention_kernel = self._full_quadratic_attn
        else:
            self.attention_kernel = self._full_auto_attn

        # Select main forward function depending on type of attention
        # (local or global, full or causal).

        # 1. Local options handle both causal and full attention internally.
        if self.local:
            if self.local == "local":
                self.forward = self.forward_local
            elif self.local == "shifted_local":
                self.forward = self.forward_shifted_local
            elif self.local == "sliding_window":
                self.forward = self.forward_sliding_window
            elif self.local == "softmax":
                self.forward = self.forward_global
            else:
                raise ValueError(
                    f"`local` argument should take one of these values: "
                    f" [False, None, 'global, 'local', 'shifted_local', "
                    f"'sliding_window'], but your provided value is "
                    f"'{self.attention_complexity}'."
                )
            # For full local attention the complexity is overridden to be
            # chosen automatically.
            if not self.causal and self.attention_complexity != "auto":
                warnings.warn(
                    f"Overrode local attention complexity in layer "
                    f"{layer_number} to 'auto'."
                )
                self.attention_complexity = "auto"
                self.attention_kernel = self._full_auto_attn
        # 2. Option for global causal attention via efficient O(N) chunk-wise
        # parallel algorithm.
        elif self.causal:
            self.forward = self.forward_causal
        # 3. Option for global full attention. Also serves as a fallback option
        # for global O(N^2) causal attention.
        else:
            self.forward = self.forward_global

        # If the layer is local or shifted local, RelPE can be applied locally,
        # in all other cases globally, using all sequence's indices.
        if config.local_relpe and self.local in ["local", "shifted_local"]:
            self.relpe_func = self._apply_local_relpe
        else:
            self.relpe_func = self._apply_global_relpe

        # Selection of where to apply RelPE
        relpe_keys = {None, False, 'qkv', 'q', 'k', 'v'}
        if not config.relpe_scheme:
            self.relpe_scheme = []
        else:
            self.relpe_scheme = config.relpe_scheme.split(",")
            if not set(self.relpe_scheme).issubset(relpe_keys):
                raise ValueError(f"Not all of the codes in scheme "
                                 f"{self.relpe_scheme} conform to acceptable codes "
                                 f"{relpe_keys}.")

        self.pre_apply_relpe = self._apply_dummy_relpe
        self.apply_q_relpe = self._apply_dummy_relpe
        self.apply_k_relpe = self._apply_dummy_relpe
        self.apply_v_relpe = self._apply_dummy_relpe
        self.pre_apply_global_relpe = self._apply_dummy_relpe
        self.apply_q_global_relpe = self._apply_dummy_relpe
        self.apply_k_global_relpe = self._apply_dummy_relpe
        self.apply_v_global_relpe = self._apply_dummy_relpe
        if "qkv" in self.relpe_scheme:
            self.pre_apply_relpe = self.relpe_func
            self.pre_apply_global_relpe = self._apply_global_relpe
        if "q" in self.relpe_scheme:
            self.apply_q_relpe = self.relpe_func
            self.apply_q_global_relpe = self._apply_global_relpe
        if "k" in self.relpe_scheme:
            self.apply_k_relpe = self.relpe_func
            self.apply_k_global_relpe = self._apply_global_relpe
        if "v" in self.relpe_scheme:
            self.apply_v_relpe = self.relpe_func
            self.apply_v_global_relpe = self._apply_global_relpe


        # Select multi-head or single-head helper utilities.
        if self.n_heads > 1:
            self.permute_dims_f = self._permute_multi_head
            self.head_dims = [self.n_heads, self.head_size]
        else:
            self.permute_dims_f = self._permute_single_head
            self.head_dims = [self.hidden_size]

        self.inference = inference
        if self.inference:
            if not self.causal:
                raise ValueError("Inference mode should be used only when "
                                 "`causal`=True in model config.")
            if self.local and self.local != "sliding_window":
                raise ValueError(f"Causal inference mode only supported only for"
                                 f" 'sliding_window' type of local attention, "
                                 f"{self.local} was provided instead.")
            self.forward = self.forward_inference

    def forward_inference(self):
        pass

    ###########################################################################
    # Basic DenseAttention computation kernels
    ###########################################################################

    def _full_linear_attn(self, queries: torch.Tensor,
                          keys: torch.Tensor,
                          values: torch.Tensor) -> torch.Tensor:
        """Full (bidirectional) DenseAttention forward with linear O(N*d^2)
        time complexity w.r.t sequence length suitable for arbitrary number
        of heads."""
        # queries, values: Batch, ..., SeqLen, EmbedDim
        # keys: Batch, ..., EmbedDim, SeqLen
        # EmbedDim can be a HeadDim in case of multiple heads.
        # Instead of <...> there can be zero, one or two dimensions
        # (Chunk, Head) if we use chunked or/ and multi-head attention.
        kv_state = torch.matmul(keys, values)
        # pre_attn: Batch, ..., EmbedDim, EmbedDim
        attention = torch.matmul(queries, kv_state)
        # attention: Batch, ..., SeqLen, EmbedDim
        return attention

    def _full_quadratic_attn(self, queries: torch.Tensor,
                             keys: torch.Tensor,
                             values: torch.Tensor) -> torch.Tensor:
        """Full (bidirectional) DenseAttention forward with quadratic O(N^2*d)
        time complexity w.r.t sequence length suitable for arbitrary number
        of heads."""
        # queries, values: Batch, ..., SeqLen, EmbedDim
        # keys: Batch, ..., EmbedDim, SeqLen
        # EmbedDim can be a HeadDim in case of multiple heads.
        # Instead of <...> there can be zero, one or two dimensions
        # (Chunk, Head) if we use chunked or/ and multi-head attention.
        pre_attn = torch.matmul(queries, keys)
        # pre_attn: Batch, ..., SeqLen, SeqLen
        attention = torch.matmul(pre_attn, values)
        # attention: Batch, ..., SeqLen, EmbedDim
        return attention

    def _full_auto_attn(self,queries: torch.Tensor,
                        keys: torch.Tensor,
                        values: torch.Tensor) -> torch.Tensor:
        """Full (bidirectional) DenseAttention forward which automatically
        selects method of computation based on optimal total number of
        operations."""
        # queries, values: Batch, ..., SeqLen, EmbedDim
        # keys: Batch, ..., EmbedDim, SeqLen
        # EmbedDim can be a HeadDim in case of multiple heads.
        # Instead of <...> there can be zero, one or two dimensions
        # (Chunk, Head) if we use chunked or/ and multi-head attention.
        n = queries.shape[-2]
        # Same as comparison between (head_size**2 * n) and (head_size * n**2)
        if self.head_size <= n:
            return self._full_linear_attn(queries, keys, values)
        else:
            return self._full_quadratic_attn(queries, keys, values)

    def _causal_attn(self, queries: torch.Tensor,
                     keys: torch.Tensor,
                     values: torch.Tensor) -> torch.Tensor:
        """Causal (left-to-right) DenseAttention forward with quadratic O(N^2*d)
        time complexity w.r.t sequence length suitable for arbitrary number
        of heads."""
        # queries, values: Batch, ..., SeqLen, EmbedDim
        # keys: Batch, ..., EmbedDim, SeqLen
        # EmbedDim can be a HeadDim in case of multiple heads.
        # Instead of <...> there can be zero, one or two dimensions
        # (Chunk, Head) if we use chunked or/ and multi-head attention.
        pre_attn = torch.tril(torch.matmul(queries, keys))
        # pre_attn: Batch, ..., SeqLen, SeqLen
        attention = torch.matmul(pre_attn, values)
        # attention: Batch, ..., SeqLen, EmbedDim
        return attention

    def _softmax_sw_attn(self, queries: torch.Tensor,
                         keys: torch.Tensor,
                         values: torch.Tensor) -> torch.Tensor:
        """Softmax Sliding Window Causal attention"""
        # Here we assume the FlexAttention API is available.
        # queries, values: B, ..., L, D; keys: B, ..., D, L.
        # We need to define a mask_mod function.

        batch_shape = queries.shape[:-2]  # could be (B,) or (B, H) etc.
        B = queries.shape[0]
        L = queries.shape[-2]  # sequence length
        D = queries.shape[-1]  # embedding (or head) dimension

        flat_queries = queries.reshape(B, self.n_heads, L, D).contiguous()
        # For keys, note that original shape is [B, ..., D, L].
        flat_keys = keys.transpose(-1, -2).reshape(B, self.n_heads, L, D).contiguous()
        flat_values = values.reshape(B, self.n_heads, L, D).contiguous()
        # Create a block mask using the mask_mod.
        # Here we pass B and H as None to indicate that the mask is the same across batch and heads.
        block_mask = create_block_mask(
            sliding_window_mask, B=None, H=None, Q_LEN=L, KV_LEN=L,
            device=queries.device
        )
        # Call flex_attention with the block mask.
        flat_attention = flex_attention(flat_queries, flat_keys, flat_values,
                                        block_mask=block_mask)
        attention = flat_attention.reshape(*batch_shape, L, D)
        return attention

    def _add_global_context(self, queries: torch.Tensor,
                            keys: torch.Tensor, values: torch.Tensor,
                            local_attention: torch.Tensor) -> torch.Tensor:
        """Adds all previous contexts to local attention chunks to turn them
        into causal global attention for chunk-wise parallel causal algorithm.
        """
        # queries, values, local_attention: Batch, Chunk, ..., ChunkLen, EmbedDim
        # keys: Batch, Chunk, ..., EmbedDim, ChunkLen
        # EmbedDim can be a HeadDim in case of multiple heads.
        # Instead of <...> there can be zero or one (Head) dimension
        # if we use multi-head attention.
        kv_state = torch.matmul(keys, values)
        # kv_state: Batch, Chunk, ..., EmbedDim, EmbedDim
        # Cumsum the tensor values along the chunk dimension and shift them
        # by one, because kv state for chunk 0 gets used to compute full
        # attention for queries of chunk 1, and so forth.
        kv_state = torch.cumsum(kv_state, dim=1).roll(shifts=1, dims=1)
        kv_state[:, 0, ...] = 0
        prev_context = torch.matmul(queries, kv_state)
        global_attention = prev_context + local_attention
        # shape: Batch, Chunk, ..., ChunkLen, EmbedDim
        return global_attention

    def _add_sliding_window_context(
            self, queries: torch.Tensor, keys: torch.Tensor,
            values: torch.Tensor, local_attention: torch.Tensor
        ) -> torch.Tensor:
        """Adds one previous inverted-causal chunk context to local attention
        chunks to produce causal Sliding Window DenseAttention (SWDA) where each
        token attends to at most `window_size` past tokens, including itself.

        Formula for chunk i:
        SWA_i = Upper(Q_i * K_{i-1}) * V_{i-1} + Lower(Q_i * K_i) * V_i,
        where the second summand is from `local_attention` argument.
        """
        # queries, values, local_attention: Batch, Chunk, ..., ChunkLen, EmbedDim
        # keys: Batch, Chunk, ..., EmbedDim, ChunkLen
        # EmbedDim can be a HeadDim in case of multiple heads.
        # Instead of <...> there can be zero or one (Head) dimension
        # if we use multi-head attention.

        # Shift keys and values by one along chunk dimension.
        keys = keys.roll(shifts=1, dims=1)
        values = values.roll(shifts=1, dims=1)
        # Main diagonal is set to 0 to keep total number of non-zero values as
        # `self.window_size` for any row in the attention matrix.
        pre_attn = torch.triu(torch.matmul(queries, keys), diagonal=1)
        # pre_attn: Batch, Chunk, ..., ChunkLen, ChunkLen
        attention = torch.matmul(pre_attn, values)
        # 0th chunk is zero because K_{-1} and V_{-1} don't exist (or,
        # equivalently, are all-zero matrices) but torch.roll sets them
        # to last chunk.
        attention[:, 0, ...] = 0
        # attention: Batch, Chunk, ..., ChunkLen, EmbedDim

        global_attention = attention + local_attention
        # shape: Batch, Chunk, ..., ChunkLen, EmbedDim
        return global_attention

    ###########################################################################
    # High-level DenseAttention forwards (full, causal, local, shifted local).
    ###########################################################################

    def forward_global(self, hidden_states: torch.Tensor,
                       rope_cache: RelPEBase = None) -> torch.Tensor:
        """Forward for computing bidirectional or causal DenseAttention over
        an entire sequence."""
        # hidden_states: Batch, SeqLen, EmbedDim
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.pre_apply_global_relpe(rope_cache, hidden_states)
        size = hidden_states.size()
        seq_len = size[1]
        queries = F.linear(hidden_states, self.queries)
        queries = self.apply_q_global_relpe(rope_cache, queries)
        keys = self.apply_k_global_relpe(rope_cache, hidden_states)
        values = self.apply_v_global_relpe(rope_cache, hidden_states)
        # queries: Batch, SeqLen, EmbedDim

        # Possibly reshape and permute the sequence in case of multi-head
        # attention.
        new_size = [size[0], seq_len] + self.head_dims
        hidden_states = hidden_states.view(*new_size)
        queries = queries.view(*new_size)
        keys = keys.view(*new_size)
        values = values.view(*new_size)
        # shape: (Batch, SeqLen, EmbedDim) or (Batch, SeqLen, Head, HeadDim)
        queries = self.permute_dims_f(queries)
        keys = self.permute_dims_f(keys)
        values = self.permute_dims_f(values)
        # shape: (Batch, SeqLen, EmbedDim) or (Batch, Head, SeqLen, HeadDim)

        attention = self.attention_kernel(
            queries=queries, keys=keys.transpose(-2, -1),
            values=values
        )
        # attention: (Batch, SeqLen, EmbedDim) or (Batch, Head, SeqLen, HeadDim)
        output = self.permute_dims_f(attention).reshape(*size)
        # output: (Batch, SeqLen, EmbedDim)
        return output

    def forward_causal(self, hidden_states: torch.Tensor,
                      rope_cache: RelPEBase = None) -> torch.Tensor:
        """Computes global causal DenseAttention via chunk-wise parellel
        algorithm` over chunks of size `self.chunk_size`."""
        return self._forward_chunked(hidden_states, self._add_global_context,
                                     self.chunk_size, rope_cache)

    def forward_local(self, hidden_states: torch.Tensor,
                      rope_cache: RelPEBase = None) -> torch.Tensor:
        """Computes local DenseAttention over sequence chunked into
        subsequences of size `self.window_size`."""
        return self._forward_chunked(hidden_states, self._add_dummy_context,
                                     self.window_size, rope_cache)

    def forward_shifted_local(self, hidden_states: torch.Tensor,
                              rope_cache=None) -> torch.Tensor:
        """Computes local DenseAttention over windows shifted by `self.window_size / 2`
        from the start of a sequence."""
        # hidden_states: Batch, SeqLen, EmbedDim
        seq_len = hidden_states.shape[1]
        if seq_len <= self.window_size // 2:
            return self.forward_global(hidden_states, rope_cache)
        hidden_states = F.pad(hidden_states,
                              pad=(0, 0, self.left_pad, self.right_pad))
        hidden_states = self.forward_local(hidden_states, rope_cache)
        return hidden_states[:, self.left_pad:-self.right_pad, :]

    def forward_sliding_window(self, hidden_states: torch.Tensor,
                               rope_cache: RelPEBase = None) -> torch.Tensor:
        """Computes Sliding Window DenseAttention over sequence chunked into
        subsequences of size `self.window_size`."""
        return self._forward_chunked(hidden_states,
                                     self._add_sliding_window_context,
                                     self.window_size, rope_cache)

    def _forward_chunked(self, hidden_states: torch.Tensor,
                         add_context_f: Callable,
                         chunk_size: int,
                         rope_cache: RelPEBase = None) -> torch.Tensor:
        """Chunk-wise algorithm for computation of local or causal parallel
        DenseAttention in linear time."""
        # hidden_states: Batch, SeqLen, EmbedDim
        size = hidden_states.size()
        seq_len = size[1]
        # Simplest case when no chunking is required
        if seq_len <= chunk_size:
            return self.forward_global(hidden_states, rope_cache)

        hidden_states = self.dropout(hidden_states)
        num_chunks = seq_len // chunk_size
        hidden_states = self.pre_apply_relpe(
            rope_cache, hidden_states, chunk_size, num_chunks
        )
        last_chunk =  seq_len - chunk_size * num_chunks
        if last_chunk != 0:
            remainder_dims = list(size)
            remainder_dims[1] = last_chunk
            remainder = torch.zeros(
                size=remainder_dims, device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            hidden_states = torch.cat([hidden_states, remainder], dim=1)
            num_chunks += 1

        queries = F.linear(hidden_states, self.queries)
        queries = self.apply_q_relpe(
            rope_cache, queries, chunk_size, num_chunks
        )
        # shape: Batch, SeqLen, EmbedDim
        keys = self.apply_k_relpe(
            rope_cache, hidden_states, chunk_size, num_chunks
        )
        values = self.apply_v_relpe(
            rope_cache, hidden_states, chunk_size, num_chunks
        )
        # shape: Batch, SeqLen, EmbedDim
        new_size = [size[0], num_chunks, chunk_size] + self.head_dims
        queries = queries.view(*new_size)
        keys = keys.view(*new_size)
        values = values.view(*new_size)
        # shape: (Batch, Chunk, ChunkLen, EmbedDim) or
        # (Batch, Chunk, ChunkLen, Head, HeadDim)
        queries = self.permute_dims_f(queries)
        keys = self.permute_dims_f(keys)
        values = self.permute_dims_f(values)
        # shape: (Batch, Chunk, ChunkLen, EmbedDim) or
        # (Batch, Chunk, Head, ChunkLen, HeadDim)
        keys = keys.transpose(-1, -2)
        # keys: (Batch, Chunk, EmbedDim, ChunkLen) or
        # (Batch, Chunk, Head, HeadDim, ChunkLen)
        local_attention = self.attention_kernel(
            queries=queries, keys=keys, values=values
        )
        # local_attention: (Batch, Chunk, ChunkLen, EmbedDim) or
        # (Batch, Chunk, Head, ChunkLen, HeadDim)

        attention = add_context_f(
            queries=queries, keys=keys, values=values,
            local_attention=local_attention
        )
        # attention: (Batch, Chunk, ChunkLen, EmbedDim) or
        # (Batch, Chunk, Head, ChunkLen, HeadDim)
        output = self.permute_dims_f(attention)
        # output: (Batch, Chunk, ChunkLen, EmbedDim) or
        # (Batch, Chunk, ChunkLen, Head, HeadDim)
        output = output.reshape(size[0], seq_len + last_chunk, size[2])
        output = output[:, :seq_len, :]
        # output: Batch, SeqLen, EmbedDim
        return output

    ###########################################################################
    # Helper methods
    ###########################################################################
    @staticmethod
    def _permute_multi_head(x: torch.Tensor) -> torch.Tensor:
        """Swaps Head and SeqLen dimensions of an input tensor for multi-head
        attention. If applied twice consecutively, returns the original tensor.
        """
        return x.transpose(-2, -3)

    @staticmethod
    def _permute_single_head(x: torch.Tensor) -> torch.Tensor:
        """No-op function for single-head attention case."""
        return x

    def _apply_global_relpe(self, relpe_cache: RelPEBase,
                            hidden_states: torch.Tensor,
                            chunk_size: Optional[int] = None,
                            num_chunks: Optional[int] = None) -> torch.Tensor:
        """Use global sequence indices for calculation of RelPE in local
        full (bidirectional) attention layers."""
        return relpe_cache.apply_relpe(hidden_states)

    def _apply_local_relpe(self, relpe_cache: RelPEBase,
                           hidden_states: torch.Tensor,
                           chunk_size: int, num_chunks: int) -> torch.Tensor:
        """Use local sequence indices for calculation of RelPE in local
        full (bidirectional) attention layers."""
        return relpe_cache.apply_local_relpe(hidden_states, chunk_size,
                                             num_chunks)

    def _apply_dummy_relpe(self, relpe_cache: RelPEBase,
                            hidden_states: torch.Tensor,
                            chunk_size: Optional[int] = None,
                            num_chunks: Optional[int] = None) -> torch.Tensor:
        """Dummy RelPE"""
        return hidden_states

    def _add_dummy_context(self, queries: torch.Tensor,
                           keys: torch.Tensor, values: torch.Tensor,
                           local_attention: torch.Tensor) -> torch.Tensor:
        """Dummy function that abides by the interface of `_add_global_context`
        and allows to reuse `_forward_chunked` for local attention."""
        return local_attention

    def extra_repr(self) -> str:
        info = (f"in_features={self.hidden_size}, n_heads={self.n_heads}, "
                f"causal={self.causal}")
        if self.causal:
            info += f", chunk_size={self.chunk_size}"
        if self.local:
            info += f", local_type={self.local}, window_size={self.window_size}"
        if not self.causal:
            info += f", complexity={self.attention_complexity}"
        return info

def sliding_window_mask(b, h, q_idx, kv_idx):
    # For each dot product, valid only if kv_idx <= q_idx (causal)
    # and the difference is less than self.window_size.
    return (q_idx >= kv_idx) and ((q_idx - kv_idx) < 128)