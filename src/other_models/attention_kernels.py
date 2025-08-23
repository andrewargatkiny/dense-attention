import math
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from itertools import combinations_with_replacement, product

from src.activations import UncenteredFixedLayerNorm, StandardLayerNorm
from src.positional_embeddings import RelPEBase


# FlexAttention doesn't work with class-namespace functions to allow dynamic
# choice of window size as of PyTorch 2.5-2.6.
def sliding_window_mask_32(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & ((q_idx - kv_idx) < 64)

def sliding_window_mask_64(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & ((q_idx - kv_idx) < 64)

def sliding_window_mask_128(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & ((q_idx - kv_idx) < 128)

def sliding_window_mask_256(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & ((q_idx - kv_idx) < 256)

def sliding_window_mask_512(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & ((q_idx - kv_idx) < 512)

def sliding_window_mask_1024(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & ((q_idx - kv_idx) < 1024)

w_size_to_func = {
    32: sliding_window_mask_32, 64: sliding_window_mask_64,
    128: sliding_window_mask_128, 256: sliding_window_mask_256,
    512: sliding_window_mask_512, 1024: sliding_window_mask_1024
}


class SymmetricPowerEmbedding(nn.Module):
    """
    A transform which computes the symmetric tensor power of degree p of an input vector, 
    combining like terms in the expansion. The output has C(d+p-1, p) terms, where d is 
    the input dimension.

    Args:
        config : a ModelConfig class instance with the configuration
        p : The degree of the symmetric tensor product. Controls the state size.

    References:
        Manifest AI.
        https://manifestai.com/articles/symmetric-power-transformers/

    """
    def __init__(self, config, p: int):
        super().__init__()
        self.p = p
        self.d = config.hidden_size // config.num_attention_heads
        # self.norm_factor = math.pow(self.d, self.p / 4.0)
        # generates list of non-decreasing multiindices
        # num_monomials = C(d+p-1, p)
        # indices: num_monomials, p
        indices = list(combinations_with_replacement(range(self.d), self.p))
        self.num_monomials = len(indices)
        if config.scaling_d_factor:
            self.scaling_d_factor = self.d ** ((p - 1) / 2)
        else:
            self.scaling_d_factor = 1
        indices_tensor = torch.tensor(indices, dtype=torch.long)

        # given a multiindex, counts how many times each index appears
        counts = torch.zeros(len(indices), self.d, dtype=torch.float32)
        # counts: num_monomials, d
        ones = torch.ones_like(indices_tensor, dtype=torch.float32)
        # ones: num_monomials, p
        
        # For each monomial (row), adds 1 to counts[m, j] whenever variable j
        # appears in indices_tensor[m]. Records how many times each variable
        # occurs in each monomial.
        counts.scatter_add_(dim=1, index=indices_tensor, src=ones)

        # computes multinomial coefficient
        # lgamma(n+1) = ln(Г(n+1)) = ln(n!)
        # PyTorch doesn't provide factorial directly, so lgamma is used
        # coeffs[m] = sqrt(p! / (count_1! * count_2! * ...)) = exp(ln(coeffs[m]))
        # ln(coeffs[m]) = 0.5*(ln(p!) - (ln(count_1!) + ln(ncount_2!) + ...))
        log_numerator = math.lgamma(self.p + 1)
        log_denom = torch.lgamma(counts + 1).sum(dim=1)
        log_coeffs = 0.5 * (log_numerator - log_denom)
        coeffs = torch.exp(log_coeffs) / self.scaling_d_factor
        # coeffs: num_monomials

        # We transpose `indices` to get a fast, contiguous view of rows in forward.
        self.register_buffer('indices', indices_tensor.t().contiguous(),
                             persistent=False) # shape: p, num_monomials
        self.register_buffer('coeffs', coeffs, persistent=False)

    def forward(self, x: torch.Tensor):
        # Create a tensor of the final flattened form and, for each of p factors,
        # multiply it with relevant x_{indices_i} values, i \in [0, ..., p-1].
        result = torch.ones(size=x.size()[:-1] + [self.num_monomials],
                            dtype=x.dtype, device=x.device)

        for i in range(self.p):
            indices = self.indices[i]
            factors = torch.index_select(x, -1, indices)
            result.mul_(factors)
        return result * self.coeffs



class SymPowFastTraining(nn.Module):
    """ A version of `SymmetricPowerEmbedding`, faster in training but slower
    at inference."""

    def __init__(self, config, p: int):
        import numpy as np
        from scipy.special import factorial
        super().__init__()
        self.p = p
        self.d = config.hidden_size // config.num_attention_heads
        num_monomials = math.comb(self.d + p - 1, p)
        if config.scaling_d_factor:
            self.scaling_d_factor = self.d ** ((p - 1) / 2)
        else:
            self.scaling_d_factor = 1

        all_terms = list(product(range(self.d), repeat=p))
        # all_terms' shape: d^p, p
        # Find only non-decreasing sequences in all terms. These are mutually
        # unique, and all remaining terms are just their permutations.
        non_dec_terms = (np.diff(all_terms, axis=-1) >= 0).all(axis=-1)
        # non_dec_terms: d^p
        unique_inds = np.arange(self.d ** p)[non_dec_terms]
        assert len(unique_inds) == num_monomials
        # unique_inds: num_monomials = C(d+p-1, p)
        unique_items = np.array(all_terms)[non_dec_terms]
        # unique_items: num_monomials, p

        # As all numbers in unique_items are in non-decreasing order, we can
        # produce cumulative counts of unique numbers in a term, starting
        # from 0. In every row, each distinct number gets replaced by its
        # label (cumulative count). E.g., d=5, p=4, an item: 1 2 2 4,
        # corresponding labels: 0 1 1 2
        labels = np.zeros_like(unique_items)
        labels[:, 1:] = ((unique_items[:, 1:] != unique_items[:, :-1])
                         .cumsum(axis=1))
        # This adds offsets to each row so labels don't repeat across rows
        labels += p * np.arange(num_monomials)[:, None] # shape: num_monomials, p
        # Finally produce counts of unique numbers across monomials (rows):
        counts = (np.bincount(labels.ravel(), minlength=num_monomials * p)
                  .reshape(num_monomials, p)) # shape: num_monomials, p
        # Find multinomial coefficients for all monomials. coeffs[m] =
        # p! / (count_1! * count_2! * ...)
        coeffs = factorial(p) / factorial(counts).prod(axis=-1)
        assert coeffs.sum() == self.d**p
        # Square root because coeffs apply both to queries and keys.
        coeffs = np.sqrt(coeffs) / self.scaling_d_factor # shape: num_monomials

        # IntTensor instead of LongTensor because very large d**p are impractical.
        self.register_buffer('indices', torch.IntTensor(unique_inds),
                             persistent=False)
        self.register_buffer('coeffs', torch.Tensor(coeffs), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full tensor power expansion
        expanded_x = x
        for _ in range(self.p - 1):
            x = x.unsqueeze(-2)
            expanded_x = expanded_x.unsqueeze(-1) * x
        expanded_x = expanded_x.flatten(start_dim=-self.p)
        # expanded_x: ..., d**p
        # Choose only unique terms of expansion and multiply by coeffs
        return torch.index_select(expanded_x, -1, self.indices) * self.coeffs


class TensorPowerEmbedding(nn.Module):
    """
    A transform which computes the full tensor power of degree p of an input vector.
    The output has d^p terms, where d is the input dimension.

    Args:
        config : a ModelConfig class instance with the configuration
        p : The degree of the tensor product. Controls the state size.
        
    References:
        Manifest AI.
        https://manifestai.com/articles/symmetric-power-transformers/
    """
    def __init__(self, config, p: int):
        super().__init__()
        self.p = p
        # self.norm_factor = math.pow(config.hidden_size // config.num_attention_heads, self.p / 4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded_x = x
        
        for _ in range(self.p - 1):
            x = x.unsqueeze(-2)
            expanded_x = expanded_x.unsqueeze(-1) * x

        return expanded_x.flatten(start_dim=-self.p)


class Based(nn.Module):
    """
    A transform which computes a second-order Taylor expansion embedding of an input vector, 
    consisting of constant (1), linear, and quadratic terms with appropriate normalization. 
    The output has 1 + d + d(d+1)/2 terms, where d is the input dimension.

    Args:
        config : a ModelConfig class instance with the configuration
    
    References:
        Simple Linear Attention Language Models Balance the Recall–Throughput Tradeoff.
        https://arxiv.org/abs/2402.18668
    """
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size // config.num_attention_heads
        
        self.r2 = math.sqrt(2)
        self.rd = math.sqrt(d)
        self.rrd = math.sqrt(math.sqrt(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        
        term1 = torch.ones_like(x[..., :1])
        term2 = x / self.rrd
        term3 = x2 / self.rd
        
        return torch.cat([term1, term2, term3], dim=-1)

    
Transform2Func = {
    "identity": lambda config: lambda x: x,
    "elu": lambda config: nn.functional.elu,
    "squared_relu": lambda config: lambda x: nn.functional.relu(x) ** 2,
    "1_plus_elu": lambda config: lambda x: 1 + nn.functional.elu(x),
    "sym_power_2": lambda config: SymmetricPowerEmbedding(config, p=2),
    "sym_power_3": lambda config: SymmetricPowerEmbedding(config, p=2),
    "sym_power_4": lambda config: SymmetricPowerEmbedding(config, p=4),
    "sympow_train_2": lambda config: SymPowFastTraining(config, p=2),
    "sympow_train_3": lambda config: SymPowFastTraining(config, p=3),
    "sympow_train_4": lambda config: SymPowFastTraining(config, p=4),
    "power_2": lambda config: TensorPowerEmbedding(config, p=2),
    "power_3": lambda config: TensorPowerEmbedding(config, p=3),
    "power_4": lambda config: TensorPowerEmbedding(config, p=4),
    "based": lambda config: Based(config),
}

class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super(SoftmaxAttention, self).__init__()

    def forward(self, queries: torch.Tensor,
                keys: torch.Tensor, values: torch.Tensor,
                attn_mask: torch.Tensor, dropout_p: float, causal: bool, **kwargs):
        return nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=attn_mask,
            dropout_p=dropout_p, is_causal=causal
        )


class SlidingWindowAttention(nn.Module):
    def __init__(self, config):
        super(SlidingWindowAttention, self).__init__()
        self.window_size = config.window_size
        self.n_heads = config.num_attention_heads
        self.sliding_window_func = w_size_to_func[self.window_size]

    def forward(self, queries: torch.Tensor,
                keys: torch.Tensor, values: torch.Tensor,
                attn_mask: torch.Tensor, dropout_p: float, causal: bool, **kwargs):
        batch_shape = queries.shape[:-2]  # could be (B,) or (B, H) etc.
        B = queries.shape[0]
        L = queries.shape[-2]  # sequence length
        D = queries.shape[-1]  # embedding (or head) dimension

        flat_queries = queries.reshape(B, self.n_heads, L, D)
        # For keys, note that original shape is [B, ..., D, L].
        flat_keys = keys.reshape(B, self.n_heads, L, D)
        flat_values = values.reshape(B, self.n_heads, L, D)
        # Create a block mask using the mask_mod.
        # Here we pass B and H as None to indicate that the mask is the same across batch and heads.
        block_mask = create_block_mask(
            self.sliding_window_func, B=None, H=None, Q_LEN=L, KV_LEN=L,
            device=queries.device
        )
        # Call flex_attention with the block mask.
        #print(f"Queries {flat_queries.contiguous().shape}, keys {flat_keys.contiguous().shape}, values {flat_values.contiguous().shape}")
        flat_attention = flex_attention(flat_queries.contiguous(), flat_keys.contiguous(), flat_values.contiguous(),
                                        block_mask=block_mask
                                        )
        attention = flat_attention.reshape(*batch_shape, L, D)
        return attention


class LinearAttention(nn.Module):
    """General Linear Attention implementation for an arbitrary feature map."""
    def __init__(self, config, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.no_reweight = config.no_reweight
        self.forward_linear = self._forward_linear
        self.forward_quadratic = self._forward_quadratic
        self.forward_causal = self._forward_causal
        self.eps = eps
        if self.no_reweight:
            self.forward_linear = self._forward_linear_no_norm
            self.forward_quadratic = self._forward_quadratic_no_norm
            self.forward_causal = self._forward_causal_no_norm
        transform = Transform2Func[config.feature_map]
        self.feature_map = transform(config)
        self.apply_relpe_after = config.apply_relpe_after
        self.local = False

        self.seq_len = None
        self.causal_mask = None

    def forward(self, queries: torch.Tensor,
                keys: torch.Tensor, values: torch.Tensor,
                attn_mask: torch.Tensor, dropout_p: float, causal: bool,
                rope_cache: RelPEBase = None):
        # TODO: implement chunk-wise parallel causal linear attention
        queries = nn.functional.dropout(queries,p=dropout_p)
        queries = self.feature_map(queries)
        keys = nn.functional.dropout(keys,p=dropout_p)
        keys = self.feature_map(keys)
        shape = queries.shape
        n, d = shape[-2], shape[-1]
        if self.apply_relpe_after:
          if self.local:
            queries = rope_cache.apply_local_relpe2(queries)
            keys = rope_cache.apply_local_relpe2(keys)
          else:
            queries = rope_cache.apply_relpe(queries)
            keys = rope_cache.apply_relpe(keys)
        if causal:
            return self.forward_causal(queries, keys, values, attn_mask, dropout_p)
        if n < d:
            return self.forward_quadratic(queries, keys, values, attn_mask, dropout_p)
        else:
            return self.forward_linear(queries, keys, values, attn_mask, dropout_p)

    def set_local_relpe_state(self, use_local=True):
        local = use_local

    def _causal_mask(self, scores: torch.Tensor):
        n = scores.shape[-1]
        if self.seq_len != n or self.causal_mask is None:
            device, dtype = scores.device, scores.dtype
            broadcast_dims = len(scores.shape) - 2
            mask_shape = [1] * broadcast_dims + [n, n]
            # Create the mask on CPU and move it to device later, to circumvent
            # the issue https://github.com/pytorch/pytorch/issues/136611.
            mask = torch.ones(size=mask_shape, dtype=dtype).tril_().to(device)
            self.causal_mask = mask
        return self.causal_mask

    def _forward_linear(self, queries: torch.Tensor,
                        keys: torch.Tensor, values: torch.Tensor,
                        attn_mask: torch.Tensor, dropout: float):
        # q, k, v: Batch, *, SeqLen, HeadDim
        context = torch.matmul(keys.transpose(-2, -1), values)
        # Batch, *, HeadDim, HeadDim
        normalizer = keys.sum(dim=-2, keepdim=True)
        # Batch, *, 1, HeadDim
        normalizer = normalizer.transpose(-2, -1)
        # Batch, *, HeadDim, 1
        queries = queries / (torch.matmul(queries, normalizer) + self.eps)
        # (Batch, *, SeqLen, HeadDim) / (Batch, *, SeqLen, 1)
        attention = torch.matmul(queries, context)
        # Batch, *, SeqLen, HeadDim
        return attention

    def _forward_quadratic(self, queries: torch.Tensor,
                           keys: torch.Tensor, values: torch.Tensor,
                           attn_mask: torch.Tensor, dropout: float):
        # q, k, v: Batch, *, SeqLen, HeadDim
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        # Batch, *, SeqLen, SeqLen
        normalizer = scores.sum(dim=-1, keepdim=True)
        # Batch, *, SeqLen, 1
        scores = scores / (normalizer + self.eps)
        # (Batch, *, SeqLen, SeqLen) / (Batch, *, SeqLen, 1)
        attention = torch.matmul(scores, values)
        # Batch, *, SeqLen, HeadDim
        return attention

    def _forward_linear_no_norm(
            self, queries: torch.Tensor, keys: torch.Tensor,
            values: torch.Tensor, attn_mask: torch.Tensor, dropout: float
    ):
        # q, k, v: Batch, *, SeqLen, HeadDim
        context = torch.matmul(keys.transpose(-2, -1), values)
        # Batch, *, HeadDim, HeadDim
        attention = torch.matmul(queries, context)
        # Batch, *, SeqLen, HeadDim
        return attention

    def _forward_quadratic_no_norm(
            self, queries: torch.Tensor, keys: torch.Tensor,
            values: torch.Tensor, attn_mask: torch.Tensor, dropout: float
    ):
        # q, k, v: Batch, *, SeqLen, HeadDim
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        # Batch, *, SeqLen, SeqLen
        attention = torch.matmul(scores, values)
        # Batch, *, SeqLen, HeadDim
        return attention

    def _forward_causal(self, queries: torch.Tensor,
                        keys: torch.Tensor, values: torch.Tensor,
                        attn_mask: torch.Tensor, dropout: float):
        # q, k, v: Batch, *, SeqLen, HeadDim
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        # See https://github.com/pytorch/pytorch/issues/136611
        if scores.shape[-1] <= 16384:
            scores.tril_()
        else:
            mask = torch.ones(scores.size()).tril_()
            scores = scores * self._causal_mask(scores)
        # Batch, *, SeqLen, SeqLen
        normalizer = scores.sum(dim=-1, keepdim=True)
        # Batch, *, SeqLen, 1
        scores = scores / (normalizer + self.eps)
        # (Batch, *, SeqLen, SeqLen) / (Batch, *, SeqLen, 1)
        attention = torch.matmul(scores, values)
        # Batch, *, SeqLen, HeadDim
        return attention

    def _forward_causal_no_norm(self, queries: torch.Tensor,
                                keys: torch.Tensor, values: torch.Tensor,
                                attn_mask: torch.Tensor, dropout: float):
        # q, k, v: Batch, *, SeqLen, HeadDim
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        # See https://github.com/pytorch/pytorch/issues/136611
        if scores.shape[-1] <= 16384:
            scores.tril_()
        else:
            mask = torch.ones(scores.size()).tril_()
            scores = scores * self._causal_mask(scores)
        # Batch, *, SeqLen, SeqLen
        attention = torch.matmul(scores, values)
        # Batch, *, SeqLen, HeadDim
        return attention


class PowerAttention(LinearAttention):
    """
    A standalone implementation of attention based on Power/ Symmetric Power
    Transformers/ PolySketchFormer (exact variant). Supports quadratic mode
    computations where it's faster.

    References:
        Symmetric Power Transformers:
            https://manifestai.com/articles/symmetric-power-transformers/
        PolySketchFormer: Fast Transformers via Sketching Polynomial Kernels:
            https://proceedings.mlr.press/v235/kacham24a.html

    Conceptual differences in relation to papers:
        – Supports no re-weighing attention scores by their row-wise sums;
        – Optional scaling_d_factor for prevention of large absolute values;
        – LayerNorm after attention in no re-weighing case

    """
    def __init__(self, config, eps=1e-6):
        super(PowerAttention, self).__init__(config, eps)
        self.p = config.power
        if self.p < 2 or self.p > 4 or not isinstance(self.p, int):
            raise ValueError(f"Integer powers greater than 1 are currently "
                             f"supported, but you provided {self.p}.")
        self.feature_map_train = SymPowFastTraining(config, self.p)
        self.feature_map_infer = SymmetricPowerEmbedding(config, self.p)
        self.feature_map = None
        d = config.hidden_size // config.num_attention_heads
        if config.scaling_d_factor:
            self.scaling_d_factor = d ** -((self.p - 1) / 2)
        else:
            self.scaling_d_factor = None
        num_monomials = math.comb(d + self.p - 1, self.p) # d'
        # 2Nd^2 (quadratic mode) vs 2Ndd' + 2Nd^p (compute + construct
        # embeddings in linear mode). Or equivalently, N vs d' + d^p-1.
        self.lin_thr = num_monomials + d ** (self.p - 1)

        if self.no_reweight:
            self.post_attn_norm = StandardLayerNorm(d)

    def forward(self, queries: torch.Tensor,
                keys: torch.Tensor, values: torch.Tensor,
                attn_mask: torch.Tensor, dropout_p: float, causal: bool,
                rope_cache: RelPEBase = None):
        # TODO: implement chunk-wise parallel causal linear attention
        queries = nn.functional.dropout(queries, p=dropout_p)
        keys = nn.functional.dropout(keys, p=dropout_p)
        if causal:
            if self.scaling_d_factor:
                queries = queries * self.scaling_d_factor
                keys = keys * self.scaling_d_factor
            return self.forward_causal(queries, keys, values, attn_mask, dropout_p)
        n = queries.shape[-2]
        if n < self.lin_thr:
            if self.scaling_d_factor:
                queries = queries * self.scaling_d_factor
                keys = keys * self.scaling_d_factor
            return self.forward_quadratic(queries, keys, values, attn_mask, dropout_p)
        else:
            feature_map = self.feature_map_train if self.training else (
                self.feature_map_infer)
            queries = feature_map(queries)
            keys = feature_map(keys)
            return self.forward_linear(queries, keys, values, attn_mask, dropout_p)

    def _forward_linear_no_norm(
            self, queries: torch.Tensor, keys: torch.Tensor,
            values: torch.Tensor, attn_mask: torch.Tensor, dropout: float
    ):
        attention = super()._forward_linear_no_norm(queries, keys, values,
                                                    attn_mask, dropout)
        return self.post_attn_norm(attention)

    def _forward_quadratic(self, queries: torch.Tensor,
                           keys: torch.Tensor, values: torch.Tensor,
                           attn_mask: torch.Tensor, dropout: float):
        # q, k, v: Batch, *, SeqLen, HeadDim
        scores = torch.matmul(queries, keys.transpose(-2, -1)).pow(self.p)
        # Batch, *, SeqLen, SeqLen
        normalizer = scores.sum(dim=-1, keepdim=True)
        # Batch, *, SeqLen, 1
        scores = scores / (normalizer + self.eps)
        # (Batch, *, SeqLen, SeqLen) / (Batch, *, SeqLen, 1)
        attention = torch.matmul(scores, values)
        # Batch, *, SeqLen, HeadDim
        return attention

    def _forward_quadratic_no_norm(
            self, queries: torch.Tensor, keys: torch.Tensor,
            values: torch.Tensor, attn_mask: torch.Tensor, dropout: float
    ):
        # q, k, v: Batch, *, SeqLen, HeadDim
        scores = torch.matmul(queries, keys.transpose(-2, -1)).pow(self.p)
        # Batch, *, SeqLen, SeqLen
        attention = torch.matmul(scores, values)
        # Batch, *, SeqLen, HeadDim
        return self.post_attn_norm(attention)

    def _forward_causal(self, queries: torch.Tensor,
                        keys: torch.Tensor, values: torch.Tensor,
                        attn_mask: torch.Tensor, dropout: float):
        # q, k, v: Batch, *, SeqLen, HeadDim
        scores = torch.matmul(queries, keys.transpose(-2, -1)).pow(self.p)
        # See https://github.com/pytorch/pytorch/issues/136611
        if scores.shape[-1] <= 16384:
            scores.tril_()
        else:
            mask = torch.ones(scores.size()).tril_()
            scores = scores * self._causal_mask(scores)
        # Batch, *, SeqLen, SeqLen
        normalizer = scores.sum(dim=-1, keepdim=True)
        # Batch, *, SeqLen, 1
        scores = scores / (normalizer + self.eps)
        # (Batch, *, SeqLen, SeqLen) / (Batch, *, SeqLen, 1)
        attention = torch.matmul(scores, values)
        # Batch, *, SeqLen, HeadDim
        return attention

    def _forward_causal_no_norm(self, queries: torch.Tensor,
                                keys: torch.Tensor, values: torch.Tensor,
                                attn_mask: torch.Tensor, dropout: float):
        # q, k, v: Batch, *, SeqLen, HeadDim
        scores = torch.matmul(queries, keys.transpose(-2, -1)).pow(self.p)
        # See https://github.com/pytorch/pytorch/issues/136611
        if scores.shape[-1] <= 16384:
            scores.tril_()
        else:
            mask = torch.ones(scores.size()).tril_()
            scores = scores * self._causal_mask(scores)
        # Batch, *, SeqLen, SeqLen
        attention = torch.matmul(scores, values)
        # Batch, *, SeqLen, HeadDim
        return self.post_attn_norm(attention)

