import math
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from itertools import combinations_with_replacement
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
    Symmetric Power Embedding for Linear Transformers.
    
    This module implements the symmetric power embedding function for linear transformers.
    Computes tensor-based embeddings of queries and keys based on tensor symmetry,
    reducing the dimensionality of the resulting queries and keys.

    Args:
        config : a ModelConfig class instance with the configuration
        p : The degree of the symmetric tensor product. Controls the state size.

    References:
        Symmetric Power Transformers. Manifest AI.
        https://manifestai.com/articles/symmetric-power-transformers/

    """
    def __init__(self, config, p: int):
        super().__init__()
        self.p = p
        self.d = config.hidden_size // config.num_attention_heads
        # self.norm_factor = math.pow(self.d, self.p / 4.0)
        
        self.register_buffer('indices_tensor', None, persistent=False)
        self.register_buffer('coeffs', None, persistent=False)

        # generates list of non-decreasing multiindices
        # num_monomials = C(d+p-1, p)
        # indices: num_monomials, p
        indices = list(combinations_with_replacement(range(self.d), self.p))
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        self.indices_tensor = indices_tensor  # сохраняем в буфер

        # given a multiindex, counts how many times each index appears
        counts = torch.zeros(len(indices), self.d, dtype=torch.float32)
        ones = torch.ones_like(indices_tensor, dtype=torch.float32)
        # counts: num_monomials, d
        
        # For each monomial (row), adds 1 to counts[m, j] whenever variable j
        # appears in indices_tensor[m]. Records how many times each variable
        # occurs in each monomial.
        counts.scatter_add_(dim=1, index=self.indices_tensor, src=ones)

        # computes multinomial coefficient
        # lgamma(n+1) = ln(Г(n+1)) = ln(n!)
        # PyTorch doesn't provide factorial directly, so lgamma is used
        # coeffs[m] = sqrt(p! / (count_1! * count_2! * ...)) = exp(ln(coeffs[m]))
        # ln(coeffs[m]) = 0.5*(ln(p!) - (ln(count_1!) + ln(ncount_2!) + ...))
        log_numerator = math.lgamma(self.p + 1)
        log_denom = torch.lgamma(counts + 1).sum(dim=1)
        log_coeffs = 0.5 * (log_numerator - log_denom)
        
        # coeffs: num_monomials
        coeffs = torch.exp(log_coeffs)
        self.coeffs = coeffs

    def forward(self, x: torch.Tensor):
        # Computes the product over the last dimension (monomials of degree p)
        # Select coordinates according to multi-indices
        # Example: if index=(0,2), we take x[...,0] and x[...,2]
        selected = x[..., self.indices_tensor]
        # selected: Batch, ..., SeqLen, num_monomials, p
        
        monomials = selected.prod(dim=-1)
        # monomials: Batch, ..., SeqLen, num_monomials

        return (monomials * self.coeffs)


class TensorPowerEmbedding(nn.Module):
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
    "sym_power_4": lambda config: SymmetricPowerEmbedding(config, p=4),
    "power_2": lambda config: TensorPowerEmbedding(config, p=2),
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
    def __init__(self, config, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.no_reweight = config.no_reweight
        self.forward_linear = self._forward_linear
        self.forward_quadratic = self._forward_quadratic
        self.eps = eps
        if self.no_reweight:
            self.forward_linear = self._forward_linear_no_norm
            self.forward_quadratic = self._forward_quadratic_no_norm  
        transform = Transform2Func[config.feature_map]
        self.feature_map = transform(config)
        self.apply_relpe_after = config.apply_relpe_after
        self.local = False

    def forward(self, queries: torch.Tensor,
                keys: torch.Tensor, values: torch.Tensor,
                attn_mask: torch.Tensor, dropout_p: float, causal: bool, rope_cache: RelPEBase):
        # TODO: implement causal linear attention
        queries = self.feature_map(queries)
        queries = nn.functional.dropout(queries,p=dropout_p)
        keys = self.feature_map(keys)
        keys = nn.functional.dropout(keys,p=dropout_p)
        shape = queries.shape
        n, d = shape[-2], shape[-1]
        if self.apply_relpe_after:
          if self.local:
            queries = rope_cache.apply_local_relpe2(queries)
            keys = rope_cache.apply_local_relpe2(keys)
          else:
            queries = rope_cache.apply_relpe(queries)
            keys = rope_cache.apply_relpe(keys)

        if n < d:
            return self.forward_quadratic(queries, keys, values, attn_mask, dropout_p)
        else:
            return self.forward_linear(queries, keys, values, attn_mask, dropout_p)

    def set_local_relpe_state(self, use_local=True):
        local = use_local

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
        normalizer = keys.sum(dim=-2, keepdim=True)
        # Batch, *, 1, HeadDim
        normalizer = normalizer.transpose(-2, -1)
        # Batch, *, HeadDim, 1
        scores = scores / (torch.matmul(queries, normalizer) + self.eps)
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

