import math
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from itertools import combinations_with_replacement


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
    def __init__(self, p: int):
        super().__init__()
        self.p = p
        self.register_buffer('indices_tensor', None)
        self.register_buffer('coeffs', None)
        self._initialized = False

    def _initialize_buffers(self, x: torch.Tensor):
        d = x.shape[-1]
        device = x.device
        dtype = x.dtype
        
        # generates list of non-decreasing multiindices
        indices = list(combinations_with_replacement(range(d), self.p))
        self.indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        
        # given a multiindex, counts how many times each index appears
        counts = torch.zeros(len(indices), d, device=device, dtype=dtype)
        ones = torch.ones_like(self.indices_tensor, dtype=dtype)
        counts.scatter_add_(1, self.indices_tensor, ones)
        
        # computes multinomial coefficient
        log_numerator = math.lgamma(self.p + 1)
        log_denom = torch.lgamma(counts + 1).sum(dim=1)
        log_coeffs = 0.5 * (log_numerator - log_denom)
        self.coeffs = torch.exp(log_coeffs)

        self._initialized = True

    def forward(self, x: torch.Tensor):
        if not self._initialized:
            self._initialize_buffers(x)
        
        d = x.shape[-1]
        norm_factor = math.pow(d, self.p / 2.0)
        
        # Computes the product over the last dimension (monomials of degree p)
        selected = x[..., self.indices_tensor]
        monomials = selected.prod(dim=-1)
        
        return (monomials * self.coeffs) / norm_factor


class TensorPowerEmbedding(nn.Module):
    def __init__(self, p: int, config):
        super().__init__()
        self.p = p
        self.norm_factor = math.pow(config.hidden_size, self.p / 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded_x = x
        
        for _ in range(self.p - 1):
            x = x.unsqueeze(-2)
            expanded_x = expanded_x.unsqueeze(-1) * x

        return expanded_x.flatten(start_dim=-self.p)


class Based(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        
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
    "identity": lambda config=None: lambda x: x,
    "elu": lambda config=None: nn.functional.elu,
    "squared_relu": lambda config=None: lambda x: nn.functional.relu(x) ** 2,
    "1_plus_elu": lambda config=None: lambda x: 1 + nn.functional.elu(x),
    "sym_power_2": lambda config=None: SymmetricPowerEmbedding(p=2),
    "sym_power_4": lambda config=None: SymmetricPowerEmbedding(p=4),
    "power_2": lambda config=None: TensorPowerEmbedding(p=2, config=config),
    "power_4": lambda config=None: TensorPowerEmbedding(p=4, config=config),
    "based": lambda config=None: Based(config),
}

class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super(SoftmaxAttention, self).__init__()

    def forward(self, queries: torch.Tensor,
                keys: torch.Tensor, values: torch.Tensor,
                attn_mask: torch.Tensor, dropout_p: float, causal: bool):
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
                attn_mask: torch.Tensor, dropout_p: float, causal: bool):
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

    def forward(self, queries: torch.Tensor,
                keys: torch.Tensor, values: torch.Tensor,
                attn_mask: torch.Tensor, dropout_p: float, causal: bool):
        # TODO: implement causal linear attention
        queries = self.feature_map(queries)
        queries = nn.functional.dropout(queries,p=dropout_p)
        keys = self.feature_map(keys)
        keys = nn.functional.dropout(keys,p=dropout_p)
        shape = queries.shape
        n, d = shape[-2], shape[-1]
        if n < d:
            return self.forward_quadratic(queries, keys, values, attn_mask, dropout_p)
        else:
            return self.forward_linear(queries, keys, values, attn_mask, dropout_p)

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

