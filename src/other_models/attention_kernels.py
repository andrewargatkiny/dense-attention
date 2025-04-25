import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

Transform2Func = {
    None: lambda x: x,
    "identity": lambda  x: x,
    "elu": nn.functional.elu,
    "squared_relu": lambda x: nn.functional.relu(x) ** 2,
    "1_plus_elu": lambda x: 1 + nn.functional.elu(x),
}

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

w_size_to_func = {
    32: sliding_window_mask_32, 64: sliding_window_mask_64,
    128: sliding_window_mask_128, 256: sliding_window_mask_256,
    512: sliding_window_mask_512
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
        self.feature_map = Transform2Func[config.feature_map]
        self.no_reweight = config.no_reweight
        self.forward_linear = self._forward_linear
        self.forward_quadratic = self._forward_quadratic
        self.eps = eps
        if self.no_reweight:
            self.forward_linear = self._forward_linear_no_norm
            self.forward_quadratic = self._forward_quadratic_no_norm

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

