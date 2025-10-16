import abc
import math
from enum import Enum, auto

import torch
from torch import nn


class PositionalEmbeddingsTypes(Enum):
    LEARNED = auto()
    RELPE = auto()
    SINUSOIDAL = auto()


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        #self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len,  d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return self.pe[:, :x.size(1)]
        #x = x + self.pe[:x.size(0)]
        #return self.dropout(x)


class RelPEBase(nn.Module, abc.ABC):
    @abc.abstractmethod
    def apply_relpe(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def apply_local_relpe(self, x: torch.Tensor, window_size, num_windows):
        raise NotImplementedError


class DummyRelPE(RelPEBase):
    def __init__(self, seq_len: int, n_elem: int,
                 base: int = 10000, sep_head_dim=True, num_heads=None):
        super(DummyRelPE, self).__init__()
        self.rel_pos_emb = None

    def apply_relpe(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def apply_local_relpe(self, x: torch.Tensor, window_size, num_windows):
        return x

    def apply_local_relpe2(self, x: torch.Tensor, window_size, num_windows):
        return x

class RoPE(RelPEBase):
    def __init__(self, seq_len: int, n_elem: int,
                 base: int = 10000, sep_head_dim=True, num_heads=None,
                 emb_fraction=1.0):
        super(RoPE, self).__init__()
        """RoPE embeddings from the paper https://arxiv.org/abs/2104.09864.
        'Enhanced Transformer with Rotary Position Embedding.'
        
        Parameters
        ----------
        seq_len : int
            Maximum length of input sequence. RoPE cache will have this length 
            in the sequence dimension.     
        n_elem : int
            Embedding dimension of one head
        base : int
            RoPE \Theta base. Default is 10000.     
        sep_head_dim : bool, optional
            Determines whether the head dimension should be separated from the
            embedding dim. If yes, then cached RoPE buffers take form of 
            `bs (1), headdim (1), seqlen, n_elem`. Otherwise, the form is
            `bs (1), seqlen, n_elem`. Default is True.
        num_heads : int, optional
            Number of heads. Default is None.
        emb_fraction: float
            Fraction of the embedding dimension to which RoPE should be 
            applied. Default is 1.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2) / n_elem))
        # theta = 1.0 / (base ** (torch.ones(n_elem // 2) / n_elem))

        if emb_fraction != 1.0:
            n_elem_rope = int(n_elem * emb_fraction)
            # RoPE dimension should be divisible by 2.
            n_elem_rope = n_elem_rope - n_elem_rope % 2
            theta_rope = 1.0 / (base ** (torch.arange(0, n_elem_rope, 2)
                                         / n_elem_rope))
            theta = torch.ones(size=(n_elem // 2,))
            theta[:n_elem_rope // 2] = theta_rope

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len)

        # 1st repeat is for x1 and the 2nd is for x2 coordinate in 2-dimensional
        # (x1, x2) vectors that comprise the whole head embedding dimension.
        # It's assumed in this implementation that all x1s are stored at first
        # within the dim, and only then all x2s.
        angles = torch.outer(seq_idx, theta).repeat(1, 2).float()
        self.rotate_half = self.rotate_half_classic

        cache_cos = torch.cos(angles).unsqueeze(0)
        cache_sin = torch.sin(angles).unsqueeze(0)
        # cache: bs (1), seqlen, head embed dim
        if sep_head_dim:
            self.apply_local_relpe = self.apply_local_relpe_sep
            cache_cos = cache_cos.unsqueeze(1)
            cache_sin = cache_sin.unsqueeze(1)
            # cache: bs (1), headdim (1), seqlen, head embed dim
        else:
            if num_heads is None:
                raise ValueError("If head and embedding dimensions are not "
                                 "separated, `num_heads` should be provided.")
            self.apply_local_relpe = self.apply_local_relpe_fused
            cache_cos = cache_cos.repeat(1, 1, num_heads)
            cache_sin = cache_sin.repeat(1, 1, num_heads)
            # cache: bs (1), seqlen, head embed dim * num heads
            if num_heads > 1:
                # Branch for correct RoPE calculation in setting of multiple
                # heads and merged head-embedding dimensions.
                self.n_heads = num_heads
                self.rotate_half = self.rotate_half_fused_dims

        self.register_buffer("cache_cos", cache_cos, persistent=False)
        self.register_buffer("cache_sin", cache_sin, persistent=False)

    @staticmethod
    def rotate_half_classic(x):
        """Rotates half the hidden dims of the input.
        From https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L84
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def rotate_half_fused_dims(self, x):
        """A version of the rotate half function for the case where head and
        embedding dimensions are not decoupled.
        """
        size = x.size()
        x = x.view(size[:-1] + (self.n_heads, size[-1] // self.n_heads))
        x = self.rotate_half_classic(x)
        return x.view(size)

    def apply_relpe(self, x: torch.Tensor) -> torch.Tensor:
        # truncate to support variable sizes
        seq_len = x.size(-2)
        cache_cos = self.cache_cos[..., :seq_len, :]
        cache_sin = self.cache_sin[..., :seq_len, :]
        pdtype = x.dtype
        # First half of the tensor [x1, x2] takes form x1 * cos - x2 * sin
        # Second half is x1 * cos + x2 * sin
        x = x * cache_cos + self.rotate_half(x) * cache_sin
        return x.to(pdtype)

    def apply_local_relpe_fused(self, x: torch.Tensor, window_size, num_windows):
        """Applies RoPE in a way that treats local attention windows as
        independent sequences. It's assumed that input `x` is of form
        (bs, num_windows * window_size, num_heads * head_dim)."""
        cache_cos = self.cache_cos[..., :window_size, :].repeat(1, num_windows, 1)
        cache_sin = self.cache_sin[..., :window_size, :].repeat(1, num_windows, 1)
        pdtype = x.dtype
        x = x * cache_cos + self.rotate_half(x) * cache_sin
        return x.to(pdtype)

    def apply_local_relpe_sep(self, x: torch.Tensor, window_size, num_windows):
        """Applies RoPE in a way that treats local attention windows as
        independent sequences. It's assumed that input `x` is of form
        (bs, num_heads, num_windows * window_size, head_dim)."""
        cache_cos = self.cache_cos[..., :window_size, :].repeat(1, 1, num_windows, 1)
        cache_sin = self.cache_sin[..., :window_size, :].repeat(1, 1, num_windows, 1)
        pdtype = x.dtype
        x = x * cache_cos + self.rotate_half(x) * cache_sin
        return x.to(pdtype)

    def apply_local_relpe2(self, x: torch.Tensor, window_size, num_windows):
        """Applies RoPE in a way that treats local attention windows as
        independent sequences. It's assumed that input `x` is of form
        (bs, num_windows, num_heads, window_size, head_dim) or
        (bs, num_windows, window_size, num_heads * head_dim),
        and that RoPE cache is respectively
        (bs (1), num_windows (1), num_heads (1), window_size, head_dim)
        or (bs (1), num_windows (1), window_size, num_heads * head_dim).

        This method has the same effect as `apply_relpe`, because by definition
        seq_len = window_size and additional `num_windows` dim is broadcasted
        either way. It will be deprecated in a future release.
        """
        cache_cos = self.cache_cos.unsqueeze(0)[..., :window_size, :]
        cache_sin = self.cache_sin.unsqueeze(0)[..., :window_size, :]
        pdtype = x.dtype
        # First half of the tensor [x1, x2] takes form x1 * cos - x2 * sin
        # Second half is x1 * cos + x2 * sin
        x = x * cache_cos + self.rotate_half(x) * cache_sin
        return x.to(pdtype)

class TrigRelPEBase(RelPEBase):
    """
    Base class for trigonometric RelPE (Cosine, Cos - Sin, etc.) from
    'MatMuls are Enough for Efficient and Performant Linear-Time Attention'

    Use sep_head_dim=None if you intend to apply RelPE to tensors which already
    have material head dimension. If all heads are implicitly stored in one
    dimension or there's only one head, the algorithm will repeat the head dim
    of `n_elem` elements `num_heads` times to cover the whole embedding
    dimension.

    Parameters
    ----------
    seq_len : int
        Maximum length of input sequence. RoPE cache will have this length
        in the sequence dimension.
    n_elem : int
        Embedding dimension of one head
    base : int
        RoPE \Theta base. Default is 10000.
    sep_head_dim : bool, optional
        Determines whether the head dimension should be separated from the
        embedding dim. If yes, then cached RoPE buffers take form of
        `bs (1), headdim (1), seqlen, n_elem`. Otherwise, the form is
        `bs (1), seqlen, n_elem * num_heads`. Default is True.
    num_heads : int, optional
        Number of heads. Default is None.
    """
    def __init__(self, seq_len: int, n_elem: int,
                 base: int = 10000, sep_head_dim=True, num_heads=None):
        super(TrigRelPEBase, self).__init__()
        theta = 1.0 / (base ** (torch.arange(0, n_elem) / n_elem))
        angles: torch.Tensor = torch.outer(torch.arange(seq_len), theta)
        cache = self.trig_transform(angles).unsqueeze(0)
        if sep_head_dim:
            self.apply_local_relpe = self.apply_local_relpe_sep
            cache = cache.unsqueeze(1)
            # cache: bs (1), headdim (1), seqlen, head embed dim
        else:
            if num_heads is None:
                raise ValueError("If head and embedding dimensions are not "
                                 "separated, `num_heads` should be provided.")
            self.apply_local_relpe = self.apply_local_relpe_fused
            cache = cache.repeat(1, 1, num_heads)
            # cache: bs (1), seqlen, head embed dim * num heads
        self.register_buffer("rel_pos_emb", cache, persistent=False)

    @staticmethod
    def trig_transform(angles: torch.Tensor) -> torch.Tensor:
        return NotImplementedError

    def apply_relpe(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.rel_pos_emb

    def apply_local_relpe_fused(self, x: torch.Tensor, window_size, num_windows):
        """Applies RelPE in a way that treats local attention windows as
        independent sequences. It's assumed that input `x` is of form
        (bs, num_windows * window_size, num_heads * head_dim)."""
        return x * self.rel_pos_emb[..., :window_size, :].repeat(1, num_windows, 1)

    def apply_local_relpe_sep(self, x: torch.Tensor, window_size, num_windows):
        """Applies RelPE in a way that treats local attention windows as
        independent sequences. It's assumed that input `x` is of form
        (bs, num_heads, num_windows * window_size, head_dim)."""
        return x * self.rel_pos_emb[..., :window_size, :].repeat(1, 1, num_windows, 1)

    def apply_local_relpe2(self, x: torch.Tensor, window_size, num_windows):
        """Applies RelPE in a way that treats local attention windows as
        independent sequences. It's assumed that input `x` is of form
        (bs, num_windows, num_heads, window_size, head_dim) or
        (bs, num_windows (1), window_size, num_heads * head_dim).

        This method has the same effect as `apply_relpe`, because by definition
        seq_len = window_size and additional `num_windows` dim is broadcasted
        either way. It will be deprecated in a future release.
        """

        return x * self.rel_pos_emb.unsqueeze(0)[..., :window_size, :]

class CosRelPE(TrigRelPEBase):
    @staticmethod
    def trig_transform(angles: torch.Tensor) -> torch.Tensor:
        return angles.cos()


class CosMinusSinRelPE(TrigRelPEBase):
    @staticmethod
    def trig_transform(angles: torch.Tensor) -> torch.Tensor:
        return angles.cos() - angles.sin()


class MultiplicativeLearnedPE(RelPEBase):
    # Multiplicative embedding similar to RotaryEmbedding but learned.
    def __init__(self, config):
        super().__init__()
        #self.dropout = nn.Dropout(0.25)
        # uniformly distributed between -1 and 1
        self.rel_pos_emb = nn.Parameter(
            (torch.rand(config.max_position_embeddings, config.hidden_size) - 0.5) * 2
        ).unsqueeze(0)

    def apply_relpe(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.rel_pos_emb / (self.rel_pos_emb.abs().max(
            axis=-1, keepdim=True)[0] + 1e-4)
        #weight = self.dropout(weight)
        return x * weight

    def apply_local_relpe(self, x: torch.Tensor, window_size, num_windows):
        weight = self.rel_pos_emb[:, :window_size, ...]
        weight = weight / (weight.abs().max(
            axis=-1, keepdim=True)[0] + 1e-4)
        weight = weight.repeat(1, num_windows, 1)
        return x * weight


class RelPEType(Enum):
    LEARNED = auto()
    ROPE = auto()
    COSINE = auto()
    COS_MINUS_SIN = auto()
    DUMMY = auto()


RelPETypeToClass = {
    RelPEType.DUMMY: DummyRelPE,
    RelPEType.ROPE: RoPE,
    RelPEType.COSINE: CosRelPE,
    RelPEType.COS_MINUS_SIN: CosMinusSinRelPE,
    RelPEType.LEARNED: MultiplicativeLearnedPE

}
