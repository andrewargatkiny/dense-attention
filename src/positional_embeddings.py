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
                 base: int = 10000, num_heads=None):
        super(DummyRelPE, self).__init__()
        self.rel_pos_emb = None

    def apply_relpe(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def apply_local_relpe(self, x: torch.Tensor, window_size, num_windows):
        return x


class RoPE(RelPEBase):
    def __init__(self, seq_len: int, n_elem: int,
                 base: int = 10000, num_heads=None):
        super(RoPE, self).__init__()
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2) / n_elem))
        # theta = 1.0 / (base ** (torch.ones(n_elem // 2) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        self.register_buffer("rel_pos_emb", cache, persistent=False)

    def apply_relpe(self, x: torch.Tensor) -> torch.Tensor:
        # truncate to support variable sizes
        T = x.size(1)
        rope_cache = self.rel_pos_emb[:T]

        # cast because the reference does
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        # xshaped = x.reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        x_out2 = x_out2.flatten(3)
        return x_out2.type_as(x)

    def apply_local_relpe(self, x: torch.Tensor, window_size, num_windows):
        raise NotImplementedError


class TrigRelPEBase(RelPEBase):
    def __init__(self, seq_len: int, n_elem: int,
                 base: int = 10000, num_heads=None):
        super(TrigRelPEBase, self).__init__()
        theta = 1.0 / (base ** (torch.arange(0, n_elem) / n_elem))
        angles: torch.Tensor = torch.outer(torch.arange(seq_len), theta)
        cache = self.trig_transform(angles).unsqueeze(0)
        if num_heads is None:
            cache = cache.unsqueeze(-2)
            # cache: bs (1), seqlen, headdim (1), embed dim
        else:
            cache = cache.repeat(1, 1, num_heads)
            # cache: bs (1), seqlen, embed dim * num heads
        self.register_buffer("rel_pos_emb", cache, persistent=False)

    @staticmethod
    def trig_transform(angles: torch.Tensor) -> torch.Tensor:
        return NotImplementedError

    def apply_relpe(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.rel_pos_emb

    def apply_local_relpe(self, x: torch.Tensor, window_size, num_windows):
        return x * self.rel_pos_emb[:, :window_size, ...].repeat(1, num_windows, 1)


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
