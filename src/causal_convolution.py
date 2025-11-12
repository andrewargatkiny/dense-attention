import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from src.model_config import ModelConfig


class SlidingFixedConvolution(nn.Module):
    """Sliding Window 1D separable causal convolution without learnable weights
    (equivalent to all-ones weight matrix)."""
    def __init__(self, config: ModelConfig = None, kernel_size: int =4):
        super(SlidingFixedConvolution, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size == 4:
            self.forward = self._size_4_forward
        else:
            self.forward = self._causal_conv

    def _size_4_forward(self, hidden_states):
        """Highly parallel forward for kernel size 4."""
        # hidden_states: Batch, ..., SeqLen, EmbedDim
        pad = torch.zeros_like(hidden_states[..., :3, :])
        aug_states = torch.cat([pad, hidden_states], dim=-2)
        x_0_1 = hidden_states + aug_states[..., 2:-1, :]
        x_2_3 = aug_states[..., 1:-2, :] + aug_states[..., :-3, :]
        x_full = x_0_1 + x_2_3
        return x_full

    def _causal_conv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward method for generic `kernel_size`"""
        # hidden_states: Batch, ..., SeqLen, EmbedDim
        cumsum = hidden_states.cumsum(dim=-2)
        cumsum[..., self.kernel_size:, :] -= cumsum[..., :-self.kernel_size, :]
        return cumsum


class CausalConv1d(nn.Module):
    """
    A causal 1D convolution.
    """

    def __init__(self, config: ModelConfig, kernel_size=4):
        super(CausalConv1d, self).__init__()

        # attributes:
        self.kernel_size = kernel_size
        self.in_channels = config.hidden_size
        self.out_channels = config.hidden_size

        # modules:
        self.conv1d = torch.nn.Conv1d(self.in_channels, self.out_channels,
                                      kernel_size,
                                      padding=(kernel_size - 1),
                                      groups=self.in_channels, bias=False)

    def forward(self, x):
        """
        Note that Conv1d expects (batch, in_channels, in_length).
        We assume that x ~ (batch, in_length, in_channels), so we'll reshape it first.
        """
        x = x.transpose(-2, -1)
        conv1d_out = self.conv1d(x).transpose(-2, -1)
        # remove k-1 values from the end:
        return conv1d_out[:, 0:-(self.kernel_size - 1), :]

