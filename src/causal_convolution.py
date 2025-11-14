import math

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

        if kernel_size == 4:
            self.forward = self._size_4_forward
            self.weight = nn.Parameter(
                torch.zeros(size=(self.kernel_size, self.in_channels))
            )
            nn.init.uniform_(self.weight, -0.5, 0.5)
        elif kernel_size == 8:
            self.forward = self._size_k_forward
            bound = 1 / math.sqrt(self.kernel_size)
            self.weight = nn.Parameter(
                torch.zeros(size=(self.kernel_size, self.in_channels))
            )
            nn.init.uniform_(self.weight, -bound, bound)
        else:
            self.conv1d = torch.nn.Conv1d(self.in_channels, self.out_channels,
                                          kernel_size,
                                          padding=(kernel_size - 1),
                                          groups=self.in_channels, bias=False)

            self.forward = self._causal_conv


    def _causal_conv(self, x):
        """
        Note that Conv1d expects (batch, in_channels, in_length).
        We assume that x ~ (batch, in_length, in_channels), so we'll reshape it first.
        """
        x = x.transpose(-2, -1)
        conv1d_out = self.conv1d(x).transpose(-2, -1)
        # remove k-1 values from the end:
        return conv1d_out[:, 0:-(self.kernel_size - 1), :]

    def _size_4_forward(self, hidden_states):
        # hidden_states: Batch, SeqLen, EmbedDim
        bs, seq_len, dim = hidden_states.size()
        pad = torch.zeros_like(hidden_states[..., :3, :])
        l = math.ceil(seq_len / 4) * 4
        padded_states = torch.cat([pad, hidden_states, pad], dim=-2)
        states_0 = padded_states[..., :l, :].view(bs, 1, -1, 4, dim)
        states_1 = padded_states[..., 1:1 + l, :].view(bs, 1, -1, 4, dim)
        states_2 = padded_states[..., 2:2 + l, :].view(bs, 1, -1, 4, dim)
        states_3 = padded_states[..., 3:3 + l, :].view(bs, 1, -1, 4, dim)
        # states_i: Batch, Shift (1), SubSeq, KerLen (4), EmbedDim

        states = torch.cat([states_0, states_1, states_2, states_3], dim=-4)
        states = states.transpose(-4, -3)
        # states: Batch, SubSeq, Shift (4), KerLen (4), EmbedDim
        states = self.weight * states
        states = states.sum(dim=-2)
        # states: Batch, SubSeq, Shift (4), EmbedDim
        return states.reshape(bs, -1, dim)[..., :seq_len, :]

    def _size_k_forward(self, hidden_states):
        # hidden_states: Batch, SeqLen, EmbedDim
        bs, seq_len, dim = hidden_states.size()
        k = self.kernel_size
        pad = torch.zeros_like(hidden_states[..., :k, :])
        l = math.ceil(seq_len / k) * k
        padded_states = torch.cat([pad, hidden_states, pad], dim=-2)
        states_arr = []
        for i in range(k):
            states_arr.append(
                padded_states[..., i:l + i, :].view(bs, 1, -1, k, dim)
            )
        # states_i: Batch, Shift (1), SubSeq, KerLen (k), EmbedDim

        states = torch.cat(states_arr, dim=-4)
        states = states.transpose(-4, -3)
        # states: Batch, SubSeq, Shift (k), KerLen (k), EmbedDim
        states = self.weight * states
        states = states.sum(dim=-2)
        # states: Batch, SubSeq, Shift (k), EmbedDim
        return states.reshape(bs, -1, dim)[..., :seq_len, :]
