import torch
from torch import nn


class MaxNormActivation(nn.Module):
    """Activation that divides the input embeddings by their max norm."""

    def __init__(self, config, eps=1e-3):
        super(MaxNormActivation, self).__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states / (hidden_states.abs().max(axis=-1, keepdim=True)[0] + self.eps)
        return hidden_states


class UncenteredLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, init_mean=1.0):
        """Construct a layernorm-like module.
        Code changes:
        1) Mean absolute value instead of standard deviation
        2) Don't center resulting vectors, just divide by std.
        """
        super(UncenteredLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size) * init_mean)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, attention_mask=None):
        pdtype = x.dtype
        x = x.float()
        s = 1.0 / (x.absolute().mean(-1, keepdim=True) + self.variance_epsilon)
        x = x * s
        return self.weight * x.to(pdtype)  # * mask #+ self.bias
