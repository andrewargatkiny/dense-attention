import inspect
import math
import warnings
from functools import partial

import torch
from torch import nn

class MaxNormActivation(nn.Module):
    """Activation that divides the input embeddings by their max norm."""

    def __init__(self, hidden_size=None, eps=1e-3):
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
        #self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, attention_mask=None):
        pdtype = x.dtype
        x = x.float()
        s = 1.0 / (x.absolute().mean(-1, keepdim=True) + self.variance_epsilon)
        x = x * s
        return self.weight * x.to(pdtype)  # * mask #+ self.bias

class UncenteredFixedLayerNorm(nn.Module):
    def __init__(self, hidden_size=None, eps=1e-12, init_mean=1.0):
        """Similar to `UncenteredLayerNorm` but holds no learnable weight"""
        super(UncenteredFixedLayerNorm, self).__init__()
        #self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, attention_mask=None):
        pdtype = x.dtype
        x = x.float()
        s = 1.0 / (x.absolute().mean(-1, keepdim=True) + self.variance_epsilon)
        x = x * s
        return x.to(pdtype)  # * mask #+ self.bias

class StandardLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        My code change: don't use biases.
        """
        super(StandardLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        #self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x, pad_adjust=1.):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype)  # + self.bias

class LegacyGeLU(nn.Module):
    """Legacy implementation so the initial pretrained DenseAttention BERT LM
    doesn't lose quality when used with up-to-date codebase."""
    def __init__(self):
        super(LegacyGeLU, self).__init__()

    @staticmethod
    @torch.jit.script
    def f_gelu(x):
        pdtype = x.dtype
        x = x.float()
        y = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return y.to(pdtype)

    def forward(self, hidden_states):
        return self.f_gelu(hidden_states)

def validate_init_args(cls_type):
    """A decorator to warn user about mismatch of supplied parameters with
    signature of class init instead of raising an error. """
    def wrapped_init(*args, **kwargs):
        signature = inspect.signature(cls_type.__init__)
        init_params = list(signature.parameters.keys())
        for kwarg in list(kwargs.keys()):
            if kwarg not in init_params:
                warnings.warn(f"Param {kwarg} with value {kwargs[kwarg]} was "
                              f"supplied at init but it's not in signature of "
                              f"{cls_type} init function. Skipping it.")
                kwargs.pop(kwarg)
        return cls_type(*args, **kwargs)
    return wrapped_init

Activation2Class = {
    "max_norm": MaxNormActivation,
    "uncentered_ln": UncenteredLayerNorm,
    "uncentered_fixed_ln": UncenteredFixedLayerNorm,
    "standard_ln": StandardLayerNorm,
    "hardtanh": validate_init_args(nn.Hardtanh),
    "no_ln": validate_init_args(nn.Identity),
    "gelu": validate_init_args(nn.GELU),
    "gelu_tanh": validate_init_args(partial(nn.GELU, approximate="tanh")),
    "legacy_gelu": validate_init_args(LegacyGeLU),
    "relu": validate_init_args(nn.ReLU),
    "tanh": validate_init_args(nn.Tanh),
    "leaky_relu": validate_init_args(nn.LeakyReLU),
    "leaky_relu_0.1": validate_init_args(
        partial(nn.LeakyReLU, negative_slope=0.1)
    ),
}