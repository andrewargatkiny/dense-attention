import math

import torch
import torch.nn.functional as F

from torch import nn
from src.model_config import ModelConfig


class ClipGradValue(torch.autograd.Function):
    """Autograd function that leaves inputs unchanged during forward but clips
     gradients during backward"""
    @staticmethod
    def forward(input, clip_value):
        return input

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, clip_value = inputs
        ctx.clip_value = clip_value

    @staticmethod
    def backward(ctx, grad_output):
        grad_clipped = torch.clamp(grad_output, min=-ctx.clip_value, max=ctx.clip_value)
        max_norm = 1.
        total_norm = grad_clipped.abs().mean()
        ratio = torch.min(torch.tensor([0.1], device=total_norm.device), max_norm / total_norm)
        grad_clipped = grad_clipped * ratio

        return grad_clipped, None  # No gradient for clip_value


def clip_grad_values(input, clip_value=10000):
    return ClipGradValue.apply(input, clip_value)

def clip_param_grad(grad: torch.Tensor):
    return torch.clamp(grad, min=-1., max=1.)


class Clamper(torch.autograd.Function):
    """Autograd function that applies Hardtanh activation to inputs during
     forward but passes outer gradients unchanged regardless of inputs during
     backward"""
    @staticmethod
    def forward(input):
        return F.hardtanh(input)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output) / 10

class BoundActivation(nn.Module):
    """activation that applies Hardtanh activation to inputs during
     forward but passes outer gradients unchanged regardless of inputs during
     backward"""
    def __init__(self, config):
        super(BoundActivation, self).__init__()

    def forward(self, hidden_states: torch.Tensor):
        return Clamper.apply(hidden_states)

class ClampingActivation(nn.Module):
    """Activation that trunks inputs ge by abs than a defined value then multiplies by a weight."""
    def __init__(self, config, bound=0.5):
        super(ClampingActivation, self).__init__()
        assert bound > 0, f"Bound {bound} <= 0"
        #self.weight = nn.Parameter(torch.ones(config.hidden_size) * 0.5)
        self.clamper = nn.Hardtanh(-bound, bound)

    def forward(self, hidden_states):
        return self.clamper(hidden_states)


class ModDivActivation(nn.Module):
    """ Activation which returns remainder of division by bound"""
    def __init__(self, config: ModelConfig, bound=1.0):
        super(ModDivActivation, self).__init__()
        self.bound = bound

    def forward(self, x: torch.Tensor):
        return torch.fmod(x, self.bound)


class ScalerActivation(nn.Module):
    """Activation with just a one-value scaler parameter which gets
    changed manually, not by gradient descent"""
    def __init__(self, config: ModelConfig):
        super(ScalerActivation, self).__init__()
        self.scaler = nn.Parameter(torch.full((config.hidden_size,), 1.0))
        self.weight = nn.Parameter(torch.ones(config.hidden_size) * 0.1)

    def forward(self, hidden_states):
        return self.scaler * self.weight * hidden_states


class NormalizingActivation(nn.Module):
    """Like LayerNorm, but no bias, uncentered, scaler is discrete,
    calculated over all dimensions of a batch"""
    def __init__(self, config: ModelConfig):
        super(NormalizingActivation, self).__init__()
        self.bound = nn.Parameter(torch.ones(1) * 2., requires_grad=False)
        self.log_adjust = nn.Parameter(torch.ones(1) * 1., requires_grad=False)
        self.max_multiplier = nn.Parameter(torch.ones(1), requires_grad=False)
        self.weight = nn.Parameter(torch.ones(config.hidden_size) * 0.1)

    def forward(self, hidden_states):
        pdtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        multiplier = torch.minimum(
            self.max_multiplier,
            self.bound ** -torch.floor(torch.log2(hidden_states.absolute().max() + 1e-6) / self.log_adjust)

        )
        return (self.weight * multiplier * hidden_states).to(pdtype)


class SmoothIndicatorActivation(nn.Module):
    """Activation with form resembling x * sigmoid(alpha * (x - low_bound)) *
    sigmoid(alpha * (up_bound - x))"""
    def __init__(self, config, alpha=100., bound=0.05):
        super(SmoothIndicatorActivation, self).__init__()
        assert alpha > 0, f"Alpha {alpha} <= 0"
        assert bound > 0, f"Bound {bound} <= 0"
        self.alpha = alpha
        self.bound = bound
        self.upper_bound = bound
        self.lower_bound = -bound

    def forward(self, hidden_states):
        exps_low = torch.exp(-self.alpha * (hidden_states + self.bound))
        exps_up = torch.exp(-self.alpha * (self.bound - hidden_states))
        return hidden_states / (1. + exps_low + exps_up)


def add_normalizer_preforward_hooks(args, model):
    def getActivation(module_name):
        # the hook signature
        def hook(module: torch.nn.Module(), input: torch.Tensor):
            max_val = input[0].detach().absolute().max()
            scaler_val = module.scaler[0].detach()
            max_scaled_val = max_val * scaler_val
            bound = 2
            l_bound_pow = -3
            if max_scaled_val > bound:
                multiplier = bound / 2 * bound ** -math.floor(math.log(max_val, bound))
                print(f"max abs input {max_val}, old scaler {scaler_val}")
                module.scaler.data.fill_(multiplier)
                print(f"Changed scaler of layer {module_name} to {multiplier}")

            return input
        return hook

    hooks = []
    for name, module in model.named_modules():
        if "layer" in name and "activation" in name \
                or "LayerNorm" in name and hasattr(module, "scaler"):
            module.scaler.requires_grad = False
            hooks.append(
                module.register_forward_pre_hook(getActivation(name))
            )
    return hooks
