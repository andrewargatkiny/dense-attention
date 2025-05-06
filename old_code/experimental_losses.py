import torch
import torch.nn.functional as F

from torch import nn


class ZeroRegularizer(nn.Module):
    """Regularization to make activations as close to 0 as possible by
    calculating MSE loss between sum of squares and 0"""
    def __init__(self, config):
        super(ZeroRegularizer, self).__init__()
        self.alpha = 1.

    def forward(self, hidden_states):
        sum_squares = hidden_states.pow(2).sum()
        loss_squares = F.mse_loss(
            sum_squares,
            torch.zeros_like(sum_squares)
        )
        return self.alpha * loss_squares


class OutlierRegularizer(nn.Module):
    """Regularization to make activations, which norms are bigger than a
    threshold smaller calculating MSE loss between sum of squares and 0"""
    def __init__(self, config):
        super(OutlierRegularizer, self).__init__()
        self.alpha = 1.
        self.bound = config.initializer_range ** 2 * 3

    def forward(self, hidden_states):
        zeros = torch.zeros_like(hidden_states)
        sum_squares = torch.maximum(hidden_states.pow(2) - self.bound, zeros)
        loss_squares = F.mse_loss(
            sum_squares,
            zeros,
            reduction='mean'
        )
        return self.alpha * loss_squares

class BimodalRegularizer(nn.Module):
    """Regularization to make activations centered around target and -target
    value by calculating MSE loss between its squares and target """
    def __init__(self, config, target=0.01):
        super(BimodalRegularizer, self).__init__()
        self.alpha = 1#0000.
        self.target = target ** 2

    def forward(self, hidden_states):
        targets = self.target * torch.ones_like(hidden_states)
        loss_squares = F.mse_loss(
            hidden_states.pow(2),
            targets,
            reduction='sum'
        )
        return self.alpha * loss_squares


class SquareRegularizer(nn.Module):
    """Regularization by 2th non-central moment of the distribution. This is MSE
    loss between sum of squares of the input and theoretical sum of std devs"""
    def __init__(self, config):
        super(SquareRegularizer, self).__init__()
        self.target_squares = config.initializer_range ** 2

    def forward(self, hidden_states):
        sum_squares = hidden_states.pow(2).sum(axis=-1)
        loss_squares = F.mse_loss(
            sum_squares,
            (self.target_squares * hidden_states.shape[-1])
            * torch.ones_like(sum_squares)
        )
        return loss_squares


class CubeRegularizer(nn.Module):
    """Regularization favoring symmetric distributions. This is MSE
    loss between sum of cubes of the input and 0"""
    def __init__(self, config):
        super(CubeRegularizer, self).__init__()

    def forward(self, hidden_states):
        sum_cubes = hidden_states.pow(3).sum()
        loss_cubes = F.mse_loss(sum_cubes, torch.zeros_like(sum_cubes))
        return loss_cubes


class QuadrupleRegularizer(nn.Module):
    """Regularization by 4th non-central moment of the distribution"""
    def __init__(self, config):
        super(QuadrupleRegularizer, self).__init__()
        # Expectation of sum of 4th moments of i.i.d normal distributions
        # with mean 0. All non-obvious operations are due to fp16 instability.
        self.target_quadruples = 3 ** 0.25 * \
                                 config.initializer_range

    def forward(self, hidden_states: torch.Tensor):
        sum_quadruples = hidden_states.pow(4).sum()
        loss_quadruples = F.mse_loss(
            sum_quadruples,
            (self.target_quadruples * torch.numel(hidden_states) ** 0.25) ** 4
            * torch.ones_like(sum_quadruples)
        )
        return loss_quadruples

