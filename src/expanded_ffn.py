from __future__ import division, absolute_import, print_function, unicode_literals

import math

import torch
from torch import nn


class ExpandedFFN(nn.Module):
    """ A two-layers Feed Forward Network with expanding and contracting layers
         similar to original Transformer block's FFN, but without biases and with
         the mechanics allowing for the weights to be scaled by their max norm."""

    def __init__(self, config):
        super(ExpandedFFN, self).__init__()
        self.hidden_size = config.hidden_size
        self.expansion_factor = config.intermediate_size
        self.intermediate_size = self.hidden_size * self.expansion_factor
        self.expanding_weight = nn.Parameter(self._init_weights())
        self.expansion_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.ReLU()
        self.contracting_weight = nn.Parameter(
            self._init_weights(expansion_flag=False)
        )
        self.contraction_dropout = nn.Dropout(config.hidden_dropout_prob)
        default_ffn_norm = config.default_ffn_norm
        if (config.default_ffn_norm is None or not config.default_ffn_norm
                or config.default_ffn_norm == 0):
            default_ffn_norm = 1 / math.sqrt(self.hidden_size * self.expansion_factor * 2)
        self.default_norm_expand = default_ffn_norm
        self.default_norm_contract = default_ffn_norm
        norm_ratio_expand = self.default_norm_expand / self.expanding_weight.abs().max().item()
        self.norm_ratio_expand = min(1., norm_ratio_expand)
        norm_ratio_contract = self.default_norm_contract / self.contracting_weight.abs().max().item()
        self.norm_ratio_contract = min(1., norm_ratio_contract)

    def adjust_norm_ratios(self):
        """To be used at some points during training to scale back effective
        weights norm if `forward_scaled` is used."""
        with torch.no_grad():
            curr_norm_expand = self.expanding_weight.abs().max().item()
            curr_norm_contract = self.contracting_weight.abs().max().item()
        self.norm_ratio_expand = min(
            self.default_norm_expand / curr_norm_expand, 1.
        )
        self.norm_ratio_contract = min(
            self.default_norm_contract / curr_norm_contract, 1.
        )

    def rescale_weights(self):
        """Multiplies weights by their norm ratios to bring their norm to
        default value."""
        self.adjust_norm_ratios()
        self.expanding_weight.data.mul_(self.norm_ratio_expand)
        self.contracting_weight.data.mul_(self.norm_ratio_contract)

    def prepare_for_inference(self):
        """Prepares weights for inference in case the model have been trained
        with weights scaling."""
        self.rescale_weights()
        self.forward = self.inference_forward

    def _init_weights(self, expansion_flag=True):
        if expansion_flag:
            shape = (self.hidden_size, self.intermediate_size)
        else:
            shape =(self.intermediate_size, self.hidden_size)
        noise_std = 1 / math.sqrt(self.hidden_size * self.expansion_factor * 2)
        noise = torch.randn(shape) * noise_std
        return noise

    def forward(self, hidden_states):
        hidden_states = torch.matmul(hidden_states, self.expanding_weight)
        hidden_states = self.expansion_dropout(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = torch.matmul(hidden_states, self.contracting_weight)
        hidden_states = self.contraction_dropout(hidden_states)
        return hidden_states

    def forward_scaled(self, hidden_states):
        hidden_states = torch.matmul(hidden_states, self.expanding_weight * self.norm_ratio_expand)
        hidden_states = self.expansion_dropout(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = torch.matmul(hidden_states, self.contracting_weight * self.norm_ratio_contract)
        hidden_states = self.contraction_dropout(hidden_states)
        return hidden_states

    def inference_forward(self, hidden_states):
        hidden_states = torch.matmul(hidden_states, self.expanding_weight)
        hidden_states = self.activation(hidden_states)
        hidden_states = torch.matmul(hidden_states, self.contracting_weight)
        return hidden_states
