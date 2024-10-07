import torch
from torch import nn

from src.activations import MaxNormActivation
from src.dense_attention import DenseLocalAttention, DenseShiftedLocalAttention, DenseAttention
from src.expanded_ffn import ExpandedFFN
from src.model_config import ModelConfig


class DANetLayer(nn.Module):
    """Basic DenseAttention Network layer which can be put into a model as a
    replacement to standard Transformer block."""
    def __init__(self, config: ModelConfig, layer_number):
        super(DANetLayer, self).__init__()
        self.activation = MaxNormActivation(config)
        self.attention = DenseAttention(config, layer_number)
        self.ffn = ExpandedFFN(config)
        self.ffn_activation = MaxNormActivation(config)

    def forward(self, hidden_states, attention_mask, rope_cache=None):
        prev_hidden_states = hidden_states
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states * attention_mask
        hidden_states = self.attention(hidden_states, rope_cache)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_activation(hidden_states)
        hidden_states = hidden_states + prev_hidden_states
        return hidden_states

    def forward_unpadded(self, hidden_states, attention_mask):
        """Not guaranteed to work."""
        prev_hidden_states = hidden_states
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states * attention_mask[0]
        seqs = []
        for seq in hidden_states.split(attention_mask[1], dim=0):
            seqs.append(self.attention(seq))
        hidden_states = torch.cat(seqs, dim=0)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_activation(hidden_states)
        hidden_states = hidden_states + prev_hidden_states
        return hidden_states


class DANetLayerWithLocalAttention(nn.Module):
    """DenseAttention Network block with Local-ShiftedLocal-Global local
    attention scheme."""
    def __init__(self, config: ModelConfig, layer_number):
        super(DANetLayerWithLocalAttention, self).__init__()
        self.activation = MaxNormActivation(config)
        self.window_size = config.window_size
        # Local mask: each token gets multiplied by window_size ** -1/3 or 0.
        self.prepare_mask_fn = lambda x: x[0]  # local mask
        if layer_number % 3 == 1:
            self.attention = DenseLocalAttention(config, layer_number)
        elif layer_number % 3 == 2:
            self.attention = DenseShiftedLocalAttention(config, layer_number)
        else:
            self.attention = DenseAttention(config, layer_number)
            # Global mask: each token gets multiplied by
            # seq_len_without_pad_tokens ** -1/3 or 0.
            self.prepare_mask_fn = lambda x: x[1]  # global mask
        self.ffn = ExpandedFFN(config)
        self.ffn_activation = MaxNormActivation(config)

    def forward(self, hidden_states, attention_mask, rope_cache=None):
        prev_hidden_states = hidden_states
        attention_mask = self.prepare_mask_fn(attention_mask)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states * attention_mask
        hidden_states = self.attention(hidden_states, rope_cache)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_activation(hidden_states)
        hidden_states = hidden_states + prev_hidden_states
        return hidden_states

