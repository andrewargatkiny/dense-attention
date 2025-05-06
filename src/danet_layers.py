import copy
import warnings

from torch import nn

from src.other_models.modeling import BertAttention, BertLocalAttention, BertShiftedLocalAttention
from src.activations import StandardLayerNorm, Activation2Class
from src.dense_attention import DenseAttention
from src.expanded_ffn import ExpandedFFN, SwiGLU
from src.model_config import ModelConfig
from src.positional_embeddings import RoPE


class DANetLayer(nn.Module):
    """Basic DenseAttention Network layer which can be put into a model as a
    replacement to a standard Transformer block."""
    def __init__(self, config: ModelConfig, layer_number: int=0):
        super(DANetLayer, self).__init__()
        self.activation = Activation2Class[config.pre_attn_ln_type](config.hidden_size)
        self.attention = DenseAttention(config, layer_number=layer_number)
        self.ffn = SwiGLU(config) if config.swiglu_ffn else ExpandedFFN(config)
        self.ffn_activation = Activation2Class[config.post_attn_ln_type](config.hidden_size)

    def forward(self, hidden_states, attention_mask, rope_cache=None):
        prev_hidden_states = hidden_states
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states * attention_mask
        hidden_states = self.attention(hidden_states, rope_cache)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_activation(hidden_states)
        hidden_states = hidden_states + prev_hidden_states
        return hidden_states


class DANetLayerWithLocalAttention(nn.Module):
    """DenseAttention Network block which supports some local attention scheme.
    Functions identically to `DANetLayer` for global attention."""
    code_to_layer = {
        'g': 'global', 'l': 'local', 'sl': 'shifted_local',
        'sw': 'sliding_window', 'softmax': 'softmax'
    }
    def __init__(self, config: ModelConfig, layer_number: int=0):
        super(DANetLayerWithLocalAttention, self).__init__()
        self.activation = Activation2Class[config.pre_attn_ln_type](config.hidden_size)
        self.window_size = config.window_size
        self.local_scheme = config.local_scheme.split("_")

        if not set(self.local_scheme).issubset(set(self.code_to_layer.keys())):
            warnings.warn(f"Not all of the codes in scheme "
                          f"{self.local_scheme} conform to acceptable codes "
                          f"{self.code_to_layer.keys()}.")


        code = self.local_scheme[layer_number % len(self.local_scheme)]
        locality_name = self.code_to_layer.get(code, "global")
        if locality_name == "global":
            self.prepare_mask_fn = lambda x: x[1]  # global mask
        else:
            # Local mask: each token gets multiplied by window_size ** -1/3 or 0.
            self.prepare_mask_fn = lambda x: x[0]  # local mask
        self.attention = DenseAttention(
            config, local=locality_name,
            layer_number=layer_number
        )
        self.ffn = SwiGLU(config) if config.swiglu_ffn else ExpandedFFN(config)
        self.ffn_activation = Activation2Class[config.post_attn_ln_type](config.hidden_size)

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

class TransformerLayer(nn.Module):
    code_to_kernel = {
        'softmax': 'softmax',
        'swa@softmax': 'swa',
        'l@softmax': 'softmax',
        'sl@softmax': 'softmax'
    }
    code_to_layer = {
        'softmax': BertAttention,
        'swa@softmax': BertAttention,
        'l@softmax': BertLocalAttention,
        'sl@softmax': BertShiftedLocalAttention
    }
    def __init__(self, config: ModelConfig, layer_number: int=0):
        super(TransformerLayer, self).__init__()
        config = copy.deepcopy(config)
        self.window_size = config.window_size
        self.local_scheme = config.local_scheme.split("_")
        code = self.local_scheme[layer_number % len(self.local_scheme)]
        config.attention_kernel = self.code_to_kernel[code]
        attention_class = self.code_to_layer[code]
        config.num_attention_heads = config.transformer_heads
        self.rope_cache = RoPE(
            config.max_position_embeddings, #args.max_seq_length
            config.hidden_size // config.num_attention_heads,
            #num_heads=config.num_attention_heads
        )
        self.pre_activation = StandardLayerNorm(config.hidden_size)
        self.attention = attention_class(config)
        self.post_activation = StandardLayerNorm(config.hidden_size)
        self.ffn = SwiGLU(config) if config.swiglu_ffn else ExpandedFFN(config)

    def forward(self, hidden_states, attention_mask, rope_cache):
        input_layer_norm = self.pre_activation(hidden_states)
        attention_output = self.attention(input_layer_norm, attention_mask, self.rope_cache)
        intermediate_input = hidden_states + attention_output
        intermediate_layer_norm = self.post_activation(intermediate_input)
        layer_output = self.ffn(intermediate_layer_norm)
        return layer_output + intermediate_input

