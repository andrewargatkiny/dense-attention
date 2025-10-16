# coding=utf-8
# Copyright 2025 Andrew Argatkiny
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from src.positional_embeddings import PositionalEmbeddingsTypes, SinusoidalPositionalEncoding, RelPETypeToClass, \
    RelPEType
from .attention_kernels import SoftmaxAttention, LinearAttention, SlidingWindowAttention, PowerAttention
from ..activations import Activation2Class


logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    pdtype = x.dtype
    x = x.float()
    y = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return y.to(pdtype)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class TransformerConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 attention_kernel="softmax",
                 feature_map=None,
                 no_reweight=False,
                 no_reweight_post_norm=None,
                 hidden_act="gelu",
                 embedding_dropout=0,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 attn_proj_biases=True,
                 max_position_embeddings=512,
                 pos_emb_type="learned",
                 relpe_type=None,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 pre_attn_ln_type="default",
                 post_attn_ln_type="default",
                 causal=False,
                 local_attention=False,
                 local_scheme = None,
                 window_size=1024,
                 apply_relpe_after=False,
                 layers_scheme=None,
                 power=2,
                 scaling_d_factor=False,
                 layers=None,
                 **kwargs):
        """Constructs ModelConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            attn_proj_biases: Whether to use bias in Q, K, V, O matrices in attention.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            apply_relpe_after: Whether Relative Positional Encoding (RELPE) is applied after the feature map (if true)
             or before the linear attention kernel (if false).
            local_scheme: Scheme to form patterns of local and global attention
                layers. Should contain lowercase-letter layer codes separated
                by underscore '_'. Available codes: 'l' (local attention), 'sl'
                (shifted local), 'swa' (sliding window), and 'g' (global).
                If None, `local_attention` flag is used with a hardcoded scheme.
            pre_attn_ln_type: If not set to "default" (which is `BertLayerNorm`),
                determines the type of layer norm or activation to use before attention.
            post_attn_ln_type: Like `pre_attn_ln_type` but for usage before FFN.
            attention_kernel: Mechanism for attention to use. Currently supported:
                "softmax", "swa", "linear", "power". Power attention is subtype of
                linear attention, and many options for linear attention also apply.
            feature_map: A feature map transform (\phi) for queries and keys in linear
                attentions.
            no_reweight: For linear attentions, if set to true, doesn't scale attention
                scores by their row-wise sums.
            no_reweight_post_norm: In case of enabled `no_reweight` option in linear
                attentions, determines whether and which layer norm to use at the end
                of attention kernel computation. Defaults to None.
            apply_relpe_after: For linear attentions, determines whether Relative
                Positional Encoding (RELPE) is applied after the feature map (if true)
                or before the linear attention kernel (if false).
            layers_scheme: Defines the sequence and repetition of layers within the encoder.
                This should be a string of layer names separated by underscores
                (e.g., 'attention_ffn_attention'). Each name must correspond to a unique layer_name
                key in one of the configuration dictionaries provided in the layers parameter.
                power: For Power Attention, determines the power (p).
            scaling_d_factor: For Power Attention, determines whether to scale q,k by a
                predetermined scaling factor depending on d for additional numerical
                stability.
            layers: A list of dictionaries, where each dictionary provides the configuration
                for a specific layer type. Each dictionary must contain a unique layer_name
                key, which is then used by the layer_scheme parameter to construct the full encoder stack.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r",
                      encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.attention_kernel = attention_kernel
            self.feature_map = feature_map
            self.no_reweight = no_reweight
            self.no_reweight_post_norm = no_reweight_post_norm
            self.embedding_dropout = embedding_dropout
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.attn_proj_biases = attn_proj_biases
            self.max_position_embeddings = max_position_embeddings
            if layers is None:
                self.pos_emb_type = PositionalEmbeddingsTypes[pos_emb_type.upper()]
            else:
                self.pos_emb_type = pos_emb_type
            self.relpe_type = relpe_type
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.pre_attn_ln_type = pre_attn_ln_type
            self.post_attn_ln_type = post_attn_ln_type
            self.causal = causal
            self.local_attention = local_attention
            self.window_size = window_size
            self.apply_relpe_after = apply_relpe_after
            self.local_scheme = local_scheme
            self.layers_scheme = layers_scheme
            self.power = power
            self.scaling_d_factor = scaling_d_factor
            self.layers = layers
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `ModelConfig` from a Python dictionary of parameters."""
        config = TransformerConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        if torch.distributed.get_rank() == 0:
            print(config)
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `ModelConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class TransformerLayerConfig(object):
    """Configuration class to store the configuration of a Transformer layer.
    """
    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 attention_kernel="softmax",
                 feature_map=None,
                 no_reweight=False,
                 no_reweight_post_norm=None,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 attn_proj_biases=True,
                 initializer_range=0.02,
                 pre_attn_ln_type="default",
                 post_attn_ln_type="default",
                 causal=False,
                 local_attention=False,
                 window_size=1024,
                 apply_relpe_after=False,
                 max_position_embeddings=512,
                 pos_emb_type="learned",
                 relpe_type=None,
                 local_scheme=None,
                 power=2,
                 scaling_d_factor=False,
                 **kwargs):
        """Constructs TransformerLayerConfig.

        Args:
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            attention_kernel: Mechanism for attention to use. Supported: "softmax", 
                "swa", "linear", "power".
            feature_map: A feature map transform (\phi) for queries and keys in linear
                attentions.
            no_reweight: For linear attentions, if True, does not scale attention
                scores by their row-wise sums.
            no_reweight_post_norm: In case of `no_reweight=True` in linear
                attentions, determines which layer norm to use at the end
                of attention kernel computation. Defaults to None.
            hidden_act: The non-linear activation function (function or string) in the
                encoder. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            attn_proj_biases: Whether to use bias in Q, K, V, O matrices in attention.
            initializer_range: The standard deviation of the truncated_normal_initializer for
                initializing all weight matrices.
            pre_attn_ln_type: If not "default" (which is `BertLayerNorm`),
                determines the type of layer norm or activation to use before attention.
            post_attn_ln_type: Like `pre_attn_ln_type` but for usage before FFN.
            causal: Whether to apply causal masking to the attention scores.
            local_attention: Whether to use local attention.
            window_size: The window size for local attention.
            apply_relpe_after: Whether Relative Positional Encoding (RELPE) is applied 
                after the feature map (if True) or before the linear attention 
                kernel (if False).
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with.
            pos_emb_type: The type of positional embedding to use. e.g., "learned",
                "rotary".
            relpe_type: The type of relative position encoding to use.
            power: For Power Attention, determines the power (p).
            scaling_d_factor: For Power Attention, determines whether to scale q,k by a
                predetermined scaling factor depending on d for additional numerical
                stability.
        """
        
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.attention_kernel = attention_kernel
        self.feature_map = feature_map
        self.no_reweight = no_reweight
        self.no_reweight_post_norm = no_reweight_post_norm
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.attn_proj_biases = attn_proj_biases
        self.max_position_embeddings = max_position_embeddings
        self.pos_emb_type = pos_emb_type
        self.relpe_type = relpe_type
        self.local_scheme = local_scheme
        self.initializer_range = initializer_range
        self.pre_attn_ln_type = pre_attn_ln_type
        self.post_attn_ln_type = post_attn_ln_type
        self.causal = causal
        self.local_attention = local_attention
        self.window_size = window_size
        self.apply_relpe_after = apply_relpe_after
        self.power = power
        self.scaling_d_factor = scaling_d_factor



#try:
#    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
#except ImportError:
#print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        if config.attention_kernel not in ["softmax", "linear", "swa", "power"]:
            raise ValueError("Attention kernel param should hold value of "
                             "either 'softmax' or 'linear' or 'swa' or 'power'.")
        if config.attention_kernel == "softmax":
            self.attention_kernel = SoftmaxAttention(config)
        elif config.attention_kernel == "swa":
            self.attention_kernel = SlidingWindowAttention(config)
        elif config.attention_kernel == "linear":
            self.attention_kernel = LinearAttention(config)
        elif config.attention_kernel == "power":
            self.attention_kernel = PowerAttention(config)
        else:
            raise NotImplementedError(
                f"Attention kernel for {config.attention_kernel} is not "
                f"implemented"
            )
        self.causal = config.causal
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size,
                               bias=config.attn_proj_biases)
        self.key = nn.Linear(config.hidden_size, self.all_head_size,
                             bias=config.attn_proj_biases)
        self.value = nn.Linear(config.hidden_size, self.all_head_size,
                               bias=config.attn_proj_biases)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout_prob = config.attention_probs_dropout_prob
        self.apply_relpe_after = config.apply_relpe_after
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, rope_cache):
        # hidden_states = rope_cache.apply_relpe(hidden_states)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        if not self.apply_relpe_after:
          query_layer = rope_cache.apply_relpe(query_layer)
          key_layer = rope_cache.apply_relpe(key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        #if torch.all(attention_mask == 0):
        attention_mask = None
        #kv = torch.matmul(key_layer.transpose(-1, -2), value_layer)
        #context_layer = torch.matmul(query_layer, kv)
        #attention_mask = None
        context_layer = self.attention_kernel(
            query_layer, key_layer, value_layer, attn_mask=attention_mask,
            dropout_p=self.dropout_prob, causal=self.causal, rope_cache=rope_cache
        )
        """
        context_layer = nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attn_mask=attention_mask,
            dropout_p=self.dropout_prob
        )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        pdtype = attention_scores.dtype
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(
            attention_scores.float()).to(pdtype)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        """
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertSelfLocalAttention(BertSelfAttention):
    def __init__(self, config):
        super(BertSelfLocalAttention, self).__init__(config)
        self.window_size = config.window_size
        assert config.max_position_embeddings % self.window_size == 0

    def transpose_for_local_scores(self, x, num_windows):
        new_x_shape = (x.size()[0], num_windows, self.window_size,
                       self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # Batch, Seq, SubSeqLen, Head, HeadDim
        return x.permute(0, 1, 3, 2, 4)
        # queries: Batch, Seq, Head, SubSeqLen, HeadDim

    def forward(self, hidden_states, attention_mask, rope_cache):
        # hidden_states: Batch, SeqLen, EmbedDim
        seq_len = hidden_states.shape[1]
        if seq_len < self.window_size:
            return super().forward(hidden_states, attention_mask, rope_cache)
        num_windows = seq_len // self.window_size
        last_window =  seq_len - self.window_size * num_windows
        # Handle the case when the seq len is not divisible by window size
        if last_window > 0:
            main_seq_len = seq_len - last_window
            main_part = self._mh_local(
                hidden_states[:, :main_seq_len, :],
                num_windows, attention_mask, rope_cache
            )
            last_part = super().forward(
                hidden_states[:, :last_window, :], attention_mask, rope_cache
            )
            return torch.cat([main_part, last_part], dim=1)

        return self._mh_local(hidden_states, num_windows, attention_mask, rope_cache)

    # TODO: no masking support yet
    def _mh_local(self, hidden_states, num_windows, attention_mask, rope_cache):
        size = hidden_states.size()
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_local_scores(mixed_query_layer, num_windows)
        key_layer = self.transpose_for_local_scores(mixed_key_layer, num_windows)
        value_layer = self.transpose_for_local_scores(mixed_value_layer, num_windows)

        if not self.apply_relpe_after:
            query_layer = rope_cache.apply_local_relpe2(
                query_layer, self.window_size, num_windows)
            key_layer = rope_cache.apply_local_relpe2(
                key_layer, self.window_size, num_windows)
        # Batch, Seq, Head, SubSeqLen, HeadDim
        #if torch.all(attention_mask == 0):
        attention_mask = None
        #kv = torch.matmul(key_layer.transpose(-1, -2), value_layer)
        #context_layer = torch.matmul(query_layer, kv)
        context_layer = self.attention_kernel(
            query_layer, key_layer, value_layer, attn_mask=attention_mask,
            dropout_p=self.dropout_prob, causal=self.causal, rope_cache=rope_cache
        )
        """
        context_layer = nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attn_mask=attention_mask,
            dropout_p=self.dropout_prob
        )
        """
        context_layer = context_layer.permute(0, 1, 3, 2, 4)
        # output: Batch, Seq, SeqLen, Head, HeadDim
        context_layer = context_layer.reshape(*size)
        # output: Batch, SeqLen, EmbedDim
        return context_layer

class BertSelfShiftedLocalAttention(BertSelfLocalAttention):
    def __init__(self, config: TransformerConfig, layer_number=1):
        super(BertSelfShiftedLocalAttention,
              self).__init__(config)
        # self.window_size = config.window_size
        assert self.window_size % 2 == 0 and self.window_size > 0
        if config.max_position_embeddings < self.window_size:
            raise ValueError(
                f"max_position_embeddings ({config.max_position_embeddings}) "
                f"should be at least equal to window_size ({self.window_size})."
            )
        else:
            self.left_pad = self.window_size // 2
            self.right_pad = self.window_size // 2

    def forward(self, hidden_states, attention_mask, rope_cache):
        # hidden_states: Batch, SeqLen, EmbedDim
        seq_len = hidden_states.shape[1]
        if seq_len <= self.window_size // 2:
            return super().forward(hidden_states, attention_mask, rope_cache)
        hidden_states = nn.functional.pad(
            hidden_states, pad=(0, 0, self.left_pad, self.right_pad))
        hidden_states = super().forward(hidden_states, attention_mask, rope_cache)
        return hidden_states[:, self.left_pad:-self.right_pad, :]

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size,
                               bias=config.attn_proj_biases)
        self.dense.bert_output_layer = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, rope_cache):
        self_output = self.self(input_tensor, attention_mask, rope_cache)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertLocalAttention(BertAttention):
    def __init__(self, config):
        super(BertLocalAttention, self).__init__(config)
        self.self = BertSelfLocalAttention(config)

class BertShiftedLocalAttention(BertAttention):
    def __init__(self, config):
        super(BertShiftedLocalAttention, self).__init__(config)
        self.self = BertSelfShiftedLocalAttention(config)

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = (ACT2FN[config.hidden_act]
            if (isinstance(config.hidden_act, str)
                and not config.hidden_act == "swiglu")
            else config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense.bert_output_layer = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

class BertSwigluUp(nn.Module):
    def __init__(self, config):
        super(BertSwigluUp, self).__init__()
        hidden_size = config.hidden_size
        n_hidden = int(2 * config.intermediate_size / 3)
        n_hidden = find_multiple(n_hidden, 128)

        self.c_fc1 = nn.Linear(hidden_size, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(hidden_size, n_hidden, bias=False)
        self.expansion_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.expansion_dropout(x)
        return x


class BertSwigluDown(nn.Module):
    def __init__(self, config):
        super(BertSwigluDown, self).__init__()
        hidden_size = config.hidden_size
        n_hidden = int(2 * config.intermediate_size / 3)
        n_hidden = find_multiple(n_hidden, 128)

        self.c_proj = nn.Linear(n_hidden, hidden_size, bias=False)
        self.contraction_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_proj(x)
        x = self.contraction_dropout(x)
        return x

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.PreAttentionLayerNorm = BertLayerNorm(config.hidden_size,
                                                   eps=1e-12)
        if config.pre_attn_ln_type != "default":
            self.PreAttentionLayerNorm = Activation2Class[
                config.pre_attn_ln_type](config.hidden_size, eps=1e-12)
        #self.MidAttentionLayerNorm = BertLayerNorm(config.hidden_size, eps = 1e-12)
        self.PostAttentionLayerNorm = BertLayerNorm(config.hidden_size,
                                                    eps=1e-12)
        if config.post_attn_ln_type != "default":
            self.PostAttentionLayerNorm = Activation2Class[
                config.post_attn_ln_type](config.hidden_size, eps=1e-12)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        if config.hidden_act == "swiglu":
            self.intermediate = BertSwigluUp(config)
            self.output = BertSwigluDown(config)
        # logic for local attention scheme
        if hasattr(config, 'local_scheme') and config.local_scheme and len(config.local_scheme.split("_"))==1:
            code = config.local_scheme.split('_')[0]
            valid_codes = {'g', 'l', 'sl', 'swa'}
            if code not in valid_codes:
                raise ValueError(f"Unknown attention type code '{code}' in local_scheme. "
                                        f"Valid codes are: {sorted(list(valid_codes))}")

            
            if code == 'l':
                self.attention.self = BertSelfLocalAttention(config)
            elif code == 'sl':
                self.attention.self = BertSelfShiftedLocalAttention(config)
            elif code == 'swa':
                layer_config = copy.deepcopy(config)
                layer_config.attention_kernel = "swa"
                self.attention.self = BertSelfAttention(layer_config)
            # 'g' is the default and requires no change, so we just pass.


        self._init_weights(config)

    def _init_weights(self, config):
        num_layers = config.num_hidden_layers
        base_std = config.initializer_range

        for module in self.modules():
            # Initialize linear layers
            if isinstance(module, nn.Linear):
                std = base_std
                # Match residual path scaling used previously via 'bert_output_layer'
                if hasattr(module, 'bert_output_layer'):
                    std = base_std / math.sqrt(2.0 * num_layers)
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            # Initialize layer norms
            elif isinstance(module, BertLayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, hidden_states, attention_mask, rope_cache, **kwargs):
        input_layer_norm = self.PreAttentionLayerNorm(hidden_states)
        attention_output = self.attention(input_layer_norm, attention_mask, rope_cache, **kwargs)
        #atention_output = self.MidAttentionLayerNorm(attention_output)
        intermediate_input = hidden_states + attention_output

        intermediate_layer_norm = self.PostAttentionLayerNorm(
            intermediate_input)
        intermediate_output = self.intermediate(intermediate_layer_norm)
        layer_output = self.output(intermediate_output)

        return layer_output + intermediate_input