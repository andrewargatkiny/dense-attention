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
from .attention_kernels import SoftmaxAttention, LinearAttention, SlidingWindowAttention

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese':
    "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


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
                 causal=False,
                 local_attention=False,
                 window_size=1024,
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
            self.embedding_dropout = embedding_dropout
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.attn_proj_biases = attn_proj_biases
            self.max_position_embeddings = max_position_embeddings
            self.pos_emb_type = PositionalEmbeddingsTypes[pos_emb_type.upper()]
            self.relpe_type = relpe_type
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.causal = causal
            self.local_attention = local_attention
            self.window_size = window_size
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


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config: TransformerConfig):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size)
        if config.pos_emb_type == PositionalEmbeddingsTypes.LEARNED:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                    config.hidden_size)
            self.add_pos_emb_fn = self.add_learned_posit_embeddings

        elif config.pos_emb_type == PositionalEmbeddingsTypes.SINUSOIDAL:
            self.position_embeddings = SinusoidalPositionalEncoding(
               config.max_position_embeddings, config.hidden_size
            )
            self.add_pos_emb_fn = self.add_sinusoidal_posit_embeddings
        else:
            self.add_pos_emb_fn = lambda x: x

        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
        #                                           config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.embedding_dropout)

    def add_learned_posit_embeddings(self, word_embeddings):
        seq_length = word_embeddings.size(1)
        position_ids = torch.arange(seq_length,
                                    dtype=torch.long,
                                    device=word_embeddings.device)
        position_ids = position_ids.unsqueeze(0)#.expand(word_embeddings.size(0), -1)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings + word_embeddings

    def add_sinusoidal_posit_embeddings(self, word_embeddings):
        return self.position_embeddings(word_embeddings) + word_embeddings

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # if token_type_ids is None:
        #    token_type_ids = torch.zeros_like(input_ids)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        words_embeddings = self.word_embeddings(input_ids)

        embeddings = self.add_pos_emb_fn(words_embeddings) #+ token_type_embeddings
        embeddings = self.LayerNorm(embeddings)

        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        if config.attention_kernel not in ["softmax", "linear", "swa"]:
            raise ValueError("Attention kernel param should hold value of "
                             "either 'softmax' or 'linear' or 'swa'.")
        if config.attention_kernel == "softmax":
            self.attention_kernel = SoftmaxAttention(config)
        elif config.attention_kernel == "swa":
            self.attention_kernel = SlidingWindowAttention(config)
        elif config.attention_kernel == "linear":
            self.attention_kernel = LinearAttention(config)
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
        query_layer = rope_cache.apply_relpe(query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        key_layer = rope_cache.apply_relpe(key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        #if torch.all(attention_mask == 0):
        attention_mask = None
        #kv = torch.matmul(key_layer.transpose(-1, -2), value_layer)
        #context_layer = torch.matmul(query_layer, kv)
        #attention_mask = None
        context_layer = self.attention_kernel(
            query_layer, key_layer, value_layer, attn_mask=attention_mask,
            dropout_p=self.dropout_prob, causal=self.causal
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
        query_layer = rope_cache.apply_local_relpe2(query_layer, self.window_size, num_windows)
        key_layer = self.transpose_for_local_scores(mixed_key_layer, num_windows)
        key_layer = rope_cache.apply_local_relpe2(key_layer, self.window_size, num_windows)
        value_layer = self.transpose_for_local_scores(mixed_value_layer, num_windows)
        # Batch, Seq, Head, SubSeqLen, HeadDim
        #if torch.all(attention_mask == 0):
        attention_mask = None
        #kv = torch.matmul(key_layer.transpose(-1, -2), value_layer)
        #context_layer = torch.matmul(query_layer, kv)
        context_layer = self.attention_kernel(
            query_layer, key_layer, value_layer, attn_mask=attention_mask,
            dropout_p=self.dropout_prob, causal=self.causal
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
        #self.MidAttentionLayerNorm = BertLayerNorm(config.hidden_size, eps = 1e-12)
        self.PostAttentionLayerNorm = BertLayerNorm(config.hidden_size,
                                                    eps=1e-12)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        if config.hidden_act == "swiglu":
            self.intermediate = BertSwigluUp(config)
            self.output = BertSwigluDown(config)

    def forward(self, hidden_states, attention_mask, rope_cache):
        input_layer_norm = self.PreAttentionLayerNorm(hidden_states)
        attention_output = self.attention(input_layer_norm, attention_mask, rope_cache)
        #atention_output = self.MidAttentionLayerNorm(attention_output)
        intermediate_input = hidden_states + attention_output

        intermediate_layer_norm = self.PostAttentionLayerNorm(
            intermediate_input)
        intermediate_output = self.intermediate(intermediate_layer_norm)
        layer_output = self.output(intermediate_output)

        return layer_output + intermediate_input


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()

        #Added later to make it similar to GPT-2
        self.FinalLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        if (config.pos_emb_type == PositionalEmbeddingsTypes.RELPE and
                config.relpe_type is not None):
            self.relpe_type = RelPEType[config.relpe_type.upper()]
        else:
            self. relpe_type = RelPEType.DUMMY
        self.rope_cache = RelPETypeToClass[self.relpe_type](
            config.max_position_embeddings, #args.max_seq_length
            config.hidden_size // config.num_attention_heads,
            #num_heads=config.num_attention_heads
        )

        layer = BertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

        if config.local_attention:
            for i, layer in enumerate(self.layer):
                if i % 3 == 0:
                    layer.attention.self = BertSelfLocalAttention(config)
                elif i % 3 == 1:
                    layer.attention.self = BertSelfShiftedLocalAttention(config)

    def forward(self,
                hidden_states,
                attention_mask,
                output_all_encoded_layers=True):

        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, self.rope_cache)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            hidden_states = self.FinalLayerNorm(hidden_states)
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if config.hidden_act is None or config.hidden_act == "swiglu":
            self.transform_act_fn = ACT2FN["gelu"]
        else:
            self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(
            torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states, masked_token_indexes):
        if masked_token_indexes is not None:
            hidden_states = torch.index_select(
                hidden_states.view(-1, hidden_states.shape[-1]), 0,
                masked_token_indexes)
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config,
                                                bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, num_labels=2):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config,
                                                bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, num_labels)

    def forward(self, sequence_output, pooled_output,
                masked_token_indexes=None):
        prediction_scores = self.predictions(sequence_output,
                                             masked_token_indexes)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        self.config = config

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.word_embeddings.weight.numel()
            n_params -= self.embeddings.position_embeddings.weight.numel()
            n_params -= self.embeddings.token_type_embeddings.weight.numel()
        return n_params

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        logger.info("Init BERT weights")
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            num_layers = self.config.num_hidden_layers
            std = self.config.initializer_range
            if hasattr(module, 'bert_output_layer'):
                if torch.distributed.get_rank() == 0:
                    print("Accounting for accumulation on the residual path")
                    std = self.config.initializer_range / math.sqrt(
                        2.0 * num_layers)
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name,
                        state_dict=None,
                        cache_dir=None,
                        *inputs,
                        **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        try:
            raise NotImplementedError("Loading weights by this way is not supported.")
            #resolved_archive_file = cached_path(archive_file,
            #                                    cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = TransformerConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                                         True, missing_keys, unexpected_keys,
                                         error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".
                format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a ModelConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.ModelConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
        logger.info("Init BERT pretrain model")

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class TransformerForPreTraining(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a ModelConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = ModelConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config: TransformerConfig, args):
        super(TransformerForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=args.num_labels)
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)
        self.num_labels = args.num_labels
        self.window_size = config.window_size

        self.head = self.mlm_cls_head
        if args.only_mlm_task and args.only_cls_task:
            raise ValueError("Only one of the options 'only_mlm_task' "
                             "and 'only_cls_task' should hold True.")
        if args.only_mlm_task:
            self.head = self.mlm_head
        elif args.only_cls_task:
            self.head = self.cls_head
        self.apply(self.init_bert_weights)
        self.args = args

    def mlm_cls_head(self,
                     masked_lm_labels: torch.Tensor,
                     masked_token_indexes: torch.Tensor,
                     prediction_scores: torch.Tensor,
                     seq_relationship_score: torch.Tensor,
                     label: torch.Tensor):
        """Head for both MLM and classification tasks"""
        target = torch.index_select(masked_lm_labels.view(-1), 0,
                                    masked_token_indexes)
        masked_lm_loss = self.loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), target
        )
        next_sentence_loss = self.loss_fct(
            seq_relationship_score.view(-1, self.num_labels), label.view(-1)
        )
        total_loss = masked_lm_loss + next_sentence_loss
        if not self.training:
            return (masked_lm_loss, next_sentence_loss, target,
                    prediction_scores, seq_relationship_score)
        return total_loss

    def mlm_head(self,
                 masked_lm_labels: torch.Tensor,
                 masked_token_indexes: torch.Tensor,
                 prediction_scores: torch.Tensor,
                 seq_relationship_score: torch.Tensor,
                 label: torch.Tensor):
        """Head for only MLM task. Doesn't return any data for cls."""
        target = torch.index_select(masked_lm_labels.view(-1), 0,
                                    masked_token_indexes)
        masked_lm_loss = self.loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), target
        )
        if not self.training:
            return masked_lm_loss, target, prediction_scores
        return masked_lm_loss


    def cls_head(self,
                 masked_lm_labels: torch.Tensor,
                 masked_token_indexes: torch.Tensor,
                 prediction_scores: torch.Tensor,
                 seq_relationship_score: torch.Tensor,
                 label: torch.Tensor):
        """Head for only classification task. Doesn't return any data for MLM."""
        next_sentence_loss = self.loss_fct(
            seq_relationship_score.view(-1, self.num_labels), label.view(-1)
        )
        if not self.training:
            return next_sentence_loss, seq_relationship_score
        return next_sentence_loss

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                masked_lm_labels=None, label=None, log=True):

        sequence_output, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
        )

        if masked_lm_labels is None:
            # In eval mode calculate all output representations for
            # compatibility with HuggingFace's Bert.
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output,
            )
            return prediction_scores, seq_relationship_score

        # filter out all masked labels.
        masked_token_indexes = torch.nonzero(
            (masked_lm_labels + 1).view(-1)).view(-1)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output,
            masked_token_indexes)
        return self.head(
            masked_lm_labels, masked_token_indexes, prediction_scores,
            seq_relationship_score, label
        )


class BertForMaskedLM(PreTrainedBertModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a ModelConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = ModelConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config,
                                   self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids,
                                       token_type_ids,
                                       attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class BertForNextSentencePrediction(PreTrainedBertModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a ModelConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = ModelConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                next_sentence_label=None):
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2),
                                          next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class TransformerForSequenceClassification(PreTrainedBertModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a ModelConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = ModelConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, args):
        super(TransformerForSequenceClassification, self).__init__(config)
        self.num_labels = args.num_labels
        self.window_size = config.window_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                label=None,
                attention_mask=None,
                token_type_ids=None):
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if label is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
            if not self.training:
                return loss, logits
            return loss
        else:
            return logits


class TransformerForRegression(PreTrainedBertModel):

    def __init__(self, config, args):
        super(TransformerForRegression, self).__init__(config)
        self.num_labels = args.num_labels
        self.window_size = config.window_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1)
        """
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=args.num_labels)
        self.classifier = self.cls.seq_relationship
        """

        self.apply(self.init_bert_weights)
        self.use_local_attention = config.local_attention

    def forward(self,
                input_ids,
                label=None,
                attention_mask=None,
                token_type_ids=None,
                checkpoint_activations=False):
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask=attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)

        if label is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), label.to(logits.dtype).view(-1))
            if not self.training:
                return loss, logits
            return loss
        else:
            return logits

class TransformerForAANMatching(PreTrainedBertModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a ModelConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = ModelConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, args):
        super(TransformerForAANMatching, self).__init__(config)
        self.num_labels = args.num_labels
        self.window_size = config.window_size
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU(approximate='tanh')
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                input_ids2,
                attention_mask=None,
                attention_mask2=None,
                label=None,
                token_type_ids=None,
                checkpoint_activations=False):
        checkpoint_activations = False

        _, pooled_output1 = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask=attention_mask,
                                     output_all_encoded_layers=False)
        _, pooled_output2 = self.bert(input_ids2,
                                     token_type_ids,
                                     attention_mask=attention_mask2,
                                     output_all_encoded_layers=False)
        hidden_states = torch.cat(
            [pooled_output1, pooled_output2,
            pooled_output1 * pooled_output2, pooled_output1 - pooled_output2],
            dim=-1)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        if label is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
            if not self.training:
                return loss, logits
            return loss
        else:
            return logits

class BertForMultipleChoice(PreTrainedBertModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a ModelConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = ModelConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = BertForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_choices=2):
        super(BertForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.bert(flat_input_ids,
                                     flat_token_type_ids,
                                     flat_attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits


class BertForTokenClassification(PreTrainedBertModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a ModelConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = ModelConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels=2):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None):
        sequence_output, _ = self.bert(input_ids,
                                       token_type_ids,
                                       attention_mask,
                                       output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForQuestionAnswering(PreTrainedBertModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a ModelConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = ModelConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                start_positions=None,
                end_positions=None):
        sequence_output, _ = self.bert(input_ids,
                                       token_type_ids,
                                       attention_mask,
                                       output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits
