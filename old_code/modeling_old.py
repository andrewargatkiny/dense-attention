# This file's is needed n case there will be problems with reproducibility
# of the old models inference results.

# DeepSpeed note, code taken from commit 3d59216cec89a363649b4fe3d15295ba936ced0f
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/modeling.py

# coding=utf-8
# Copyright 2023 Andrew Argatkiny
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

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from old_code.turing.file_utils import cached_path

from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init

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
TF_WEIGHTS_NAME = 'model.ckpt'


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


@torch.jit.script
def f_gelu(x):
    pdtype = x.dtype
    x = x.float()
    y = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return y.to(pdtype)


@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


@torch.jit.script
def bias_tanh(bias, y):
    x = bias + y
    return torch.tanh(x)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return f_gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "tanh": torch.nn.functional.tanh}


class LinearActivation(Module):
    r"""Fused Linear and activation Module.
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, act='gelu', bias=False):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fused_gelu = False
        self.fused_tanh = False
        if isinstance(act, str) or (sys.version_info[0] == 2
                                    and isinstance(act, unicode)):
            if bias and act == 'gelu':
                self.fused_gelu = True
            elif bias and act == 'tanh':
                self.fused_tanh = True
            else:
                self.act_fn = ACT2FN[act]
        else:
            self.act_fn = act
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # init.normal_(self.weight, mean=0., std=1/1024)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.fused_gelu:
            return bias_gelu(self.bias, F.linear(input, self.weight, None))
        elif self.fused_tanh:
            return bias_tanh(self.bias, F.linear(input, self.weight, None))
        else:
            return self.act_fn(F.linear(input, self.weight, self.bias))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class ModelConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 **kwargs
                 ):
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
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file,
                      str) or (sys.version_info[0] == 2 and isinstance(
            vocab_size_or_config_json_file, unicode)):
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
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `ModelConfig` from a Python dictionary of parameters."""
        config = ModelConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
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


try:
    import apex
    # apex.amp.register_half_function(apex.normalization.fused_layer_norm, 'FusedLayerNorm')
    import apex.normalization

    # apex.amp.register_float_function(apex.normalization.FusedLayerNorm, 'forward')
    BertLayerNorm = apex.normalization.FusedLayerNorm
except ImportError:
    print(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex."
    )
finally:
    class StandardLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            My code change: don't use biases.
            """
            super(StandardLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x, pad_adjust=1.):
            pdtype = x.dtype
            x = x.float()
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x.to(pdtype)  # + self.bias


    class UncenteredLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12, init_mean=1.0):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            Code changes:
            1) Mean and std over all numbers in a batch, not only one embedding
            2) When calculating mean and std, ignore all-zero `pad` token
            embeddings by multiplying mean and std by adjustment coefficient
            3) Don't center resulting vectors, just divide by std.
            """
            super(UncenteredLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size) * init_mean)
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x: torch.Tensor, attention_mask=None):
            """pad_adjust is coefficient that accounts for zero vectors of
            pad tokens which should be ignored in the calculation"""
            pdtype = x.dtype
            x = x.float()
            """
            mask, pad_adjust = attention_mask
            
            #u = x.mean(-1, keepdim=True)
            u = x.mean() * pad_adjust
            #s = (x - u).pow(2).mean(-1, keepdim=True)
            s = x.pow(2).mean() * pad_adjust - u.pow(2)
            """
            s = 1.0 / (x.absolute().mean(-1, keepdim=True) + self.variance_epsilon)
            x = x * s
            return self.weight * x.to(pdtype)  # * mask #+ self.bias


    BertLayerNorm = UncenteredLayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.Hardtanh()
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length,
                                    dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        # embeddings = embeddings / embeddings.abs().max(axis=-1, keepdim=True)[0]
        # embeddings = embeddings * attention_mask[0]
        embeddings = self.LayerNorm(embeddings)
        # embeddings = clip_grad_values(embeddings)
        # embeddings = self.LayerNorm(embeddings, attention_mask)
        # embeddings = self.dropout(embeddings)
        return embeddings

    def forward_unpadded(self, input_ids, lengths, token_type_ids=None):
        position_ids = torch.cat(
            [torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
             for seq_length in lengths],
            dim=0
        )
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class DenseAttention(nn.Module):
    """ No dropout, no biases"""

    def __init__(self, config, layer_number=1):
        super().__init__()

        self.n_heads = config.num_attention_heads
        # Number of Head Groups and heads in a Head Group. 'HeadGroup' in
        # the comments below.
        self.sqrt_n_heads = int(math.sqrt(self.n_heads))
        # Hidden size is 'EmbedDim' in the comments below.
        if config.hidden_size % self.sqrt_n_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the "
                f"root of number of"
                f" attention heads ({config.num_attention_heads})"
            )
        self.hidden_size = config.hidden_size
        # 'Head' in the comments below.
        self.head_size = int(self.hidden_size / math.sqrt(self.n_heads))

        # self.scaler = nn.Parameter(torch.full((self.head_size,), 1.0))

        # First 3 dims represent X split vertically into `sqrt_n_heads` parts
        # into a tensor of size (sqrt_n_heads, batch_size, head_size). Each of
        # splits is a foundation for a so-called Head Group. Last dimension is
        # for turning each X part into `sqrt_n_heads` key and 1 value matrices.
        # Multiplying the quantity of X parts by a part's contents, we get
        # `n_heads` different queries (like in original attention) and
        # `sqrt_n_heads` values.
        #
        # Dims: # HeadGroup, Head, Head * HeadGroup
        queries = torch.Tensor(self.sqrt_n_heads, self.head_size,
                               self.head_size * self.sqrt_n_heads)
        self.queries = nn.Parameter(queries)
        std = config.initializer_range  # / math.sqrt(config.num_hidden_layers)
        # std = math.log(self.head_size - 1) ** 0.5
        mu = - math.log(self.head_size) - std ** 2 / 2
        min_val = math.exp(mu - 3 * std)
        max_val = math.exp(mu + 3 * std)
        # self.qv.data.log_normal_(mean=mu, std=std).clamp_(min_val, max_val)
        torch.nn.init.normal_(self.queries, mean=0, std=std)
        num_layers = config.num_hidden_layers
        # Adjust q weights to compensate for softmax absence.
        adjust_q = torch.ones(self.sqrt_n_heads, self.head_size,
                              self.hidden_size) / math.sqrt(2.0 * num_layers * self.head_size)
        self.queries.data.mul_(adjust_q)
        """
        torch.nn.init.uniform_(
            self.qv,
            a=-config.initializer_range / math.sqrt(config.num_hidden_layers),
            b=config.initializer_range / math.sqrt(config.num_hidden_layers)
        )
        """
        self.softmax = nn.Softmax(dim=-1)
        # A linear layer that combines representations of all heads in a head
        # group into projection of a head's size.
        # Dims: HeadGroup, EmbedDim, HeadDim
        self.combiners = nn.Parameter(
            torch.Tensor(self.sqrt_n_heads, self.head_size * self.sqrt_n_heads,
                         self.head_size)
        )
        torch.nn.init.normal_(self.combiners, mean=0, std=std / math.sqrt(2.0 * num_layers * self.head_size))
        # self.combiners.data.log_normal_(mean=mu, std=std).clamp_(min_val, max_val)
        """
        torch.nn.init.uniform_(
            self.combiners,
            a=-config.initializer_range / math.sqrt(config.num_hidden_layers),
            b=config.initializer_range / math.sqrt(config.num_hidden_layers)
        )
        """

    def forward(self, hidden_states: torch.Tensor):
        bs, seqlen, embeddim = hidden_states.size()
        new_size = hidden_states.size()[:-1] + (self.sqrt_n_heads, self.head_size)
        hidden_states = (hidden_states
                         # Batch, SeqLen, EmbedDim
                         .view(new_size)
                         # Batch, SeqLen, HeadGroup, Head
                         )  # * self.scaler

        projected_q = (torch.matmul(hidden_states
                                    .view(bs * seqlen, self.sqrt_n_heads, self.head_size)
                                    # Batch * SeqLen, HeadGroup, Head
                                    .permute(1, 0, 2),
                                    # HeadGroup, Batch * SeqLen, Head
                                    self.queries)
                       # HeadGroup_1, Batch * SeqLen, Head * HeadGroup_2
                       .view(self.sqrt_n_heads, bs, seqlen,
                             self.sqrt_n_heads, self.head_size)
                       # HeadGroup_1, Batch, SeqLen, HeadGroup_2, Head
                       # .permute(1, 0, 2, 3, 4)
                       # Batch, HeadGroup, SeqLen, HeadGroup, Head
                       )
        queries = (projected_q
                   # HeadGroup_1, Batch, SeqLen, HeadGroup_2, Head
                   .permute(3, 1, 2, 0, 4)
                   # HeadGroup_2, Batch, SeqLen, HeadGroup_1, Head
                   .contiguous()
                   .view(self.sqrt_n_heads, bs, seqlen * self.sqrt_n_heads,
                         self.head_size)
                   # HeadGroup_2, Batch, SeqLen * HeadGroup_1, Head
                   )
        keys = hidden_states.permute(2, 0, 3, 1)
        # keys: HeadGroup_1, Batch, Head, SeqLen

        pre_attn = torch.matmul(queries, keys)
        # pre_attn: HeadGroup_2_1, Batch, SeqLen * HeadGroup_1, SeqLen
        # pre_attn = self.softmax(pre_attn)

        values = hidden_states.permute(2, 0, 1, 3)
        # values: HeadGroup_1, Batch, SeqLen, Head

        attention = (torch.matmul(pre_attn, values)
                     # HeadGroup_2_1_1, Batch, SeqLen * HeadGroup_1, Head
                     .view(self.sqrt_n_heads, bs, seqlen, self.sqrt_n_heads, self.head_size)
                     # HeadGroup_2_1_1, Batch, SeqLen, HeadGroup_1, Head
                     # .view(self.sqrt_n_heads, bs * seqlen, -1)
                     # HeadGroup_2_1_1, Batch * SeqLen, HeadGroup_1 * Head
                     .permute(3, 1, 2, 0, 4)
                     # HeadGroup, Batch, SeqLen, HeadGroup, Head
                     .reshape(self.sqrt_n_heads, bs * seqlen, -1)
                     # HeadGroup, Batch * SeqLen, HeadGroup * Head
                     )

        output = (torch.matmul(attention, self.combiners)
                  # HeadGroup, Batch * SeqLen, Head
                  .permute(1, 0, 2)
                  # Batch * SeqLen, HeadGroup, Head
                  .reshape(bs, seqlen, -1)
                  # Batch, SeqLen, EmbedDim
                  )
        """
        if not torch.all(torch.isfinite(output)):
            print("hidden states:", hidden_states.max(), hidden_states.min())
            print("projected q", projected_q.max(), projected_q.min())
            print("pre_attn:", pre_attn.max(), pre_attn.min())
            print("attention:", attention.max(), attention.min())
            print("output:", output.max(), output.min())
        """
        return output


class DenseAttentionMultiHead(nn.Module):
    """ DenseAttention with one dense projection operation and multiple heads"""

    def __init__(self, config, layer_number=1):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = int(self.hidden_size / self.n_heads)
        num_layers = config.num_hidden_layers
        std = config.initializer_range
        self.layer_number = layer_number
        self.queries = nn.Parameter(
            torch.zeros(self.hidden_size, self.hidden_size)
        )
        torch.nn.init.normal_(self.queries, mean=0,
                              std=std / math.sqrt(2.0 * num_layers * self.hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: Batch, SeqLen, EmbedDim
        queries = F.linear(hidden_states, self.queries)  # * self.norm_ratio_queries)
        # queries: Batch, SeqLen, EmbedDim
        size = hidden_states.size()
        new_size = hidden_states.size()[:-1] + (self.n_heads, self.head_size)
        queries = queries.view(new_size)
        # queries: Batch, SeqLen, Head, HeadDim
        queries = queries.permute(0, 2, 1, 3)
        # queries: Batch, Head, SeqLen, HeadDim

        hidden_states = hidden_states.view(new_size)
        # hidden_states: Batch, SeqLen, Head, HeadDim
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        # hidden_states: Batch, Head, SeqLen, HeadDim
        keys = hidden_states.transpose(2, 3)
        # keys: Batch, Head, HeadDim, SeqLen
        pre_attn = torch.matmul(queries, keys)
        # pre_attn: Batch, Head, SeqLen, Seqlen
        attention = torch.matmul(pre_attn, hidden_states)
        # attention: Batch, Head, SeqLen, HeadDim

        output = attention.permute(0, 2, 1, 3)
        # output: Batch, SeqLen, Head, HeadDim
        output = output.reshape(*size)
        # output: Batch, SeqLen, EmbedDim
        return output


class DenseAttentionMultiHeadLinComplexity(DenseAttentionMultiHead):
    """ DenseAttention with one dense projection operation and multiple heads
    with O(d^2*n) time complexity"""

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: Batch, SeqLen, EmbedDim
        # queries: Batch, SeqLen, EmbedDim
        size = hidden_states.size()
        new_size = size[:-1] + (self.n_heads, self.head_size)
        queries = F.linear(hidden_states, self.queries)  # * self.norm_ratio_queries)
        queries = queries.view(new_size)
        # queries: Batch, SeqLen, Head, HeadDim
        queries = queries.permute(0, 2, 1, 3)
        # queries: Batch, Head, SeqLen, HeadDim

        hidden_states = hidden_states.view(new_size)
        # hidden_states: Batch, SeqLen, Head, HeadDim
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        # hidden_states: Batch, Head, SeqLen, HeadDim
        keys = hidden_states.transpose(2, 3)
        # keys: Batch, Head, HeadDim, SeqLen

        pre_attn = torch.matmul(keys, hidden_states)
        # pre_attn: Batch, Head, HeadDim, HeadDim
        attention = torch.matmul(queries, pre_attn)
        # attention: Batch, Head, SeqLen, HeadDim

        output = attention.permute(0, 2, 1, 3)
        # output: Batch, SeqLen, Head, HeadDim
        output = output.reshape(*size)
        # output: Batch, SeqLen, EmbedDim
        return output


class DenseAttentionOneHead(nn.Module):
    """ Simplified implementation of DenseAttention equivalent to it in case
     there is only one head"""

    def __init__(self, config, layer_number=1):
        super().__init__()
        self.hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        std = config.initializer_range
        self.layer_number = layer_number
        self.queries = nn.Parameter(
            torch.zeros(self.hidden_size, self.hidden_size)
        )
        torch.nn.init.normal_(self.queries, mean=0,
                              std=std / math.sqrt(2.0 * num_layers * self.hidden_size))

        # self.combiners = nn.Parameter(
        #    torch.eye(self.hidden_size, self.hidden_size)
        # )
        # torch.nn.init.normal_(self.combiners, mean=0,
        #                      std=std / math.sqrt(2.0 * num_layers * self.hidden_size)) 
        # correction_ratio = (num_layers - layer_number + 1) / num_layers
        # correction_ratio = layer_number / num_layers
        correction_ratio = 1. / math.sqrt(layer_number)
        self.default_norm_queries = 1. / math.sqrt(self.hidden_size)  # * correction_ratio ** 2
        # self.default_norm_combiners = 1. / math.sqrt(self.hidden_size) * correction_ratio
        self.norm_ratio_queries = self.default_norm_queries / self.queries.abs().max().item()
        # self.norm_ratio_combiners = self.default_norm_combiners/ self.combiners.abs().max().item()

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: Batch, SeqLen, EmbedDim
        queries = F.linear(hidden_states, self.queries)  # * self.norm_ratio_queries)
        # queries = hidden_states
        # queries: Batch, SeqLen, EmbedDim
        keys = hidden_states.transpose(1, 2)
        # keys: Batch, EmbedDim, SeqLen
        pre_attn = torch.matmul(queries, keys)
        # pre_attn: Batch, SeqLen, Seqlen
        attention = torch.matmul(pre_attn, hidden_states)
        # attention: Batch, SeqLen, EmbedDim
        # output = F.linear(attention, self.combiners) #* self.norm_ratio_combiners)
        # output: Batch, SeqLen, EmbedDim
        return attention  # output


class DenseAttentionOneHeadLinComplexity(DenseAttentionOneHead):
    """ Simplified implementation of DenseAttention equivalent to it in case
     there is only one head but with altered order of operations which yields
     linear complexity"""

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: Batch, SeqLen, EmbedDim
        queries = F.linear(hidden_states, self.queries)  # * self.norm_ratio_queries)
        # queries: Batch, SeqLen, EmbedDim
        keys = hidden_states.transpose(-1, -2)
        # keys: Batch, EmbedDim, SeqLen
        pre_attn = torch.matmul(keys, hidden_states)
        # pre_attn: Batch, EmbedDim, EmbedDim
        attention = torch.matmul(queries, pre_attn)
        # attention: Batch, SeqLen, EmbedDim
        # output = F.linear(attention, self.combiners) #* self.norm_ratio_combiners)
        # output: Batch, SeqLen, EmbedDim
        return attention  # output


class FFN(nn.Module):
    def __init__(self, config: ModelConfig, num_dense_layers=4):
        super(FFN, self).__init__()
        self.layers = nn.ModuleList([DenseLayer(config) for _ in range(num_dense_layers)])

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class ExpandedFFN(nn.Module):
    """ A two-layers Feed Forward Network with expanding and contracting layers
     similar to original Transformer block's FFN, but with arbitrary natural
    `expansion_factor`,  without biases and with a property that it
    additionally acts as if it adds input hidden states to the output of FFN.
    The last trait is achieved by initialization of weights with identity
    matrices divided by expansion_factor."""

    def __init__(self, config):
        super(ExpandedFFN, self).__init__()
        self.hidden_size = config.hidden_size
        self.expansion_factor = config.intermediate_size

        self.expanding_weight = nn.Parameter(self._init_weights())
        self.activation = nn.ReLU()
        self.contracting_weight = nn.Parameter(
            self._init_weights(expansion_flag=False)
        )

        self.default_norm_expand = 1 / math.sqrt(self.hidden_size * self.expansion_factor * 2)
        self.default_norm_contract = 1 / math.sqrt(self.hidden_size * self.expansion_factor * 2)
        self.norm_ratio_expand = self.default_norm_expand / self.expanding_weight.abs().max().item()
        self.norm_ratio_contract = self.default_norm_contract / self.contracting_weight.abs().max().item()

    def _init_weights(self, expansion_flag=True):
        if expansion_flag:
            repeat_dims = (1, self.expansion_factor)
            divisor = self.expansion_factor
        else:
            repeat_dims = (self.expansion_factor, 1)
            divisor = 1.
        identity_matrix = torch.eye(self.hidden_size)
        identity_blocks = identity_matrix.repeat(*repeat_dims)
        identity_blocks = identity_blocks / divisor

        noise_std = 1 / math.sqrt(self.hidden_size * self.expansion_factor * 2)
        noise = torch.randn_like(identity_blocks) * noise_std

        return noise  # identity_blocks #+ noise

    def prepare_for_inference(self):
        norm_ratio_expand = (self.default_norm_expand /
                             self.expanding_weight.abs().max().item())
        self.expanding_weight.data.mul_(norm_ratio_expand)
        norm_ratio_contract = (self.default_norm_contract /
                               self.contracting_weight.abs().max().item())
        self.contracting_weight.data.mul_(norm_ratio_contract)
        self.forward = self.inference_forward

    def forward(self, hidden_states):
        hidden_states = torch.matmul(hidden_states, self.expanding_weight * self.norm_ratio_expand)
        hidden_states = self.activation(hidden_states)
        hidden_states = torch.matmul(hidden_states, self.contracting_weight * self.norm_ratio_contract)
        return hidden_states

    def inference_forward(self, hidden_states):
        hidden_states = torch.matmul(hidden_states, self.expanding_weight)
        hidden_states = self.activation(hidden_states)
        hidden_states = torch.matmul(hidden_states, self.contracting_weight)
        return hidden_states


class MaxNormActivation(nn.Module):
    """Activation that divides the input embeddings by their max norm."""

    def __init__(self, config):
        super(MaxNormActivation, self).__init__()

    def forward(self, hidden_states):
        hidden_states = hidden_states / (hidden_states.abs().max(axis=-1, keepdim=True)[0] + 1e-3)
        return hidden_states


class BertLayerWithActivation(nn.Module):
    def __init__(self, config: ModelConfig, layer_number):
        super(BertLayerWithActivation, self).__init__()
        self.activation = MaxNormActivation(config)
        self.attention = DenseAttentionMultiHeadLinComplexity(config, layer_number)
        # self.post_activation = MaxNormActivation(config)
        self.ffn = ExpandedFFN(config)
        self.ffn_activation = MaxNormActivation(config)

    def forward(self, hidden_states, attention_mask):
        prev_hidden_states = hidden_states
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states * attention_mask
        hidden_states = self.attention(hidden_states)
        # hidden_states = self.post_activation(hidden_states)
        # prev_hidden_states = hidden_states
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_activation(hidden_states)
        hidden_states = hidden_states + prev_hidden_states
        return hidden_states

    def forward_unpadded(self, hidden_states, attention_mask):
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


class BertEncoder(nn.Module):
    def __init__(self, config, args, sparse_attention_config=None):
        super(BertEncoder, self).__init__()

        # Added later to make it similar to GPT-2
        self.FinalLayerNorm = BertLayerNorm(config.hidden_size)
        layers = [BertLayerWithActivation(config, layer_number=n + 1)
                  if n % 1 == 0
                  else BertLayerWithActivation(config, layer_number=n + 1)
                  for n in range(config.num_hidden_layers)]
        self.layer = nn.ModuleList(layers)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask,
                output_all_encoded_layers=True,
                checkpoint_activations=False):
        all_encoder_layers = []

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            # if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)

        # if not output_all_encoded_layers or checkpoint_activations:
        hidden_states = self.FinalLayerNorm(hidden_states, attention_mask)
        all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense_act = LinearActivation(config.hidden_size,
                                          config.hidden_size,
                                          act="tanh")

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense_act(first_token_tensor)
        return pooled_output

    def forward_unpadded(self, hidden_states, cs_lengths):
        #first_token_tensor = hidden_states[cs_lengths]
        first_token_tensor = hidden_states.index_select(dim=0, index=cs_lengths)
        pooled_output = self.dense_act(first_token_tensor)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense_act = LinearActivation(config.hidden_size,
                                          config.hidden_size,
                                          act=config.hidden_act)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
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
        # self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(
            torch.zeros(bert_model_embedding_weights.size(0)))
        self.activation = nn.Hardtanh(-20, 2)

    def forward(self, hidden_states, masked_token_indexes):
        hidden_states = self.transform(hidden_states)

        if masked_token_indexes is not None:
            hidden_states = torch.index_select(
                hidden_states.view(-1, hidden_states.shape[-1]), 0,
                masked_token_indexes)

        torch.cuda.nvtx.range_push(
            "decoder input.size() = {}, weight.size() = {}".format(
                hidden_states.size(), self.decoder.weight.size()))
        hidden_states = self.decoder(hidden_states) + self.bias
        torch.cuda.nvtx.range_pop()
        return self.activation(hidden_states)


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
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config,
                                                bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self,
                sequence_output,
                pooled_output,
                masked_token_indexes=None):
        prediction_scores = self.predictions(sequence_output,
                                             masked_token_indexes)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, ModelConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `ModelConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__))
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
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            num_layers = self.config.num_hidden_layers
            std = self.config.initializer_range  # / math.sqrt(3)
            head_size = int(self.config.hidden_size /
                            math.sqrt(self.config.num_attention_heads))
            # std = math.log(head_size - 1) ** 0.5 / math.sqrt(3)
            mu = (- math.log(head_size) - std ** 2 / 2) / 3
            min_val = math.exp(mu - 3 * std)
            max_val = math.exp(mu + 3 * std)
            if hasattr(module, 'bert_output_layer'):
                # "Accounting for accumulation on the residual path"
                # print("Accounting for accumulation on Nthe residual path")
                std = self.config.initializer_range / math.sqrt(
                    2.0 * num_layers)
            module.weight.data.normal_(mean=0.0 / 3, std=std)
            # module.weight.data.uniform_(-std, std)
            # module.weight.data.log_normal_(mean=mu, std=std).clamp_(min_val, max_val)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            # module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        state_dict=None,
                        cache_dir=None,
                        from_tf=False,
                        *inputs,
                        **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
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
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[
                pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file,
                                                cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
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
        config = ModelConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(
                weights_path,
                map_location='cpu' if not torch.cuda.is_available() else None)
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
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

        start_prefix = ''
        if not hasattr(model, 'bert') and any(
                s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".
                    format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError(
                'Error(s) in loading state_dict for {}:\n\t{}'.format(
                    model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


class BertModel(BertPreTrainedModel):
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
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

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

    def __init__(self, config: ModelConfig, args=None):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        # set pad_token_id that is used for sparse attention padding
        self.pad_token_id = config.pad_token_id if hasattr(
            config, 'pad_token_id') and config.pad_token_id is not None else 0
        # set sparse_attention_config if it has been selected
        self.sparse_attention_config = None  # get_sparse_attention_config(
        #     args, config.num_attention_heads)
        # self.sparse_attention_utils = get_sparse_attention_utils(self.sparse_attention_config)
        self.encoder = BertEncoder(
            config, args, sparse_attention_config=self.sparse_attention_config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
        logger.info("Init BERT pretrain model")
        logger.info(f"Total parameters in transformer blocks: {self.get_num_params(non_embedding=False)}")

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                output_all_encoded_layers=False,
                checkpoint_activations=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        """
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
        """
        # If BertEncoder uses sparse attention, it needs to be padded based on the sparse attention block size
        embedding_output = self.embeddings(input_ids,
                                           attention_mask,
                                           token_type_ids)
        encoded_layers = self.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            checkpoint_activations=checkpoint_activations)
        encoded_layers = [embedding_output] + encoded_layers
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        # If BertEncoder uses sparse attention, and input_ids were padded, sequence output needs to be unpadded to original length
        # if not output_all_encoded_layers:
        encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

    def forward_unpadded(self, input_ids, token_type_ids,
                       scalers, lengths,
                       output_all_encoded_layers=False,
                       checkpoint_activations=False):
        lengths_list = lengths.tolist()
        embedding_output = self.embeddings(input_ids,
                                           lengths_list,
                                           token_type_ids)
        #cs_lengths = list(accumulate([0] + lengths_list[:-1]))
        cs_lengths = torch.cat([
            torch.tensor([0], dtype=lengths.dtype, device=lengths.device),
            lengths[:-1]
        ], dim=0).cumsum(dim=0)

        encoded_layers = self.encoder(
            embedding_output,
            (scalers, lengths_list),
            output_all_encoded_layers=output_all_encoded_layers,
            checkpoint_activations=checkpoint_activations)
        encoded_layers = [embedding_output] + encoded_layers
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output, cs_lengths)
        encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForPreTrainingNewAttention(BertPreTrainedModel):
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
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
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

    def __init__(self, config: ModelConfig, args):
        super(BertForPreTrainingNewAttention, self).__init__(config)
        self.bert = BertModel(config, args)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        if args.unpad_inputs:
            for i in range(config.num_hidden_layers):
                self.bert.embeddings.forward = self.bert.embeddings.forward_unpadded
                self.bert.encoder.layer[i].forward = (
                    self.bert.encoder.layer[i].forward_unpadded
                )
            self.bert.forward = self.bert.forward_unpadded
            self.bert.pooler.forward = self.bert.pooler.forward_unpadded
            self.forward = self.forward_unpadded
        self.apply(self.init_bert_weights)
        self.args = args

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                masked_lm_labels=None, label=None, log=True):
        checkpoint_activations = False
        # pad_attention = attention_mask.numel() / attention_mask.sum()
        dtype = self.bert.embeddings.word_embeddings.weight.dtype
        extended_attention_mask = (attention_mask / attention_mask
                                   .sum(axis=-1, keepdim=True)
                                   .pow(1. / 3)).to(dtype).unsqueeze(-1)
        encoded_layers, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            extended_attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations
        )
        # if isinstance(encoded_layers, (list, tuple)):
        #     sequence_output = encoded_layers[-1]
        # else:
        sequence_output = encoded_layers
        if not self.training:
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

        target = torch.index_select(masked_lm_labels.view(-1), 0,
                                    masked_token_indexes)
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), target)
        next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2),
                                      next_sentence_label.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss

    def forward_unpadded(self, batch, log=True):
        attention_mask = batch[2]
        masked_lm_labels = batch[5]
        next_sentence_label = batch[4]
        checkpoint_activations = False

        lengths = attention_mask.sum(axis=-1)
        filled_indices = attention_mask > 0
        dtype = self.bert.embeddings.word_embeddings.weight.dtype
        scalers = (attention_mask / lengths.pow(1. / 3)
                   .unsqueeze(-1)).to(dtype)[filled_indices].unsqueeze(-1)
        input_ids = batch[1][filled_indices]
        token_type_ids = batch[3][filled_indices]
        masked_lm_labels = masked_lm_labels[filled_indices]
        encoded_layers, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            scalers, lengths,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations
        )
        sequence_output = encoded_layers
        if not self.training:
            # In eval mode calculate all output representations for
            # compatibility with HuggingFace's Bert.
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output,
            )
            return prediction_scores, seq_relationship_score

        # filter out all masked labels.
        masked_token_indexes = torch.nonzero(
            (masked_lm_labels + 1),
        ).view(-1)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output,
            masked_token_indexes)

        target = torch.index_select(masked_lm_labels.view(-1), 0,
                                    masked_token_indexes)
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), target)
        next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2),
                                      next_sentence_label.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss


class BertForMaskedLM(BertPreTrainedModel):
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
                masked_lm_labels=None,
                checkpoint_activations=False):
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


class BertForNextSentencePrediction(BertPreTrainedModel):
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
                next_sentence_label=None,
                checkpoint_activations=False):
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


class BertForSequenceClassification(BertPreTrainedModel):
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

    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                checkpoint_activations=False):
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForMultipleChoice(BertPreTrainedModel):
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

    def __init__(self, config, num_choices):
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
                labels=None,
                checkpoint_activations=False):
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


class BertForTokenClassification(BertPreTrainedModel):
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
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
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

    def __init__(self, config, num_labels):
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
                labels=None,
                checkpoint_activations=False):
        sequence_output, _ = self.bert(input_ids,
                                       token_type_ids,
                                       attention_mask,
                                       output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))
            return loss
        else:
            return logits


class BertForQuestionAnswering(BertPreTrainedModel):
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
                end_positions=None,
                checkpoint_activations=False):
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

