# DeepSpeed note, code taken from commit 3d59216cec89a363649b4fe3d15295ba936ced0f
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/modeling.py

# coding=utf-8
# Copyright 2024 Andrew Argatkiny
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
"""Bert with DenseAttention and other architectural choices as described in the paper."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import sys
from io import open
from itertools import accumulate
from typing import List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils import checkpoint

from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init

from dense_attention import (DenseAttentionOneHead,
                             DenseAttentionOneHeadLinComplexity,
                             DenseAttentionMultiHead,
                             DenseAttentionMultiHeadLinComplexity)
from activations import MaxNormActivation, UncenteredLayerNorm

logger = logging.getLogger(__name__)


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


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=1024,
                 num_hidden_layers=32,
                 num_attention_heads=1,
                 intermediate_size=4096,
                 hidden_act="gelu",
                 max_position_embeddings=1024,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 ):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                pooler. If string, "gelu", "relu" and "swish" are supported.
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
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
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

        self.LayerNorm = nn.Hardtanh()

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


class ExpandedFFN(nn.Module):
    """ A two-layers Feed Forward Network with expanding and contracting layers
     similar to original Transformer block's FFN, but without biases and with
     the mechanics allowing for the weights to be scaled by their max norm."""

    def __init__(self, config):
        super(ExpandedFFN, self).__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.expanding_weight = nn.Parameter(self._init_weights())
        self.activation = nn.ReLU()
        self.contracting_weight = nn.Parameter(
            self._init_weights(expansion_flag=False)
        )

        self.default_norm_expand = 1 / math.sqrt(self.intermediate_size * 2)
        self.default_norm_contract = 1 / math.sqrt(self.intermediate_size * 2)
        self.norm_ratio_expand = self.default_norm_expand / self.expanding_weight.abs().max().item()
        self.norm_ratio_contract = self.default_norm_contract / self.contracting_weight.abs().max().item()

    def _init_weights(self, expansion_flag=True):
        if expansion_flag:
            weight_shape = (self.hidden_size, self.intermediate_size)
        else:
            weight_shape = (self.intermediate_size, self.hidden_size)

        weight = torch.randn(weight_shape) * (1 / math.sqrt(self.intermediate_size * 2))
        return weight

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


class BertLayerWithActivation(nn.Module):
    def __init__(self, config: BertConfig, layer_number):
        super(BertLayerWithActivation, self).__init__()
        self.activation = MaxNormActivation(config)
        if config.num_attention_heads == 1:
            # N^2*d < N*d^2?
            if (config.max_position_embeddings ** 2 * config.hidden_size
                    < config.max_position_embeddings * config.hidden_size ** 2):
                attention_class = DenseAttentionOneHead
            else:
                attention_class = DenseAttentionOneHeadLinComplexity
        else:
            # N^2*d < N*d^2/h?
            if (config.max_position_embeddings ** 2 * config.hidden_size
                    < config.max_position_embeddings * config.hidden_size ** 2
                    / config.num_attention_heads):
                attention_class = DenseAttentionMultiHead
            else:
                attention_class = DenseAttentionMultiHeadLinComplexity

        self.attention = attention_class(config, layer_number)
        self.ffn = ExpandedFFN(config)
        self.ffn_activation = MaxNormActivation(config)

    def forward(self, hidden_states, attention_mask):
        prev_hidden_states = hidden_states
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states * attention_mask
        hidden_states = self.attention(hidden_states)
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
            # For multi-head attention change to:
            # seqs.append(self.attention(seq.unsqueeze(0)).squeeze(0))
            seqs.append(self.attention(seq))
        hidden_states = torch.cat(seqs, dim=0)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_activation(hidden_states)
        hidden_states = hidden_states + prev_hidden_states
        return hidden_states


class BertEncoder(nn.Module):
    def __init__(self, config, args, sparse_attention_config=None):
        super(BertEncoder, self).__init__()

        self.FinalLayerNorm = BertLayerNorm(config.hidden_size)
        layers = [BertLayerWithActivation(config, layer_number=n + 1)
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
        # first_token_tensor = hidden_states[cs_lengths]
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
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
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
        raise NotImplementedError()


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

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

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config: BertConfig, args=None):
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
        # cs_lengths = list(accumulate([0] + lengths_list[:-1]))
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
        config: a BertConfig class instance with the configuration to build a new model.

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

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config: BertConfig, args):
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

    def forward(self, batch, log=True):
        input_ids = batch[1]
        token_type_ids = batch[3]
        attention_mask = batch[2]
        masked_lm_labels = batch[5]
        next_sentence_label = batch[4]
        checkpoint_activations = False
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
