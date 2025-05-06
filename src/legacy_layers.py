"""
Auxiliary layers used for pretraining of initial DenseAttention BERT model and
LRA models. This code is deprecated and generally shouldn't be used.
"""
from __future__ import division, absolute_import, print_function, unicode_literals

import math
import sys

import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.nn.init as init

from src.activations import Activation2Class, StandardLayerNorm
from src.model_config import ModelConfig
from src.positional_embeddings import RelPEBase

# DeepSpeed note, code taken from commit 3d59216cec89a363649b4fe3d15295ba936ced0f

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


def init_bert_weights_legacy(model, module):
    """Initialization routine which was used for first decoder LM pretraining."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        std = model.config.initializer_range  # / math.sqrt(3)
        module.weight.data.normal_(mean=0, std=std)
        # module.weight.data.uniform_(-std, std)
    elif isinstance(module, StandardLayerNorm) and module.bias is not None:
        module.bias.data.zero_()
        # module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


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



class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        """
        self.dense_act = LinearActivation(config.hidden_size,
                                          config.hidden_size,
                                          act="tanh")
        """
        if config.pooler_function == "first":
            self.pooler_function = lambda x: x[:, 0]
        elif config.pooler_function == "max":
            self.pooler_function = lambda x: torch.max(x, dim=1)[0]
        else:
            self.pooler_function = lambda x: x.mean(dim=1)

        self.post_pool_transform = lambda x: x
        if not config.pooler_no_dense:
            self.dense_act = LinearActivation(config.hidden_size,
                                          config.hidden_size,
                                          act="tanh")
            if config.pooler_ln_type is not None:
                self.layer_norm = Activation2Class[config.pooler_ln_type](config.hidden_size)
                self.pooler_ln_fn = lambda x: self.layer_norm(x)
            else:
                self.pooler_ln_fn = lambda x: x
            self.post_pool_transform = self.pooler_dense

    def pooler_dense(self, pooled_output):
        pooled_output = self.dense_act(pooled_output)
        pooled_output = self.pooler_ln_fn(pooled_output)
        return pooled_output

    def forward(self, hidden_states: torch.Tensor):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.pooler_function(hidden_states)
        pooled_output = self.post_pool_transform(pooled_output)
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
                                          act=config.lm_head_act)
        if config.lm_head_ln_type is not None:
            self.LayerNorm = Activation2Class[config.lm_head_ln_type](
                config.hidden_size, eps=1e-12
            )
            self.apply_ln_fn = lambda x: self.LayerNorm(x)
        else:
            self.apply_ln_fn = lambda x: x

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        hidden_states = self.apply_ln_fn(hidden_states)
        return hidden_states


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
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        torch.nn.init.normal_(self.queries, mean=0,
                              std=std / math.sqrt(2.0 * num_layers * self.hidden_size))

    def forward(self, hidden_states: torch.Tensor, rope_cache: RelPEBase = None):
        hidden_states = self.dropout(hidden_states)
        hidden_states = rope_cache.apply_relpe(hidden_states)
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

    def forward(self, hidden_states: torch.Tensor, rope_cache: RelPEBase = None):
        hidden_states = self.dropout(hidden_states)
        hidden_states = rope_cache.apply_relpe(hidden_states)
        # hidden_states: Batch, SeqLen, EmbedDim
        # queries: Batch, SeqLen, EmbedDim
        size = hidden_states.size()
        new_size = size[:-1] + (self.n_heads, self.head_size)
        queries = F.linear(hidden_states, self.queries)  # * self.norm_ratio_queries)
        queries = queries.view(new_size)
        # queries = rope.apply_rope(queries, rope_cache)
        # queries: Batch, SeqLen, Head, HeadDim
        queries = queries.permute(0, 2, 1, 3)
        # queries: Batch, Head, SeqLen, HeadDim

        hidden_states = hidden_states.view(new_size)
        #hidden_states = rope.apply_rope(hidden_states, rope_cache)
        # hidden_states: Batch, SeqLen, Head, HeadDim
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        # hidden_states: Batch, Head, SeqLen, HeadDim
        keys = hidden_states.transpose(2, 3)
        # keys: Batch, Head, HeadDim, SeqLen
        #keys = rope.apply_rope(keys, rope_cache.permute(0, 2, 3, 1))

        pre_attn = torch.matmul(keys, hidden_states)
        #pre_attn = self.dropout(pre_attn)
        # pre_attn: Batch, Head, HeadDim, HeadDim
        attention = torch.matmul(queries, pre_attn)
        # attention: Batch, Head, SeqLen, HeadDim

        output = attention.permute(0, 2, 1, 3)
        # output: Batch, SeqLen, Head, HeadDim
        output = output.reshape(*size)
        # output: Batch, SeqLen, EmbedDim
        return output


class DenseAttentionLinMHLocal(DenseAttentionMultiHead):
    def __init__(self, config: ModelConfig, layer_number=1):
        super(DenseAttentionLinMHLocal,
              self).__init__(config, layer_number)
        self.window_size = config.window_size
        assert config.max_position_embeddings % self.window_size == 0
        self.num_windows = config.max_position_embeddings // self.window_size

    def forward(self, hidden_states: torch.Tensor, rope_cache: RelPEBase = None):
        hidden_states = self.dropout(hidden_states)
        hidden_states = rope_cache.apply_local_relpe(
            hidden_states, self.window_size, self.num_windows
        )
        # hidden_states: Batch, SeqLen, EmbedDim
        size = hidden_states.size()
        new_size = (size[0], self.num_windows, self.window_size,
                    self.n_heads, self.head_size)
        queries = F.linear(hidden_states, self.queries)  # * self.norm_ratio_queries)
        # queries: Batch, SeqLen, EmbedDim
        queries = queries.view(new_size)
        # queries = rope.apply_rope(queries, rope_cache)
        # queries: Batch, Seq, SubSeqLen, Head, HeadDim
        queries = queries.permute(0, 1, 3, 2, 4)
        # queries: Batch, Seq, Head, SubSeqLen, HeadDim

        hidden_states = hidden_states.view(new_size)
        #hidden_states = rope.apply_rope(hidden_states, rope_cache)
        # hidden_states: Batch, Seq, SubSeqLen, Head, HeadDim
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4)
        # hidden_states: Batch, Seq, Head, SubSeqLen, HeadDim
        keys = hidden_states.transpose(-2, -1)
        # keys: Batch, Seq, Head, HeadDim, SubSeqLen
        #keys = rope.apply_rope(keys, rope_cache.permute(0, 2, 3, 1))

        pre_attn = torch.matmul(keys, hidden_states)
        #pre_attn = self.dropout(pre_attn)
        # pre_attn: Batch, Seq, Head, HeadDim, HeadDim
        attention = torch.matmul(queries, pre_attn)
        # attention: Batch, Seq, Head, SubSeqLen, HeadDim

        output = attention.permute(0, 1, 3, 2, 4)
        # output: Batch, Seq, SeqLen, Head, HeadDim
        output = output.reshape(*size)
        # output: Batch, SeqLen, EmbedDim
        return output


class DenseAttentionLinMHLocalShifted(DenseAttentionLinMHLocal):
    def __init__(self, config: ModelConfig, layer_number=1):
        super(DenseAttentionLinMHLocalShifted,
              self).__init__(config, layer_number)
        # self.window_size = config.window_size
        assert self.window_size % 2 == 0
        if config.max_position_embeddings < self.window_size:
            self.left_pad = 0
            self.right_pad = 0
            self.num_windows = 1
        else:
            self.left_pad = self.window_size // 2
            self.right_pad = self.window_size // 2
            self.num_windows = self.num_windows + 1

    def forward(self, hidden_states: torch.Tensor, rope_cache=None):
        hidden_states = F.pad(hidden_states,
                              pad=(0, 0, self.left_pad, self.right_pad))
        hidden_states = super(DenseAttentionLinMHLocalShifted,
                              self).forward(hidden_states, rope_cache)
        return hidden_states[:, self.left_pad:-self.right_pad, :]


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
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
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

    def forward(self, hidden_states: torch.Tensor, rope_cache: RelPEBase = None):
        hidden_states = self.dropout(hidden_states)
        hidden_states = rope_cache.apply_relpe(hidden_states)
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

    def forward(self, hidden_states: torch.Tensor, rope_cache: RelPEBase = None):
        hidden_states = self.dropout(hidden_states)
        hidden_states = rope_cache.apply_relpe(hidden_states)
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
