# DeepSpeed note, code taken from commit 3d59216cec89a363649b4fe3d15295ba936ced0f
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/modeling.py

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

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from src.activations import Activation2Class
from src.model_config import ModelConfig
from src.positional_embeddings import (
    PositionalEmbeddingsTypes, SinusoidalPositionalEncoding, RelPEType,
    RelPETypeToClass
)

from torch.nn.parameter import Parameter

from src.danet_layers import DANetLayerWithLocalAttention, DANetLayer, TransformerLayer

from train_utils import get_num_params

logger = logging.getLogger(__name__)


class RealNumberEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        embedding_dim = config.hidden_size
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.Tensor(embedding_dim))
        #self.bias = Parameter(torch.zeros(embedding_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=self.config.initializer_range)
        #nn.init.normal_(self.bias, mean=0.0, std=0.1)

    def forward(self, x):
        x = x.to(self.weight.dtype)
        emb = (x.unsqueeze(-1) / 255. - 0.5) * 0.5 * self.weight #+ self.bias
        return emb


class DANetEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config: ModelConfig):
        super(DANetEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size)
        #self.word_embeddings = RealNumberEmbedding(config)
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

        if config.token_type_embeddings:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                      config.hidden_size)
            self.add_tok_typ_emb_fn = self.add_token_type_embeddings
        else:
            self.add_tok_typ_emb_fn = lambda x, y: x

        # By default nn.Hardtanh
        self.LayerNorm = Activation2Class[config.embedding_ln_type](
            hidden_size=config.hidden_size
        )
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

    def add_token_type_embeddings(self, embeddings, token_type_ids=None):
        if token_type_ids is None:
           return embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        return token_type_embeddings + embeddings

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        words_embeddings = self.word_embeddings(input_ids)

        embeddings = self.add_pos_emb_fn(words_embeddings)
        embeddings = self.add_tok_typ_emb_fn(embeddings, token_type_ids)
        # embeddings = embeddings / embeddings.abs().max(axis=-1, keepdim=True)[0]
        # embeddings = embeddings * attention_mask[0]
        embeddings = self.LayerNorm(embeddings)
        # embeddings = clip_grad_values(embeddings)
        # embeddings = self.LayerNorm(embeddings, attention_mask)
        embeddings = self.dropout(embeddings)
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

class DANetEncoder(nn.Module):
    def __init__(self, config: ModelConfig, args):
        super(DANetEncoder, self).__init__()
        if (config.pos_emb_type == PositionalEmbeddingsTypes.RELPE and
                config.relpe_type is not None):
            self.relpe_type = RelPEType[config.relpe_type.upper()]
        else:
            self. relpe_type = RelPEType.DUMMY
        # Added later to make it similar to GPT-2
        if config.final_ln_type is not None:
            self.FinalLayerNorm = Activation2Class[config.final_ln_type](config.hidden_size)
            self.final_transform_fn = lambda x: self.FinalLayerNorm(x)
        else:
            self.final_transform_fn = lambda x: x

        self.rope_cache = RelPETypeToClass[self.relpe_type](
            config.max_position_embeddings, #args.max_seq_length
            config.hidden_size // config.num_attention_heads,
            num_heads=config.num_attention_heads
        )
        self.local_scheme = config.local_scheme.split("_")
        layer_class = (DANetLayer if not config.local_attention
                       else DANetLayerWithLocalAttention)
        layers = [layer_class(config, layer_number=n)
                  for n in range(config.num_hidden_layers)]
        if config.hybrid:
            for i in range(config.num_hidden_layers):
                if 'softmax' in self.local_scheme[i % len(self.local_scheme)]:
                    layers[i] = TransformerLayer(config, i)
        if hasattr(args, "scale_ffn_weights") and args.scale_ffn_weights:
            for layer in layers:
                layer.ffn.forward = layer.ffn.forward_scaled
        self.layer = nn.ModuleList(layers)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask,
                output_all_encoded_layers=True,
                checkpoint_activations=False):
        all_encoder_layers = []

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, self.rope_cache)
            # if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)

        # if not output_all_encoded_layers or checkpoint_activations:
        hidden_states = self.final_transform_fn(hidden_states)
        all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    """Combine last layer's representations and optionally transform the result."""
    def __init__(self, config: ModelConfig):
        super(BertPooler, self).__init__()
        if config.pooler_function == "first":
            self.pooler_function = lambda x: x[:, 0]
        elif config.pooler_function == "max":
            self.pooler_function = lambda x: torch.max(x, dim=1)[0]
        else:
            self.pooler_function = lambda x: x.mean(dim=1)

        self.post_pool_transform = lambda x: x
        if not config.pooler_no_dense:
            # This name `dense_act` is kept for compatibility with old trained
            # models as well as the whole post_pool_transform which should
            # logically be placed in later classifier modules.
            self.dense_act = nn.Linear(config.hidden_size, config.hidden_size,
                                       bias=False)
            self.activation = Activation2Class[config.pooler_act]()
            if config.pooler_ln_type is not None:
                self.layer_norm = Activation2Class[config.pooler_ln_type](config.hidden_size)
                self.pooler_ln_fn = lambda x: self.layer_norm(x)
            else:
                self.pooler_ln_fn = lambda x: x
            self.post_pool_transform = self.pooler_dense

    def pooler_dense(self, pooled_output):
        pooled_output = self.activation(self.dense_act(pooled_output))
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
        self.dense_act = nn.Linear(config.hidden_size,
                                          config.hidden_size,
                                          bias=False)
        self.activation = Activation2Class[config.lm_head_act]()
        if config.lm_head_ln_type is not None:
            self.LayerNorm = Activation2Class[config.lm_head_ln_type](
                config.hidden_size, eps=1e-12
            )
            self.apply_ln_fn = lambda x: self.LayerNorm(x)
        else:
            self.apply_ln_fn = lambda x: x

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.apply_ln_fn(hidden_states)
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
        #self.activation = nn.Hardtanh(-20, 2)

    def forward(self, hidden_states, masked_token_indexes):
        hidden_states = self.transform(hidden_states)

        if masked_token_indexes is not None:
            hidden_states = torch.index_select(
                hidden_states.view(-1, hidden_states.shape[-1]), 0,
                masked_token_indexes)
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
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.seq_relationship = nn.Linear(config.hidden_size, num_labels,
                                          bias=config.classifier_bias)

    def forward(self,
                sequence_output,
                pooled_output,
                masked_token_indexes=None):
        prediction_scores = self.predictions(sequence_output,
                                             masked_token_indexes)
        seq_relationship_score = self.seq_relationship(self.dropout(pooled_output))
        return prediction_scores, seq_relationship_score


class DANetPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(DANetPreTrainedModel, self).__init__()
        if not isinstance(config, ModelConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `ModelConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def resize_learned_pos_embeddings(self, src: OrderedDict, dst: nn.Module) -> None:
        """
        Resizes learned positional embeddings during loading of the model in Deepspeed.
        Params:
            src: PyTorch state dict of a Deepspeed checkpoint (source model)
            dst: PyTorch module of the destination model
        """
        if self.config.pos_emb_type != PositionalEmbeddingsTypes.LEARNED:
            logger.warning(
                "Attempted to resize positional embeddings. In this instance "
                "they are not learnable, there is no need for resizing.")
            dst.load_state_dict(src, strict=False)
            return

        old_embeds = src["bert.embeddings.position_embeddings.weight"]
        num_old_embeds = old_embeds.shape[0]
        num_new_embeds = self.config.max_position_embeddings
        new_embeddings = (torch.zeros(num_new_embeds, self.config.hidden_size)
                          .to(old_embeds.device)
                          .to(old_embeds.dtype))
        num_repeats = (num_new_embeds + num_old_embeds - 1) // num_old_embeds
        new_embeddings.data[:num_new_embeds, :] = (
            old_embeds.data.detach().clone()
            .repeat(num_repeats, 1)[:num_new_embeds, :]
        )
        src["bert.embeddings.position_embeddings.weight"] = new_embeddings
        logger.info(f"Successfully restored learned positional embeddings to "
                    f"size {self.config.max_position_embeddings}")
        del old_embeds
        dst.load_state_dict(src, strict=False)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        std = self.config.initializer_range  # / math.sqrt(3)
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0, std=std)
        # if isinstance(module, nn.Linear):
        elif isinstance(module, nn.Linear):
            module.weight.data.uniform_(-std, std)
            if module.bias is not None:
                module.bias.data.zero_()


class DANetModel(DANetPreTrainedModel):
    """DANet model ("Dense Attention Network").

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

    model = modeling.DANetModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config: ModelConfig, args=None):
        super(DANetModel, self).__init__(config)
        self.PATH_TO_LAYERS = "encoder.layer"
        self.PATH_TO_EMBEDDINGS = "embeddings"
        self.embeddings = DANetEmbeddings(config)
        #self.posit_embs = MultiplicativePositionalEmbedding(config)
        # set pad_token_id that is used for sparse attention padding
        self.pad_token_id = config.pad_token_id if hasattr(
            config, 'pad_token_id') and config.pad_token_id is not None else 0
        # set sparse_attention_config if it has been selected
        self.sparse_attention_config = None  # get_sparse_attention_config(
        #     args, config.num_attention_heads)
        # self.sparse_attention_utils = get_sparse_attention_utils(self.sparse_attention_config)
        self.encoder = DANetEncoder(config, args)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
        logger.info("Init BERT pretrain model")
        logger.info(f"Total parameters in transformer blocks: {get_num_params(self, non_embedding=False)}")

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                output_all_encoded_layers=False,
                checkpoint_activations=False):
        #if attention_mask is None:
        #    attention_mask = torch.ones_like(input_ids)
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
        embedding_output = self.embeddings(input_ids,
                                           attention_mask,
                                           token_type_ids)
        #attention_mask = self.posit_embs(attention_mask)
        encoded_layers = self.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            checkpoint_activations=checkpoint_activations)
        encoded_layers = [embedding_output] + encoded_layers
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

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


class DANetForPreTraining(DANetPreTrainedModel):
    """DANet model with pre-training heads.
    This module comprises the DANet model followed by the two pre-training heads:
        - the language modeling head, and
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
        `masked_lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `label` is `None`:
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
        super(DANetForPreTraining, self).__init__(config)
        self.PATH_TO_BACKBONE = "bert"
        self.bert = DANetModel(config, args)
        self.num_labels = args.num_labels if hasattr(args, "num_labels") else 2
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=self.num_labels)
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)
        self.window_size = config.window_size

        self.head = self.mlm_cls_head
        if (hasattr(args,'only_mlm_task') and hasattr(args,'only_cls_task')
                and args.only_mlm_task and args.only_cls_task):
            raise ValueError("Only one of the options 'only_mlm_task' "
                             "and 'only_cls_task' should hold True.")
        if hasattr(args,'only_mlm_task') and args.only_mlm_task:
            self.head = self.mlm_head
        elif hasattr(args,'only_cls_task') and args.only_cls_task:
            self.head = self.cls_head

        if hasattr(args,'unpad_inputs') and args.unpad_inputs:
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
        self.use_local_attention = config.local_attention

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
        checkpoint_activations = False
        # pad_attention = attention_mask.numel() / attention_mask.sum()
        dtype = self.bert.embeddings.word_embeddings.weight.dtype
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_attention_mask = (
            attention_mask /
            attention_mask.sum(axis=-1, keepdim=True).pow(1. / 3)
        ).to(dtype).unsqueeze(-1)
        if self.use_local_attention:
            local_attention_mask = (
                    attention_mask / self.window_size ** (1. / 3)
            ).to(dtype).unsqueeze(-1)
            extended_attention_mask = (
                local_attention_mask,
                extended_attention_mask
            )
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

    def forward_unpadded(self, batch, log=True):
        attention_mask = batch[2]
        masked_lm_labels = batch[5]
        label = batch[4]
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
                                      label.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss


class BertForMaskedLM(DANetPreTrainedModel):
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
        self.PATH_TO_BACKBONE = "bert"
        self.bert = DANetModel(config)
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


class BertForNextSentencePrediction(DANetPreTrainedModel):
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
        `label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `label` is `None`:
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
        self.PATH_TO_BACKBONE = "bert"
        self.bert = DANetModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                label=None,
                checkpoint_activations=False):
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls(pooled_output)

        if label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2),
                                          label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSequenceClassification(DANetPreTrainedModel):
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
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = args.num_labels if hasattr(args, "num_labels") else 2
        self.window_size = config.window_size
        self.PATH_TO_BACKBONE = "bert"
        self.bert = DANetModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels,
                                    bias=config.classifier_bias)
        """
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=args.num_labels)
        self.classifier = self.cls.seq_relationship
        """

        self.apply(self.init_bert_weights)
        if hasattr(args, "zero_init_pooler") and args.zero_init_pooler:
            self.bert.pooler.dense_act.weight.data.zero_()
        self.use_local_attention = config.local_attention

    def forward(self,
                input_ids,
                label=None,
                attention_mask=None,
                token_type_ids=None,
                checkpoint_activations=False):
        checkpoint_activations = False
        dtype = self.bert.embeddings.word_embeddings.weight.dtype
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_attention_mask = (
            attention_mask /
            attention_mask.sum(axis=-1, keepdim=True).pow(1. / 3)
        ).to(dtype).unsqueeze(-1)
        if self.use_local_attention:
            local_attention_mask = (
                    attention_mask / self.window_size ** (1. / 3)
            ).to(dtype).unsqueeze(-1)
            extended_attention_mask = (
                local_attention_mask,
                extended_attention_mask
            )

        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask=extended_attention_mask,
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

class BertForRegression(DANetPreTrainedModel):
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
        super(BertForRegression, self).__init__(config)
        self.num_labels = args.num_labels if hasattr(args, "num_labels") else 2
        self.window_size = config.window_size
        self.PATH_TO_BACKBONE = "bert"
        self.bert = DANetModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1,
                                    bias=config.classifier_bias)
        """
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=args.num_labels)
        self.classifier = self.cls.seq_relationship
        """

        self.apply(self.init_bert_weights)
        if hasattr(args, "zero_init_pooler") and args.zero_init_pooler:
            self.bert.pooler.dense_act.weight.data.zero_()
        self.use_local_attention = config.local_attention

    def forward(self,
                input_ids,
                label=None,
                attention_mask=None,
                token_type_ids=None,
                checkpoint_activations=False):
        checkpoint_activations = False
        dtype = self.bert.embeddings.word_embeddings.weight.dtype
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_attention_mask = (
            attention_mask /
            attention_mask.sum(axis=-1, keepdim=True).pow(1. / 3)
        ).to(dtype).unsqueeze(-1)
        if self.use_local_attention:
            local_attention_mask = (
                    attention_mask / self.window_size ** (1. / 3)
            ).to(dtype).unsqueeze(-1)
            extended_attention_mask = (
                local_attention_mask,
                extended_attention_mask
            )

        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask=extended_attention_mask,
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

class BertForAANMatching(DANetPreTrainedModel):
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
        super(BertForAANMatching, self).__init__(config)
        self.num_labels = args.num_labels if hasattr(args, "num_labels") else 2
        self.window_size = config.window_size
        self.PATH_TO_BACKBONE = "bert"
        self.bert = DANetModel(config, args)
        self.dense = nn.Linear(config.hidden_size * 4, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU(approximate='tanh')
        self.classifier = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.apply(self.init_bert_weights)
        self.use_local_attention = config.local_attention

    def process_mask(self, input_ids, attention_mask):
        dtype = self.bert.embeddings.word_embeddings.weight.dtype
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_attention_mask = (
            attention_mask /
            attention_mask.sum(axis=-1, keepdim=True).pow(1. / 3)
        ).to(dtype).unsqueeze(-1)
        if self.use_local_attention:
            local_attention_mask = (
                    attention_mask / self.window_size ** (1. / 3)
            ).to(dtype).unsqueeze(-1)
            extended_attention_mask = (
                local_attention_mask,
                extended_attention_mask
            )
        return extended_attention_mask

    def forward(self,
                input_ids,
                input_ids2,
                attention_mask=None,
                attention_mask2=None,
                label=None,
                token_type_ids=None,
                checkpoint_activations=False):
        checkpoint_activations = False

        extended_attention_mask1 = self.process_mask(input_ids, attention_mask)
        _, pooled_output1 = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask=extended_attention_mask1,
                                     output_all_encoded_layers=False)
        extended_attention_mask2 = self.process_mask(input_ids2, attention_mask2)
        _, pooled_output2 = self.bert(input_ids2,
                                     token_type_ids,
                                     attention_mask=extended_attention_mask2,
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

class BertForMultipleChoice(DANetPreTrainedModel):
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
        self.PATH_TO_BACKBONE = "bert"
        self.bert = DANetModel(config)
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


class BertForTokenClassification(DANetPreTrainedModel):
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
        self.PATH_TO_BACKBONE = "bert"
        self.bert = DANetModel(config)
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


class BertForQuestionAnswering(DANetPreTrainedModel):
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
        self.PATH_TO_BACKBONE = "bert"
        self.bert = DANetModel(config)
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
