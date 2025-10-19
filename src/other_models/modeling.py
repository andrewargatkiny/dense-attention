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
import inspect

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from src.positional_embeddings import PositionalEmbeddingsTypes, SinusoidalPositionalEncoding, RelPETypeToClass, \
    RelPEType
from .attention_kernels import SoftmaxAttention, LinearAttention, SlidingWindowAttention, PowerAttention
from ..activations import Activation2Class
from .transformers import ACT2FN, BertLayer, BertSelfAttention, BertSelfLocalAttention, BertSelfShiftedLocalAttention, TransformerConfig, BertLayerNorm
from .layers_registry import LayerTypeToClass, LayerConfigToClass

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




class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()

        #Added later to make it similar to GPT-2
        self.FinalLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        


        if config.layers is not None:
            rope_caches = []
            modules = []

            if config.layers_scheme:
                name_to_config = {}
                for layer in config.layers:
                    if not layer.get("layer_name"):
                        raise ValueError("Each layer in config.layers must have a 'layer_name' when 'layers_scheme' is used.")
                    if "layer_type" not in layer:
                        raise ValueError(f"Layer '{layer['layer_name']}' must contain 'layer_type'")
                    
                    layer_name = layer["layer_name"]
                    if layer_name in name_to_config:
                        raise ValueError(f"Duplicate layer_name detected in config.layers: '{layer_name}'")
                    name_to_config[layer_name] = layer

                ordered_names = config.layers_scheme.split("_")
                
                if config.num_hidden_layers % len(ordered_names) != 0:
                    raise ValueError("num_hidden_layers must be divisible by the number of layers in layers_scheme.")
                
                self.repeats = config.num_hidden_layers // len(ordered_names)
                
                self.layer_scheme = []
                for layer_name in ordered_names:
                    if layer_name not in name_to_config:
                        raise ValueError(f"Layer name '{layer_name}' from layers_scheme not found in config.layers.")

                    layer_config_dict = name_to_config[layer_name]
                    layer_type = layer_config_dict["layer_type"]
                    
                    config_class = LayerConfigToClass[layer_type]
                    final_layer_params = layer_config_dict.copy()
                    init_signature = inspect.signature(config_class.__init__)
                    init_params = init_signature.parameters

                    for param_name in init_params:
                        if param_name not in final_layer_params and hasattr(config, param_name):
                            final_layer_params[param_name] = getattr(config, param_name)
                    
                    layer_config = config_class(**final_layer_params)
                    self.layer_scheme.append(final_layer_params)
                    if (final_layer_params.get("pos_emb_type") == PositionalEmbeddingsTypes.RELPE and
                            final_layer_params.get("relpe_type") is not None):
                        relpe_type = RelPEType[final_layer_params["relpe_type"].upper()]
                    else:
                        relpe_type = RelPEType.DUMMY

                    relpe_class = RelPETypeToClass[relpe_type]
                    if layer_type == "danet":
                        rope_cache = relpe_class(
                            final_layer_params["max_position_embeddings"],
                            final_layer_params["hidden_size"] // final_layer_params["num_attention_heads"],
                            num_heads=final_layer_params["num_attention_heads"]
                        )
                    else:
                        rope_cache = relpe_class(
                            final_layer_params["max_position_embeddings"],
                            final_layer_params["hidden_size"] // final_layer_params["num_attention_heads"],
                            num_heads=None
                        )
                    rope_caches.append(rope_cache)

                    layer_class = LayerTypeToClass[layer_type]
                    module = layer_class(layer_config)
                    modules.append(module)
                modules = modules * self.repeats
                self.rope_caches = nn.ModuleList(rope_caches * self.repeats)
            elif not config.layers_scheme and len(config.layers) == 1:
                single_layer_config = config.layers[0]
                if "layer_type" not in single_layer_config:
                    raise ValueError("The single layer defined in config.layers must have a 'layer_type'.")
                final_layer_params = single_layer_config.copy()
                init_signature = inspect.signature(config_class.__init__)
                init_params = init_signature.parameters
                for param_name in init_params:
                    if param_name not in final_layer_params and hasattr(config, param_name):
                        final_layer_params[param_name] = getattr(config, param_name)
                layer_config = config_class(**final_layer_params)
                self.repeats = config.num_hidden_layers
                
                if (config.get("pos_emb_type") == PositionalEmbeddingsTypes.RELPE and
                        final_layer_params.get("relpe_type") is not None):
                    relpe_type = RelPEType[final_layer_params["relpe_type"].upper()]
                else:
                    relpe_type = RelPEType.DUMMY
                relpe_class = RelPETypeToClass[relpe_type]
                if layer_type == "danet":
                    rope_cache = relpe_class(
                        final_layer_params["max_position_embeddings"],
                        final_layer_params["hidden_size"] // final_layer_params["num_attention_heads"],
                        num_heads=final_layer_params["num_attention_heads"]
                    )
                else:
                    rope_cache = relpe_class(
                        final_layer_params["max_position_embeddings"],
                        final_layer_params["hidden_size"] // final_layer_params["num_attention_heads"],
                        num_heads=None
                    )
                rope_caches.append(rope_cache)

                layer_class = LayerTypeToClass[layer_type]
                module = layer_class(layer_config)
                modules.append(module)
                modules = modules * self.repeats
                self.rope_caches = nn.ModuleList(rope_caches * self.repeats)
                self.layer_scheme = [single_layer_config]
            else:
                raise ValueError("Invalid layer configuration. Provide either a 'layers_scheme' with corresponding layer definitions, or a single layer definition in 'config.layers' without a 'layers_scheme'.")

            self.layer = nn.ModuleList(modules)

        else:
            self. relpe_type = RelPEType.DUMMY
            self.rope_cache = RelPETypeToClass[self.relpe_type](
                seq_len=config.max_position_embeddings,
                n_elem=config.hidden_size // config.num_attention_heads,
                sep_head_dim=True
            )

            # logic for local attention scheme
            if hasattr(config, 'local_scheme') and config.local_scheme:
                scheme = config.local_scheme.split('_')
                valid_codes = {'g', 'l', 'sl', 'swa'}
                for code in scheme:
                    if code not in valid_codes:
                        raise ValueError(f"Unknown attention type code '{code}' in local_scheme. "
                                            f"Valid codes are: {sorted(list(valid_codes))}")

                for i, layer_module in enumerate(self.layer):
                    code = scheme[i % len(scheme)]
                    if code == 'l':
                        layer_module.attention.self = BertSelfLocalAttention(config)
                    elif code == 'sl':
                        layer_module.attention.self = BertSelfShiftedLocalAttention(config)
                    elif code == 'swa':
                        layer_config = copy.deepcopy(config)
                        layer_config.attention_kernel = "swa"
                        layer_module.attention.self = BertSelfAttention(layer_config)
                    # 'g' is the default and requires no change, so we just pass.
            # fallback to old logic for backward compatibility
            elif config.local_attention:
                for i, layer in enumerate(self.layer):
                    if i % 3 == 0:
                        layer.attention.self = BertSelfLocalAttention(config)
                    elif i % 3 == 1:
                        layer.attention.self = BertSelfShiftedLocalAttention(config)
    def prepare_mask(self, hidden_states, attention_mask, layer_config):
        
        if layer_config["layer_type"] == "danet":
            dtype = hidden_states.dtype
            if attention_mask is None:
                attention_mask = torch.ones_like(hidden_states)
            extended_attention_mask = (
                attention_mask /
                attention_mask.sum(axis=-1, keepdim=True).pow(1. / 3)
            ).to(dtype)
            if layer_config["local_scheme"] in ["l", "sl"]:
                local_attention_mask = (
                        attention_mask / layer_config["window_size"] ** (1. / 3)
                ).to(dtype)
                extended_attention_mask = (
                    local_attention_mask,
                    extended_attention_mask
                )
            return extended_attention_mask
        else:
            if attention_mask is None:
                attention_mask = torch.ones_like(hidden_states)
            attention_mask = attention_mask.to(
                dtype=hidden_states.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
            return attention_mask
    def forward(self,
                hidden_states,
                attention_mask,
                output_all_encoded_layers=True,
                **kwargs):
        masks = []
        for i in range(len(self.layer_scheme)):
            masks.append(self.prepare_mask(hidden_states, attention_mask, self.layer_scheme[i]))
        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states, 
                attention_mask=masks[i % len(self.layer_scheme)],
                rope_cache= self.rope_caches[i % len(self.rope_caches)] if hasattr(self, "rope_caches") else self.rope_cache,
                  **kwargs) 
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

    def init_weights(self, module):
        """ Initialize the weights.
        """
        logger.info("Init BERT weights")
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            num_layers = self.config.num_hidden_layers
            std = self.config.initializer_range
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
        self.PATH_TO_LAYERS = "encoder.layer"
        self.PATH_TO_EMBEDDINGS = "embeddings"
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        # Do not initialize encoder layers here; only non-layer parts
        self.embeddings.apply(self.init_weights)
        self.pooler.apply(self.init_weights)
        logger.info("Init BERT pretrain model")

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                output_all_encoded_layers=True,
                **kwargs):



        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            **kwargs
        )
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
        self.PATH_TO_BACKBONE = "bert"
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
        # Initialize only heads; encoder layers self-initialize.
        self.cls.apply(self.init_weights)
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
                masked_lm_labels=None, label=None, log=True, **kwargs):

        sequence_output, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            **kwargs
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
        self.PATH_TO_BACKBONE = "bert"
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config,
                                   self.bert.embeddings.word_embeddings.weight)
        # Initialize only head; encoder layers self-initialize.
        self.cls.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                masked_lm_labels=None, **kwargs):
        sequence_output, _ = self.bert(input_ids,
                                       token_type_ids,
                                       attention_mask,
                                       output_all_encoded_layers=False, **kwargs)
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
        self.PATH_TO_BACKBONE = "bert"
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        # Initialize only head; encoder layers self-initialize.
        self.cls.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                next_sentence_label=None,
                **kwargs):
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     output_all_encoded_layers=False,
                                     **kwargs)
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
        self.PATH_TO_BACKBONE = "bert"
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        # Initialize only head; encoder layers self-initialize.
        self.classifier.apply(self.init_weights)

    def forward(self,
                input_ids,
                label=None,
                attention_mask=None,
                token_type_ids=None,
                **kwargs):
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     output_all_encoded_layers=False,
                                     **kwargs)
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
        self.PATH_TO_BACKBONE = "bert"
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1)
        """
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=args.num_labels)
        self.classifier = self.cls.seq_relationship
        """

        # Initialize only head; encoder layers self-initialize.
        self.regressor.apply(self.init_weights)
        self.use_local_attention = config.local_attention

    def forward(self,
                input_ids,
                label=None,
                attention_mask=None,
                token_type_ids=None,
                checkpoint_activations=False, 
                **kwargs):
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask=attention_mask,
                                     output_all_encoded_layers=False, 
                                     **kwargs)
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
        self.PATH_TO_BACKBONE = "bert"
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU(approximate='tanh')
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        # Initialize only head parts; encoder layers self-initialize.
        self.dense.apply(self.init_weights)
        self.classifier.apply(self.init_weights)

    def forward(self,
                input_ids,
                input_ids2,
                attention_mask=None,
                attention_mask2=None,
                label=None,
                token_type_ids=None,
                checkpoint_activations=False, 
                **kwargs):
        checkpoint_activations = False

        _, pooled_output1 = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask=attention_mask,
                                     output_all_encoded_layers=False, 
                                     **kwargs)
        _, pooled_output2 = self.bert(input_ids2,
                                     token_type_ids,
                                     attention_mask=attention_mask2,
                                     output_all_encoded_layers=False, 
                                     **kwargs)
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
        self.PATH_TO_BACKBONE = "bert"
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize only head; encoder layers self-initialize.
        self.classifier.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None, 
                **kwargs):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.bert(flat_input_ids,
                                     flat_token_type_ids,
                                     flat_attention_mask,
                                     output_all_encoded_layers=False, 
                                     **kwargs)
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
        self.PATH_TO_BACKBONE = "bert"
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # Initialize only head; encoder layers self-initialize.
        self.classifier.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None, 
                **kwargs):
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
        self.PATH_TO_BACKBONE = "bert"
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        # Initialize only head; encoder layers self-initialize.
        self.qa_outputs.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                start_positions=None,
                end_positions=None, 
                **kwargs):
        sequence_output, _ = self.bert(input_ids,
                                       token_type_ids,
                                       attention_mask,
                                       output_all_encoded_layers=False, 
                                       **kwargs)
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