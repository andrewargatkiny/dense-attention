from __future__ import division, absolute_import, print_function, unicode_literals

from transformers import AutoConfig
import copy
import json
import sys
from io import open

from src.positional_embeddings import PositionalEmbeddingsTypes


class ModelConfig(object):
    """Configuration class to store the configuration of a `DANetModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file=30528,
                 hidden_size=1024,
                 num_hidden_layers=32,
                 num_attention_heads=1,
                 intermediate_size=4,
                 default_ffn_norm=None,
                 hidden_dropout_prob=0,
                 hidden_act="relu",
                 swiglu_ffn=False,
                 attention_complexity="auto",
                 attention_probs_dropout_prob=0,
                 pre_attn_ln_type="max_norm",
                 post_attn_ln_type="max_norm",
                 max_position_embeddings=512,
                 token_type_embeddings=False,
                 embedding_ln_type="hardtanh",
                 type_vocab_size=2,
                 initializer_range=0.02,
                 pos_emb_type="learned",
                 embedding_dropout=0,
                 relpe_type=None,
                 relpe_scheme='qkv',
                 final_ln_type=None,
                 pooler_function="mean",
                 pooler_no_dense=False,
                 pooler_act="gelu",
                 pooler_ln_type=None,
                 classifier_bias=False,
                 lm_head_act="gelu",
                 lm_head_ln_type="uncentered_ln",
                 causal=False,
                 chunk_size=1024,
                 local_attention=False,
                 window_size=1024,
                 local_relpe=True,
                 local_scheme="l_sl_g",
                 hybrid=False,
                 transformer_heads=None,
                 attn_proj_biases=False,
                 hf_model_name_or_path=None,
                 **kwargs
                 ):
        """Constructs ModelConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the encoder.
            intermediate_size: The size ratio of the expansion in the
                `ExpandedFFN` layer as compared to input size. Default: 4.
            default_ffn_norm: The value used as a reference L infinity norm for
                weights in `ExpandedFFN` layer if scaling is used. If None or
                0, defaults to:
                `1 / math.sqrt(self.hidden_size * self.expansion_factor * 2)`.
            hidden_act: The non-linear activation function (function or string)
                in the encoder. Currently, isn't used anywhere but kept for
                compatibility with BERT configs.
            swiglu_ffn: Boolean flag indicating whether to use SwiGLU FFN
                instead of ordinary one. Default `False`.
            hidden_dropout_prob: The dropout probability after FFN's expansion
                and contraction transforms and after the pooler.
            attention_complexity: Selection of runtime complexity of
                computational kernel of DenseAttention. Should be 'linear',
                'quadratic' or 'auto' (default).
            attention_probs_dropout_prob: The dropout probability at the start
                of attention sub-layer.
            pre_attn_ln_type: Type of layer norm to use before the main
                DenseAttention matrix multiplication kernel. Default: `max_norm`.
            post_attn_ln_type: Type of layer norm to use after the main
                DenseAttention matrix multiplication kernel. Default: `max_norm`.
            max_position_embeddings: The maximum sequence length for the model.
                Impacts shape of learned positional embeddings tensor or, in
                case of RelPE, shape of precomputed tensor holding trigonometric
                multipliers.
            token_type_embeddings: Boolean flag indicating whether to use
                token type embeddings in the model. Default: `False`.
            type_vocab_size: The vocabulary size of the `token_type_ids` passed
                into model.
            embedding_ln_type: Activation or layer norm function at the end of
                embeddings layer.
            pos_emb_type: The type of positional embeddings to use. All
                possible types are defined in `PositionalEmbeddingsTypes` enum.
            initializer_range: A value used in the initializer functions as a
                parameter to init distributions.
            embedding_dropout: Dropout ratio at the end of embeddings layer.
            relpe_type: In case the chosen type of positional embeddings is
                RelPE, then determines which type of RelPEs to use. All
                possible types are defined in the `RelPEType` enum.
            relpe_scheme: If relpe are enabled, sets how they are applied.
                Possible values: 'qkv' (applied to hidden_states at the very
                start of DenseAttention calculation, BEFORE all the logic);
                'q', 'k', 'v', 'q,k', 'q,v', 'k,v', 'q,k,v'. In all cases
                except the first, RelPE apply individually to tensor(s)
                mentioned in the spec string, where q is queries, k is keys and
                 v is values.
            final_ln_type: the type of activation function or layer norm right
                after the last layer in the encoder.
            pooler_function: Function, used to pool all tokens embeddings
                after the last layer into one representation for classification
                tasks. Can take one of these values: 'first', 'mean', 'max'.
            pooler_no_dense: Boolean flag indicating whether NOT to insert a
                Linear layer (and possibly an activation and a layer norm, too)
                right after pooler transform.
            pooler_act: If `pooler_no_dense` is `False`, chooses the activation
                to insert right after Linear layer in post pool transform.
            pooler_ln_type: If `pooler_no_dense` is `False`, chooses the layer
                norm type to insert right after `pooler_act`.
            classifier_bias: Boolean flag indicating whether to use bias in the
                last output logits layer in sequence classification head.
            lm_head_act: Type of activation function for Language Modeling head.
            lm_head_ln_type: Type of layer norm or second activation to use
                after activation in Language Modeling head.
            causal: Boolean flag indicating whether the model and its layers
                have decoder architecture (causal language modeling). Default:
                `False`.
            chunk_size: Length of a sequence's chunk for causal LM parallel
                chunk-wise computation algorithm.
            local_attention: Boolean flag indicating whether to use local
                attention layers scheme. Default: `False`.
            window_size: length of local attention span in local attention
                layers. Default: 1024.
            local_relpe: Applicable only for `local` and `shifted_local` types
                of layer. For them, it indicates whether to apply RelPE using
                local or global indices along sequence dimension. Default:
                `True`.
            local_scheme: Scheme to form patterns of local and global attention
                layers. Should contain lowercase-letter layer codes separated
                by underscore '_'. Available codes: 'l' (local attention), 'sl'
                (shifted local), 'sw' (sliding window), and 'g' (global).
                Default: 'l_sl_g'.
            hybrid: Boolean flag indicating whether to use standard Transformer
                layers in the model. If `True`, layers corresponding to
                `softmax` in `local_scheme` are Transformer with Sliding Window
                Attention. Default: `False`.
            transformer_heads: Number of heads in Transformer layers for hybrid
                models. Default: None.
            attn_proj_biases: Whether to use bias in Q, K, V, O matrices in
                Transformer layers' attention. Default: `False`.

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
            self.intermediate_size = intermediate_size
            self.hidden_act = hidden_act # left only for compat. with old configs
            self.swiglu_ffn = swiglu_ffn
            self.default_ffn_norm = default_ffn_norm
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_complexity = attention_complexity
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.pre_attn_ln_type = pre_attn_ln_type
            self.post_attn_ln_type = post_attn_ln_type
            self.max_position_embeddings = max_position_embeddings
            self.token_type_embeddings = token_type_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.pos_emb_type = PositionalEmbeddingsTypes[pos_emb_type.upper()]
            self.embedding_ln_type = embedding_ln_type
            self.embedding_dropout = embedding_dropout
            self.relpe_type = relpe_type
            self.relpe_scheme = relpe_scheme
            self.final_ln_type = final_ln_type
            self.pooler_function = pooler_function
            self.pooler_no_dense = pooler_no_dense
            self.pooler_act = pooler_act
            self.pooler_ln_type = pooler_ln_type
            self.classifier_bias = classifier_bias
            self.lm_head_act = lm_head_act
            self.lm_head_ln_type = lm_head_ln_type
            self.causal = causal
            self.chunk_size = chunk_size
            self.local_attention = local_attention
            self.window_size = window_size
            self.local_relpe = local_relpe
            self.local_scheme = local_scheme
            self.hybrid = hybrid
            self.transformer_heads = transformer_heads
            self.attn_proj_biases = attn_proj_biases
            self.hf_config = None
            if hf_model_name_or_path:
                self.hf_config = AutoConfig.from_pretrained(hf_model_name_or_path)
                self.hidden_size = self.hf_config.hidden_size
                self.num_attention_heads = self.hf_config.num_attention_heads
                self.num_hidden_layers = self.hf_config.num_hidden_layers
                self.hidden_act = self.hf_config.hidden_act
                self.vocab_size = self.hf_config.vocab_size
                self.hf_config.max_position_embeddings = self.max_position_embeddings
                
            for key, value in kwargs.items():
                setattr(self, key, value)

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
