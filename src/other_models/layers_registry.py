from ..danet_layers import DANetLayer, DANetLayerWithLocalAttention, DANetLayerConfig, DANetLayerWithLocalAttentionConfig
from ..other_models.modeling_new import BertLayer, TransformerConfig


LAYER_TYPE2CLASS = {
    'danet': DANetLayer,
    'danet_with_local_attention': DANetLayerWithLocalAttention,
    'transformer': BertLayer,
}

LAYER_TYPE2CONFIG_CLASS = {
    'transformer': TransformerConfig,
    'danet': DANetLayerConfig,
    'danet_with_local_attention': DANetLayerWithLocalAttentionConfig,
}