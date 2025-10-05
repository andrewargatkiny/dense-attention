from ..danet_layers import DANetLayer, DANetLayerWithLocalAttention
from ..model_config import DANetLayerConfig
from ..other_models.modeling import BertLayer, TransformerLayerConfig


LAYER_TYPE2CLASS = {
    'danet': DANetLayer,
    'danet_with_local_attention': DANetLayerWithLocalAttention,
    'transformer': BertLayer,
}

LAYER_TYPE2CONFIG_CLASS = {
    'transformer': TransformerLayerConfig,
    'danet': DANetLayerConfig,
    'danet_with_local_attention': DANetLayerConfig,
}