from ..danet_layers import DANetLayerForMixing
from ..model_config import DANetLayerConfig
from ..other_models.modeling import BertLayer, TransformerLayerConfig


LAYER_TYPE2CLASS = {
    'danet': DANetLayerForMixing,
    'transformer': BertLayer,
}

LAYER_TYPE2CONFIG_CLASS = {
    'transformer': TransformerLayerConfig,
    'danet': DANetLayerConfig,
}