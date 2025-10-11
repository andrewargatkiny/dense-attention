from ..danet_layers import DANetLayerWrapper
from ..model_config import ModelConfig
from ..other_models.transformers import BertLayer, TransformerLayerConfig


LayerTypeToClass = {
    'danet': DANetLayerWrapper,
    'transformer': BertLayer,
}

LayerConfigToClass = {
    'transformer': TransformerLayerConfig,
    'danet': ModelConfig,
}