from ..danet_layers import DANetLayerWithLocalAttention
from ..model_config import ModelConfig
from ..other_models.transformers import BertLayer, TransformerLayerConfig


LayerTypeToClass = {
    'danet': DANetLayerWithLocalAttention,
    'transformer': BertLayer,
}

LayerTypeToConfigClass = {
    'transformer': TransformerLayerConfig,
    'danet': ModelConfig,
}