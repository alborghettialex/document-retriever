from dataclasses import dataclass
from src.core.embeddings.base import BaseTextEmbeddingsGenerator
import numpy as np
from fastembed import TextEmbedding as DenseTextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource
from src.core.embeddings.models import TextEmbedding

@dataclass
class HFDenseTextEmbeddingsGenerator(BaseTextEmbeddingsGenerator):
    """
    Embeddings generation ONNX with HuggingFace models.
    """ 
    model_name: str
    dim: int # dimensionality of the embedding space

    def __post_init__(self):
        supported_models = [supported['model'] for supported in DenseTextEmbedding.list_supported_models()]

        if self.model_name not in supported_models:
            DenseTextEmbedding.add_custom_model(
                model=self.model_name,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=self.model_name),  # can be used with an `url` to load files from a private storage
                dim=self.dim,
                model_file="onnx/model.onnx",  # can be used to load an already supported model with another optimization or quantization, e.g. onnx/model_O4.onnx
            )

        self.model = DenseTextEmbedding(model_name=self.model_name)

    def get_embedding(self, text: str, return_numpy: bool) -> TextEmbedding:
        vector = list(self.model.embed(text))[0] 

        if return_numpy:
            return TextEmbedding(text, vector)
        else: 
            return TextEmbedding(text, vector.tolist())