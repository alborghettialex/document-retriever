from dataclasses import dataclass
from src.core.embeddings.base import BaseTextEmbeddingsGenerator
import numpy as np
from fastembed import TextEmbedding as DenseTextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource
from src.core.embeddings.models import TextEmbedding
from src.core.embeddings.registry import register_embedder
from src.core.text.chunk.models import TextChunk

@register_embedder("hf-dense")
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

    def get_embeddings(self, chunks: list[TextChunk], return_numpy: bool, batch_size: int) -> list[TextEmbedding]:
        texts = [chunk.text for chunk in chunks]
        vectors = list(self.model.embed(texts, batch_size=batch_size))

        if return_numpy:
            return [TextEmbedding(text, vector) for text, vector in zip(texts, vectors)]
        else: 
            return [TextEmbedding(text, vector.tolist()) for text, vector in zip(texts, vectors)]