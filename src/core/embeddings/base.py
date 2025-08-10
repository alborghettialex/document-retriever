import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.core.embeddings.models import TextEmbedding

@dataclass
class TextEmbeddingsGenerator(ABC):
    """
    Abstract base class for embedding generation.
    """

    model_name: str 

    @abstractmethod
    def get_embedding(text: str, return_tensor: bool) -> TextEmbedding:
        """
        Method to get a text embedding. 

        Args:
            text (str): the text to embed.
            return_tensor (bool): whether to return an np.array or a standard list.
        
        Returns:
            TextEmbedding: the generated embedding.
        """
        ...
