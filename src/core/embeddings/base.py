import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class TextEmbeddingsGenerator(ABC):
    """
    Abstract base class for embedding generation.
    """

    model_name: str 

    @abstractmethod
    def get_embedding(text: str, return_tensor: bool) -> list | np.ndarray:
        """
        Method to get a text embedding. 

        Args:
            text (str): the text to embed.
            return_tensor (bool): whether to return an np.array or a standard list.
        
        Returns:
            list|np.ndarray: the generated embedding.
        """
        ...
