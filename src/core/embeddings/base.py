from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.core.embeddings.models import TextEmbedding
from src.core.text.chunk.models import TextChunk

@dataclass
class BaseTextEmbeddingsGenerator(ABC):
    """
    Abstract base class for embedding generation.
    """

    model_name: str 

    def get_embedding(self, text: str, return_numpy: bool) -> TextEmbedding:
        """
        Method to get a text embedding. 

        Args:
            text (str): the text to embed.
            return_numpy (bool): whether to return an np.array or a standard list.
        
        Returns:
            TextEmbedding: the generated embedding.
        """
        return self.get_embeddings([text], batch_size=1, return_numpy=return_numpy)[0]


    @abstractmethod
    def get_embeddings(self, chunks: list[TextChunk], batch_size: int, return_numpy: bool) -> list[TextEmbedding]:
        """
        Generate embeddings for a batch of texts.

        Args:
            chunks (list[TextChunk]): List of text chunks to be embedded.
            batch_size (int): Number of texts to process in each batch.
            return_numpy (bool): If True, embeddings are returned as numpy arrays; 
                                otherwise, as standard Python lists.
        
        Returns:
            list[TextEmbedding]: A list of embeddings corresponding to the input texts.
        """
        ...
