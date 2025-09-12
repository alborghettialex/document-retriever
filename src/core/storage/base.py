from abc import ABC, abstractmethod
from src.core.embeddings.models import TextEmbedding
from typing import Any

class BaseVectorDB(ABC):
    @abstractmethod
    def insert(self, text_embeddings: list[TextEmbedding], metadata: list[dict]):
        """
        Insert a list of text embeddings into the vector database.

        Args:
            text_embeddings (list[TextEmbedding]): The embeddings to insert.
            metadata (list[dict]): A list of metadata dictionaries, one for each embedding.
        """
        ...

    @abstractmethod
    def query(self, query_text_embedding: TextEmbedding, top_k: int = 5):
        """
        Search the vector database for the most similar embeddings.

        Args:
            query_text_embedding (TextEmbedding): The embedding to compare against the database.
            top_k (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            list: A list of search results with similarity scores and associated metadata.
        """
        ...

    @abstractmethod
    def delete(self, ids: Any):
        """
        Delete one or more entries from the vector database.

        Args:
            ids (Any): The IDs of the entries to delete.
        """
        ...