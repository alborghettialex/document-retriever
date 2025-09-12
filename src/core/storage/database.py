from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, PointsSelector
from src.core.storage.base import BaseVectorDB
from src.core.storage.registry import register_vectordb
from src.core.embeddings.models import TextEmbedding
from dataclasses import dataclass

@register_vectordb("qdrant")
@dataclass
class QdrantVectorDB(BaseVectorDB):
    """
    Qdrant vector database.
    """
    
    collection_name: str 
    host: str = "localhost"
    port: int = 6333

    def __post_init__(self):
        self.client = QdrantClient(host=self.host, port=self.port)

    def insert(self, text_embeddings: list[TextEmbedding], metadata: list[dict]):
        vectors = [text_embedding.vector for text_embedding in text_embeddings]
        dim = text_embeddings[0].dim

        if self.collection_name not in [col.name for col in self.client.get_collections().collections]:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

        points = [
            PointStruct(
                id=i,
                vector=vector,
                payload=meta
            )
            for i, (vector, meta) in enumerate(zip(vectors, metadata))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, query_text_embedding: TextEmbedding, top_k: int = 5):
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_text_embedding.vector,
            limit=top_k
        )

    def delete(self, ids: PointsSelector):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector={"points": ids}
        )
