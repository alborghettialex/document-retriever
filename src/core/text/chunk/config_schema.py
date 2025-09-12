from dataclasses import dataclass
from src.core.embeddings.base import BaseTextEmbeddingsGenerator

@dataclass
class ChunkerParams:
    chunk_size: int = None
    chunk_overlap: int = None
    embedder: BaseTextEmbeddingsGenerator = None