from dataclasses import dataclass
from src.core.embeddings.base import BaseTextEmbeddingsGenerator
from enum import Enum 

@dataclass
class TextChunk:
    ref: str | None # reference to the document name
    text: str
    start_idx: int
    end_idx: int
    token_count: int

    def __repr__(self):
        return f"{self.__class__.__name__}(text={self.text}, reference={self.ref})"
    
class ChunkingMode(Enum):
    TOKEN = "token"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"

@dataclass
class ChunkerParams:
    chunk_size: int = None
    chunk_overlap: int = None
    embedder: BaseTextEmbeddingsGenerator = None