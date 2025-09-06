from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.core.text.chunk.models import TextChunk, ChunkingMode, ChunkerParams
from typing import ClassVar

@dataclass
class BaseTextChunker(ABC):
    """
    Abstract base class for text chunking
    """

    mode: ChunkingMode
    supported_modes: ClassVar[set[ChunkingMode]] = set()
    params: ChunkerParams

    required_params_map: ClassVar[dict[ChunkingMode, set[str]]] = {
            ChunkingMode.TOKEN: {"chunk_size", "chunk_overlap"},
            ChunkingMode.SENTENCE: {"chunk_size", "chunk_overlap"},
            ChunkingMode.SEMANTIC: {"chunk_size", "embedder"}
        }

    def __post_init__(self):
        if self.mode not in self.supported_modes:
            raise ValueError(
                f"Mode '{self.mode}' is not supported by {self.__class__.__name__}. "
                f"Supported modes: {[m.value for m in self.supported_modes]}."
            )

        required = self.required_params_map.get(self.mode, set())
        missing_params = [
            p for p in required
            if getattr(self.params, p, None) is None
        ]
        if missing_params:
            raise ValueError(
                f"Missing required parameters: {missing_params} for chunking mode '{self.mode.value}'."
            )

    @abstractmethod
    def get_chunks(text: str) -> list[TextChunk]:
        """
        Method to get chunks from a document. 

        Args:
            text (str): the text to chunk.

        Returns:
            list[TextChunk]: the generated chunks.
        """
        ...