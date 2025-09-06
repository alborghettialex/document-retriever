from chonkie import TokenChunker, SentenceChunker, SemanticChunker, Chunk
from dataclasses import dataclass
from src.core.text.chunk.base import BaseTextChunker
from src.core.text.chunk.models import ChunkingMode, TextChunk, ChunkerParams
from src.core.text.chunk.wrapper import ChonkieEmbeddingWrapper
from typing import ClassVar

@dataclass 
class ChonkieChunker(BaseTextChunker):
    """
    Class for text chunking with chonkie library. 
    """

    mode: ChunkingMode
    supported_modes: ClassVar[set[ChunkingMode]] = {
        ChunkingMode.TOKEN,
        ChunkingMode.SENTENCE, 
        ChunkingMode.SEMANTIC
        }
    params: ChunkerParams
    
    def get_chunks(self, text: str, doc_name: str | None = None) -> list[TextChunk]:
        dispatcher = {
            ChunkingMode.TOKEN: self._chunk_by_token,
            ChunkingMode.SENTENCE: self._chunk_by_sentence,
            ChunkingMode.SEMANTIC: self._chunk_by_semantic,
        }
        
        if self.mode not in dispatcher:
            raise ValueError(f"Unsupported mode: {self.mode}")
        
        chunk_func = dispatcher[self.mode]
        return chunk_func(text, doc_name)
    
    def _chunk_by_token(self, text: str, doc_name: str | None = None) -> list[TextChunk]: 
        token_chunker = TokenChunker(chunk_size=self.params.chunk_size, chunk_overlap=self.params.chunk_overlap)
        chonkie_chunks = token_chunker.chunk(text)
        return self._format_chunks(chonkie_chunks, doc_name)
    
    def _chunk_by_sentence(self, text: str, doc_name: str | None = None) -> list[TextChunk]: 
        sentence_chunker = SentenceChunker(chunk_size=self.params.chunk_size, chunk_overlap=self.params.chunk_overlap)
        chonkie_chunks = sentence_chunker.chunk(text)
        return self._format_chunks(chonkie_chunks, doc_name)
    
    def _chunk_by_semantic(self, text: str, doc_name: str | None = None) -> list[TextChunk]: 
        chonkie_embedder = ChonkieEmbeddingWrapper(embedder=self.params.embedder)
        semantic_chunker = SemanticChunker(embedding_model=chonkie_embedder, chunk_size=self.params.chunk_size)
        chonkie_chunks = semantic_chunker.chunk(text)
        return self._format_chunks(chonkie_chunks, doc_name)
    
    def _format_chunks(self, chonkie_chunks: list[Chunk], doc_name: str) -> list[TextChunk]:
        chunk_li: list[TextChunk] = []

        for chunk in chonkie_chunks:
            chunk_li.append(
                TextChunk(
                    ref = doc_name if doc_name is not None else "unknown",
                    text=chunk.text,
                    start_idx=chunk.start_index,
                    end_idx=chunk.end_index,
                    token_count=chunk.token_count
                )
            )
        return chunk_li