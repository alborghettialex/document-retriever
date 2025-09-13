from chonkie import BaseEmbeddings
from src.core.embeddings.base import BaseTextEmbeddingsGenerator
from transformers import AutoTokenizer
from src.core.text.chunk.models import TextChunk

class ChonkieEmbeddingWrapper(BaseEmbeddings):
    """
    A wrapper class for generating embeddings for Chonkie's semantic chunker.
    """
    def __init__(self, embedder: BaseTextEmbeddingsGenerator):
        super().__init__()
        self.embedder = embedder
        
    @property
    def dimension(self) -> int:
        return self.embedder.params.dim

    def embed(self, text: str) -> list[float]:
        emb = self.embedder.get_embedding(text, return_numpy=True)
        return emb.vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def get_tokenizer_or_token_counter(self):
        return AutoTokenizer.from_pretrained(self.embedder.params.model_name)