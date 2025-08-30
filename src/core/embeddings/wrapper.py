from chonkie import BaseEmbeddings
from src.core.embeddings.algorithms import HFDenseTextEmbeddingsGenerator
from transformers import AutoTokenizer

class ChonkieEmbeddingWrapper(BaseEmbeddings):
    """
    A wrapper class for generating embeddings for Chonkie's semantic chunker.
    """
    def __init__(self, embedder: HFDenseTextEmbeddingsGenerator):
        super().__init__()
        self.embedder = embedder
        
    @property
    def dimension(self) -> int:
        return self.embedder.dim

    def embed(self, text: str) -> list[float]:
        emb = self.embedder.get_embedding(text, return_numpy=True)
        return emb.vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def get_tokenizer_or_token_counter(self):
        return AutoTokenizer.from_pretrained(self.embedder.model_name)

