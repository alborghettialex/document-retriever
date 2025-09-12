from src.core.text.text_extraction import TextExtractor
from src.core.text.chunk.base import BaseTextChunker 
from src.core.embeddings.base import BaseTextEmbeddingsGenerator
from src.core.storage.base import BaseVectorDB
from typing import Union

def process_document(
        file_path: Union[str, bytes], 
        text_extractor: TextExtractor, 
        chunker: BaseTextChunker, 
        embedder: BaseTextEmbeddingsGenerator, 
        db: BaseVectorDB,
        batch_size: int = 100
    ):
    text = text_extractor.extract_text(file_path)
    chunks = chunker.get_chunks(text)
    vectors = embedder.get_embeddings(chunks, batch_size=32, return_numpy=False)
    metadata = [{"text": c.text} for c in chunks]

    for i in range(0, len(vectors), batch_size):
        db.insert(
            vectors[i:i + batch_size],
            metadata[i:i + batch_size]
        )
