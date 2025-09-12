from typing import Type
from src.core.embeddings.base import BaseTextEmbeddingsGenerator

_EMBEDDER_REGISTRY: dict[str, Type[BaseTextEmbeddingsGenerator]] = {}

def register_embedder(name: str):
    def decorator(cls: Type[BaseTextEmbeddingsGenerator]):
        _EMBEDDER_REGISTRY[name] = cls
        return cls
    return decorator

def get_embedder(name: str, **kwargs) -> BaseTextEmbeddingsGenerator:
    try:
        cls = _EMBEDDER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Embedder '{name}' not found in registry")
    return cls(**kwargs)