from typing import Type
from src.core.text.chunk.base import BaseTextChunker

_CHUNKER_REGISTRY: dict[str, Type[BaseTextChunker]] = {}

def register_chunker(name: str):
    def decorator(cls: Type[BaseTextChunker]):
        _CHUNKER_REGISTRY[name] = cls
        return cls
    return decorator

def get_chunker(name: str, **kwargs) -> BaseTextChunker:
    try:
        cls = _CHUNKER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Chunker '{name}' not found in registry")
    return cls(**kwargs)