from typing import Type
from src.core.storage.base import BaseVectorDB

_VECTORDB_REGISTRY: dict[str, Type[BaseVectorDB]] = {}

def register_vectordb(name: str):
    def decorator(cls: Type[BaseVectorDB]):
        _VECTORDB_REGISTRY[name] = cls
        return cls
    return decorator

def get_vectordb(name: str, **kwargs) -> BaseVectorDB:
    try:
        cls = _VECTORDB_REGISTRY[name]
    except KeyError:
        raise ValueError(f"VectorDB '{name}' not found in registry")
    return cls(**kwargs)