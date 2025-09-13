from dataclasses import dataclass

@dataclass
class EmbedderParams:
    model_name: str = None
    dim: int = None