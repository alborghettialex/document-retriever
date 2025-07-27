from dataclasses import dataclass
import numpy as np

@dataclass
class TextEmbedding:
    text: str
    vector: list | np.ndarray

    @property
    def dim(self) -> int:
        """
        The dimensionality of the embedding space.
        """
        return len(self.vector)
    
    def __repr__(self):
        return f"text={self.text}, vector_dim={self.dim}"