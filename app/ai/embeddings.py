"""Embedding provider abstraction for RAG."""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension (number of floats per vector)."""
        ...

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors, one per input text, in the same order.
            Returns empty list for empty input.
        """
        ...