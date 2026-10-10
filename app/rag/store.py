"""Vector store abstraction for RAG."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class VectorRecord:
    """A stored vector record with text, embedding, and metadata."""

    id: str
    text: str
    embedding: List[float]
    metadata: dict[str, Any]


@dataclass
class SearchResult:
    """A search result with record and similarity score."""

    record: VectorRecord
    score: float


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, records: List[VectorRecord]) -> None:
        """Add records to the store.

        Args:
            records: List of VectorRecord objects to add.

        Raises:
            ValueError: If any record has invalid embedding dimension
                or if records have inconsistent dimensions.
        """
        ...

    @abstractmethod
    def search(
        self, query_embedding: List[float], k: int = 5
    ) -> List[SearchResult]:
        """Search for the top-k most similar records.

        Args:
            query_embedding: The query vector to search for.
            k: Number of top results to return (default: 5).

        Returns:
            List of SearchResult objects sorted by similarity (highest first).
            Returns empty list if store is empty or k <= 0.

        Raises:
            ValueError: If query_embedding has invalid dimension.
        """
        ...

    @abstractmethod
    def delete_by_metadata(self, metadata_filter: dict[str, Any]) -> int:
        """Delete records matching a metadata filter.

        Args:
            metadata_filter: Dictionary of key/value pairs. A record matches
                if its metadata contains all specified key/value pairs.

        Returns:
            Number of records deleted.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all records from the store."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of records in the store."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> Optional[int]:
        """Return the embedding dimension of stored vectors, or None if empty."""
        ...


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store using cosine similarity.

    Uses numpy for efficient similarity computation.
    Validates embedding dimensions on add and search.
    """

    def __init__(self):
        self._records: List[VectorRecord] = []
        self._dimension: Optional[int] = None

    def add(self, records: List[VectorRecord]) -> None:
        if not records:
            return

        # Validate all records have embeddings
        for record in records:
            if record.embedding is None or len(record.embedding) == 0:
                raise ValueError(f"Record {record.id} has empty embedding")

        # Determine and validate dimension
        new_dim = len(records[0].embedding)
        for record in records[1:]:
            if len(record.embedding) != new_dim:
                raise ValueError("All records must have the same embedding dimension")

        if self._dimension is None:
            self._dimension = new_dim
        elif new_dim != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: store has {self._dimension}, "
                f"new records have {new_dim}"
            )

        self._records.extend(records)

    def search(
        self, query_embedding: List[float], k: int = 5
    ) -> List[SearchResult]:
        if k <= 0:
            return []

        if not self._records:
            return []

        if self._dimension is None:
            return []

        if len(query_embedding) != self._dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self._dimension}, "
                f"got {len(query_embedding)}"
            )

        import numpy as np

        # Convert to numpy arrays
        query_vec = np.array(query_embedding, dtype=np.float32)
        record_vecs = np.array([r.embedding for r in self._records], dtype=np.float32)

        # Cosine similarity: (A · B) / (||A|| ||B||)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        record_norms = np.linalg.norm(record_vecs, axis=1)
        # Avoid division by zero
        valid_norms = record_norms > 0
        if not np.any(valid_norms):
            return []

        # Compute similarities for valid records
        similarities = np.full(len(self._records), -1.0, dtype=np.float32)  # -1 for invalid
        dot_products = np.dot(record_vecs[valid_norms], query_vec)
        similarities[valid_norms] = dot_products / (
            record_norms[valid_norms] * query_norm
        )

        # Get top-k indices (highest similarity first)
        # Only consider records with valid norms (similarity != -1)
        valid_indices = np.where(similarities != -1.0)[0]
        if len(valid_indices) == 0:
            return []

        top_k = min(k, len(valid_indices))
        top_valid_indices = np.argsort(similarities[valid_indices])[-top_k:][::-1]
        top_indices = valid_indices[top_valid_indices]

        results = []
        for idx in top_indices:
            results.append(
                SearchResult(record=self._records[idx], score=float(similarities[idx]))
            )

        return results

    def clear(self) -> None:
        self._records.clear()
        self._dimension = None

    def delete_by_metadata(self, metadata_filter: dict[str, Any]) -> int:
        if not metadata_filter:
            return 0

        original_count = len(self._records)
        self._records = [
            r for r in self._records
            if not all(r.metadata.get(k) == v for k, v in metadata_filter.items())
        ]
        deleted = original_count - len(self._records)

        # Reset dimension if store becomes empty
        if not self._records:
            self._dimension = None

        return deleted

    def count(self) -> int:
        return len(self._records)

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension