"""Tests for RAG retrieval pipeline."""

import pytest

from app.rag.retrieve import retrieve
from app.rag.store import InMemoryVectorStore, SearchResult, VectorRecord
from app.ai.embeddings import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 384, should_fail: bool = False, query_embedding: list[float] = None):
        self._dimension = dimension
        self._should_fail = should_fail
        self._query_embedding = query_embedding
        self.last_query = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._should_fail:
            raise RuntimeError("Embedding API error")
        if self._query_embedding is not None:
            return [self._query_embedding]
        # Return a simple deterministic embedding based on text
        return [[0.1] * self._dimension] * len(texts)


class TestRetrieve:
    """Tests for retrieve function."""

    def test_retrieve_returns_relevant_chunks(self):
        """Query retrieves expected relevant chunks."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4)

        # Add records with known embeddings
        store.add([
            VectorRecord(
                id="doc-1_chunk_0",
                text="Apple fruit information",
                embedding=[1.0, 0.0, 0.0, 0.0],
                metadata={"doc_id": "doc-1", "chunk_index": 0},
            ),
            VectorRecord(
                id="doc-1_chunk_1",
                text="Banana fruit details",
                embedding=[0.0, 1.0, 0.0, 0.0],
                metadata={"doc_id": "doc-1", "chunk_index": 1},
            ),
            VectorRecord(
                id="doc-2_chunk_0",
                text="Car engine specifications",
                embedding=[0.0, 0.0, 1.0, 0.0],
                metadata={"doc_id": "doc-2", "chunk_index": 0},
            ),
        ])

        # Query embedding matches first record
        embedder = MockEmbeddingProvider(dimension=4, query_embedding=[1.0, 0.0, 0.0, 0.0])
        results = retrieve("apple", embedder, store, top_k=3)

        assert len(results) == 3
        assert results[0].record.text == "Apple fruit information"
        assert results[0].score == 1.0  # Perfect match
        # Remaining two are orthogonal (score 0.0), order not guaranteed
        texts = {r.record.text for r in results[1:]}
        assert texts == {"Banana fruit details", "Car engine specifications"}

    def test_retrieve_passes_top_k_to_store(self):
        """top_k parameter limits results."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4, query_embedding=[0.1] * 4)

        # Add 5 records
        for i in range(5):
            store.add([
                VectorRecord(
                    id=f"doc_chunk_{i}",
                    text=f"Content {i}",
                    embedding=[0.1] * 4,
                    metadata={"doc_id": "doc", "chunk_index": i},
                )
            ])

        results = retrieve("test", embedder, store, top_k=2)
        assert len(results) == 2

        results = retrieve("test", embedder, store, top_k=10)
        assert len(results) == 5  # Only 5 records exist

    def test_retrieve_returns_search_results_with_scores(self):
        """Results preserve SearchResult structure with scores and metadata."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4, query_embedding=[1.0, 0.0, 0.0, 0.0])

        store.add([
            VectorRecord(
                id="doc-1_chunk_0",
                text="Relevant text",
                embedding=[1.0, 0.0, 0.0, 0.0],
                metadata={"doc_id": "doc-1", "chunk_index": 0, "source": "test.pdf"},
            ),
        ])

        results = retrieve("query", embedder, store, top_k=1)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].record.text == "Relevant text"
        assert results[0].record.metadata["doc_id"] == "doc-1"
        assert results[0].record.metadata["source"] == "test.pdf"
        assert results[0].score == 1.0

    def test_retrieve_blank_query_raises(self):
        """Blank query raises ValueError."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4)

        with pytest.raises(ValueError, match="Query must not be empty"):
            retrieve("", embedder, store)

        with pytest.raises(ValueError, match="Query must not be empty"):
            retrieve("   ", embedder, store)

    def test_retrieve_invalid_top_k_raises(self):
        """Non-positive top_k raises ValueError."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4)

        with pytest.raises(ValueError, match="top_k must be positive"):
            retrieve("query", embedder, store, top_k=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            retrieve("query", embedder, store, top_k=-1)

    def test_retrieve_empty_store_returns_empty(self):
        """Empty store returns empty list without embedding call."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4)

        results = retrieve("query", embedder, store, top_k=5)
        assert results == []

    def test_retrieve_embedding_failure_propagates(self):
        """Embedding failures propagate as RuntimeError."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4, should_fail=True)

        with pytest.raises(RuntimeError, match="Query embedding failed"):
            retrieve("query", embedder, store)

    def test_retrieve_dimension_mismatch_raises(self):
        """Query embedding dimension mismatch with store raises ValueError."""
        store = InMemoryVectorStore()
        # Store has 3D embeddings
        store.add([
            VectorRecord(
                id="1",
                text="test",
                embedding=[1.0, 0.0, 0.0],
                metadata={},
            ),
        ])

        # Embedder returns 4D embedding
        embedder = MockEmbeddingProvider(dimension=4, query_embedding=[0.1] * 4)

        with pytest.raises(ValueError, match="Query embedding dimension mismatch"):
            retrieve("query", embedder, store)

    def test_retrieve_zero_norm_query_returns_empty(self):
        """Zero-norm query embedding returns empty results."""
        store = InMemoryVectorStore()
        store.add([
            VectorRecord(
                id="1",
                text="test",
                embedding=[1.0, 0.0],
                metadata={},
            ),
        ])

        # Zero embedding
        embedder = MockEmbeddingProvider(dimension=2, query_embedding=[0.0, 0.0])

        results = retrieve("query", embedder, store)
        assert results == []

    def test_retrieve_preserves_order_by_similarity(self):
        """Results are sorted by similarity descending (negative similarities excluded)."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=2)

        # Add records at different angles from query [1, 0]
        store.add([
            VectorRecord(id="a", text="a", embedding=[1.0, 0.0], metadata={}),      # 0 deg, score 1.0
            VectorRecord(id="b", text="b", embedding=[0.707, 0.707], metadata={}),  # 45 deg, score ~0.707
            VectorRecord(id="c", text="c", embedding=[0.0, 1.0], metadata={}),      # 90 deg, score 0.0
            VectorRecord(id="d", text="d", embedding=[-1.0, 0.0], metadata={}),     # 180 deg, score -1.0 (excluded)
        ])

        embedder = MockEmbeddingProvider(dimension=2, query_embedding=[1.0, 0.0])
        results = retrieve("query", embedder, store, top_k=4)

        # Negative similarity (-1.0) is filtered out by store, so only 3 results
        assert len(results) == 3
        assert results[0].record.id == "a"
        assert results[1].record.id == "b"
        assert results[2].record.id == "c"
        assert results[0].score > results[1].score > results[2].score