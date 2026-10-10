"""Tests for document ingestion."""

import pytest

from app.rag.ingest import chunk_text, ingest_document
from app.rag.store import InMemoryVectorStore, VectorRecord
from app.ai.embeddings_local import LocalEmbeddingProvider
from app.ai.embeddings import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 384, should_fail: bool = False, wrong_count: int = 0):
        self._dimension = dimension
        self._should_fail = should_fail
        self._wrong_count = wrong_count
        self.call_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        if self._should_fail:
            raise RuntimeError("Embedding API error")
        if self._wrong_count:
            return [[0.1] * self._dimension] * self._wrong_count
        return [[0.1] * self._dimension] * len(texts)


class TestIngestDocument:
    """Tests for ingest_document function."""

    def test_successful_ingestion_basic(self):
        """Basic ingestion creates correct number of chunks with metadata."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4)

        text = "This is a test document. " * 20  # Long enough for multiple chunks
        count = ingest_document(
            text=text,
            doc_id="doc-1",
            store=store,
            embedder=embedder,
            chunk_size=100,
            overlap=20,
        )

        assert count > 0
        assert store.count() == count
        assert store.dimension == 4

    def test_ingestion_creates_stable_ids_and_metadata(self):
        """Chunks get stable IDs and correct metadata."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4)

        text = "A" * 250  # 3 chunks with chunk_size=100, overlap=20
        ingest_document(
            text=text,
            doc_id="doc-abc",
            store=store,
            embedder=embedder,
            chunk_size=100,
            overlap=20,
        )

        # Check all records have correct structure
        assert store.count() == 3
        # Search to retrieve records
        results = store.search([0.1] * 4, k=10)
        assert len(results) == 3

        for result in results:
            record = result.record
            assert record.id.startswith("doc-abc_chunk_")
            assert record.metadata["doc_id"] == "doc-abc"
            assert "chunk_index" in record.metadata
            assert isinstance(record.metadata["chunk_index"], int)

    def test_ingestion_metadata_merged(self):
        """Optional source metadata is merged into each chunk."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4)

        text = "Short text"
        ingest_document(
            text=text,
            doc_id="doc-1",
            store=store,
            embedder=embedder,
            metadata={"source": "test.pdf", "page": 5},
        )

        results = store.search([0.1] * 4, k=10)
        for result in results:
            assert result.record.metadata["source"] == "test.pdf"
            assert result.record.metadata["page"] == 5
            assert result.record.metadata["doc_id"] == "doc-1"

    def test_reingestion_replaces_old_chunks(self):
        """Re-ingesting same doc_id replaces old chunks."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4)

        # First ingestion
        text1 = "First version of the document. " * 10
        count1 = ingest_document(
            text=text1,
            doc_id="doc-1",
            store=store,
            embedder=embedder,
            chunk_size=100,
            overlap=20,
        )

        # Second ingestion with different text
        text2 = "Second version completely different content. " * 10
        count2 = ingest_document(
            text=text2,
            doc_id="doc-1",
            store=store,
            embedder=embedder,
            chunk_size=100,
            overlap=20,
        )

        # Should have only new chunks
        assert store.count() == count2
        # Old chunks should be gone
        results = store.search([0.1] * 4, k=100)
        assert len(results) == count2

    def test_empty_text_removes_existing_chunks(self):
        """Empty text removes existing chunks for doc_id."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4)

        # Ingest first
        ingest_document(
            text="Some content here",
            doc_id="doc-1",
            store=store,
            embedder=embedder,
        )
        assert store.count() > 0

        # Re-ingest with empty text
        count = ingest_document(
            text="",
            doc_id="doc-1",
            store=store,
            embedder=embedder,
        )

        assert count == 0
        assert store.count() == 0

    def test_whitespace_text_removes_existing_chunks(self):
        """Whitespace-only text removes existing chunks."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4)

        ingest_document(
            text="Content",
            doc_id="doc-1",
            store=store,
            embedder=embedder,
        )
        assert store.count() > 0

        count = ingest_document(
            text="   \n\t  ",
            doc_id="doc-1",
            store=store,
            embedder=embedder,
        )

        assert count == 0
        assert store.count() == 0

    def test_empty_text_on_empty_store_returns_zero(self):
        """Empty text on empty store returns 0 without error."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4)

        count = ingest_document(
            text="",
            doc_id="doc-1",
            store=store,
            embedder=embedder,
        )

        assert count == 0
        assert store.count() == 0

    def test_embedding_count_mismatch_raises(self):
        """Mismatched embedding count raises ValueError."""
        store = InMemoryVectorStore()
        # Provider returns 5 embeddings regardless of input
        embedder = MockEmbeddingProvider(dimension=4, wrong_count=5)

        text = "Short text"  # Will produce 1 chunk
        with pytest.raises(ValueError, match="Embedding count mismatch"):
            ingest_document(
                text=text,
                doc_id="doc-1",
                store=store,
                embedder=embedder,
            )

    def test_embedding_failure_preserves_existing_chunks(self):
        """Embedding failure does not delete previously stored chunks."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4, should_fail=True)

        # First ingestion succeeds
        ingest_document(
            text="Original content",
            doc_id="doc-1",
            store=store,
            embedder=MockEmbeddingProvider(dimension=4),  # Different instance, works
        )
        original_count = store.count()
        assert original_count > 0

        # Second ingestion fails
        with pytest.raises(RuntimeError, match="Embedding failed"):
            ingest_document(
                text="New content that will fail",
                doc_id="doc-1",
                store=store,
                embedder=embedder,  # This one fails
            )

        # Original chunks should still be there
        assert store.count() == original_count

    def test_embedding_failure_on_first_ingestion_leaves_store_empty(self):
        """Embedding failure on first ingestion leaves store empty."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4, should_fail=True)

        with pytest.raises(RuntimeError, match="Embedding failed"):
            ingest_document(
                text="Content",
                doc_id="doc-1",
                store=store,
                embedder=embedder,
            )

        assert store.count() == 0

    def test_embedding_validation_before_deletion(self):
        """Embeddings are validated before deleting old data."""
        store = InMemoryVectorStore()
        working_embedder = MockEmbeddingProvider(dimension=4)
        failing_embedder = MockEmbeddingProvider(dimension=4, should_fail=True)

        # Ingest with working embedder
        ingest_document(
            text="Original",
            doc_id="doc-1",
            store=store,
            embedder=working_embedder,
        )
        assert store.count() > 0

        # Try re-ingest with failing embedder
        with pytest.raises(RuntimeError):
            ingest_document(
                text="New content",
                doc_id="doc-1",
                store=store,
                embedder=failing_embedder,
            )

        # Original data preserved
        assert store.count() > 0


class TestDeleteByMetadata:
    """Tests for delete_by_metadata on InMemoryVectorStore."""

    def test_delete_by_doc_id(self):
        """Deleting by doc_id removes only that document's chunks."""
        store = InMemoryVectorStore()

        # Add chunks for two documents
        store.add([
            VectorRecord(id="doc-1_chunk_0", text="a", embedding=[1.0, 0.0], metadata={"doc_id": "doc-1", "chunk_index": 0}),
            VectorRecord(id="doc-1_chunk_1", text="b", embedding=[0.0, 1.0], metadata={"doc_id": "doc-1", "chunk_index": 1}),
            VectorRecord(id="doc-2_chunk_0", text="c", embedding=[1.0, 1.0], metadata={"doc_id": "doc-2", "chunk_index": 0}),
        ])
        assert store.count() == 3

        # Delete doc-1
        deleted = store.delete_by_metadata({"doc_id": "doc-1"})
        assert deleted == 2
        assert store.count() == 1

        remaining = store.search([1.0, 0.0], k=10)
        assert len(remaining) == 1
        assert remaining[0].record.metadata["doc_id"] == "doc-2"

    def test_delete_by_multiple_metadata_keys(self):
        """Filter matches all key/value pairs."""
        store = InMemoryVectorStore()

        store.add([
            VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={"doc_id": "doc-1", "type": "rfp"}),
            VectorRecord(id="2", text="b", embedding=[0.0, 1.0], metadata={"doc_id": "doc-1", "type": "proposal"}),
            VectorRecord(id="3", text="c", embedding=[1.0, 1.0], metadata={"doc_id": "doc-2", "type": "rfp"}),
        ])

        # Delete only doc-1 AND type=rfp
        deleted = store.delete_by_metadata({"doc_id": "doc-1", "type": "rfp"})
        assert deleted == 1
        assert store.count() == 2

        remaining_ids = {r.record.id for r in store.search([1.0, 0.0], k=10)}
        assert remaining_ids == {"2", "3"}

    def test_delete_nonexistent_returns_zero(self):
        """Deleting non-existent metadata returns 0."""
        store = InMemoryVectorStore()
        store.add([VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={"doc_id": "doc-1"})])

        deleted = store.delete_by_metadata({"doc_id": "doc-999"})
        assert deleted == 0
        assert store.count() == 1

    def test_delete_empty_filter_returns_zero(self):
        """Empty filter deletes nothing."""
        store = InMemoryVectorStore()
        store.add([VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={"doc_id": "doc-1"})])

        deleted = store.delete_by_metadata({})
        assert deleted == 0
        assert store.count() == 1

    def test_delete_resets_dimension_when_empty(self):
        """Deleting all records resets dimension to None."""
        store = InMemoryVectorStore()
        store.add([VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={"doc_id": "doc-1"})])
        assert store.dimension == 2

        store.delete_by_metadata({"doc_id": "doc-1"})
        assert store.count() == 0
        assert store.dimension is None


class TestChunkTextEdgeCases:
    """Additional edge case tests for chunk_text used in ingestion."""

    def test_chunk_text_empty_string(self):
        assert chunk_text("") == []

    def test_chunk_text_whitespace(self):
        assert chunk_text("  \n\t ") == []

    def test_chunk_text_single_chunk(self):
        chunks = chunk_text("Hello world", chunk_size=100, overlap=20)
        assert chunks == ["Hello world"]

    def test_chunk_text_exact_size(self):
        text = "x" * 100
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert chunks == [text]

    def test_chunk_text_overlap(self):
        text = "0123456789" * 5  # 50 chars
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks) == 3
        # Check overlap
        assert chunks[0][-5:] == chunks[1][:5]
        assert chunks[1][-5:] == chunks[2][:5]