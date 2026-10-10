"""Tests for vector store."""

import pytest

from app.rag.store import InMemoryVectorStore, VectorRecord, SearchResult


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore."""

    def test_empty_store_count_zero(self):
        store = InMemoryVectorStore()
        assert store.count() == 0

    def test_empty_store_dimension_none(self):
        store = InMemoryVectorStore()
        assert store.dimension is None

    def test_add_single_record_sets_dimension(self):
        store = InMemoryVectorStore()
        record = VectorRecord(
            id="1",
            text="test text",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"},
        )
        store.add([record])
        assert store.count() == 1
        assert store.dimension == 3

    def test_add_multiple_records_same_dimension(self):
        store = InMemoryVectorStore()
        records = [
            VectorRecord(id="1", text="a", embedding=[0.1, 0.2], metadata={}),
            VectorRecord(id="2", text="b", embedding=[0.3, 0.4], metadata={}),
        ]
        store.add(records)
        assert store.count() == 2
        assert store.dimension == 2

    def test_add_empty_list_noop(self):
        store = InMemoryVectorStore()
        store.add([])
        assert store.count() == 0

    def test_add_rejects_empty_embedding(self):
        store = InMemoryVectorStore()
        record = VectorRecord(id="1", text="test", embedding=[], metadata={})
        with pytest.raises(ValueError, match="empty embedding"):
            store.add([record])

    def test_add_rejects_inconsistent_dimensions_in_batch(self):
        store = InMemoryVectorStore()
        records = [
            VectorRecord(id="1", text="a", embedding=[0.1, 0.2], metadata={}),
            VectorRecord(id="2", text="b", embedding=[0.1], metadata={}),  # different dim
        ]
        with pytest.raises(ValueError, match="same embedding dimension"):
            store.add(records)

    def test_add_rejects_dimension_mismatch_with_existing(self):
        store = InMemoryVectorStore()
        store.add([VectorRecord(id="1", text="a", embedding=[0.1, 0.2], metadata={})])
        assert store.dimension == 2

        # Try to add 3-dimensional vector
        record = VectorRecord(id="2", text="b", embedding=[0.1, 0.2, 0.3], metadata={})
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add([record])

    def test_search_empty_store_returns_empty(self):
        store = InMemoryVectorStore()
        results = store.search([0.1, 0.2, 0.3], k=5)
        assert results == []

    def test_search_k_zero_returns_empty(self):
        store = InMemoryVectorStore()
        store.add([VectorRecord(id="1", text="a", embedding=[0.1, 0.2], metadata={})])
        results = store.search([0.1, 0.2], k=0)
        assert results == []

    def test_search_negative_k_returns_empty(self):
        store = InMemoryVectorStore()
        store.add([VectorRecord(id="1", text="a", embedding=[0.1, 0.2], metadata={})])
        results = store.search([0.1, 0.2], k=-1)
        assert results == []

    def test_search_validates_query_dimension(self):
        store = InMemoryVectorStore()
        store.add([VectorRecord(id="1", text="a", embedding=[0.1, 0.2], metadata={})])
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.search([0.1, 0.2, 0.3])  # 3D query, 2D store

    def test_search_returns_top_k_by_cosine_similarity(self):
        store = InMemoryVectorStore()
        # Add orthogonal-ish vectors
        store.add([
            VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={}),
            VectorRecord(id="2", text="b", embedding=[0.0, 1.0], metadata={}),
            VectorRecord(id="3", text="c", embedding=[0.707, 0.707], metadata={}),  # ~45 deg
        ])

        # Query matches record 1 exactly
        results = store.search([1.0, 0.0], k=3)
        assert len(results) == 3
        assert results[0].record.id == "1"
        assert results[0].score == 1.0  # Perfect match
        assert results[1].record.id == "3"
        assert results[2].record.id == "2"
        assert results[2].score == 0.0  # Orthogonal

    def test_search_respects_k_limit(self):
        store = InMemoryVectorStore()
        for i in range(10):
            # Use non-zero embeddings to avoid zero-norm skipping
            store.add([VectorRecord(id=str(i), text=f"t{i}", embedding=[float(i) + 1.0, 0.0], metadata={})])

        results = store.search([6.0, 0.0], k=3)
        assert len(results) == 3

        results = store.search([6.0, 0.0], k=100)
        assert len(results) == 10  # Only 10 records exist

    def test_search_returns_search_result_with_score(self):
        store = InMemoryVectorStore()
        store.add([VectorRecord(id="1", text="test", embedding=[1.0, 0.0], metadata={"key": "val"})])
        results = store.search([1.0, 0.0], k=1)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].record.id == "1"
        assert results[0].record.text == "test"
        assert results[0].record.metadata == {"key": "val"}
        assert results[0].score == 1.0

    def test_search_zero_norm_query_returns_empty(self):
        store = InMemoryVectorStore()
        store.add([VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={})])
        results = store.search([0.0, 0.0], k=5)
        assert results == []

    def test_search_zero_norm_records_skipped(self):
        store = InMemoryVectorStore()
        store.add([
            VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={}),
            VectorRecord(id="2", text="b", embedding=[0.0, 0.0], metadata={}),  # zero norm
        ])
        results = store.search([1.0, 0.0], k=5)
        assert len(results) == 1
        assert results[0].record.id == "1"

    def test_clear_removes_all_records(self):
        store = InMemoryVectorStore()
        store.add([VectorRecord(id="1", text="a", embedding=[0.1, 0.2], metadata={})])
        assert store.count() == 1
        store.clear()
        assert store.count() == 0
        assert store.dimension is None

    def test_search_only_positive_similarity(self):
        """Search should only return results with positive cosine similarity."""
        store = InMemoryVectorStore()
        # Add vectors pointing in opposite directions
        store.add([
            VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={}),
            VectorRecord(id="2", text="b", embedding=[-1.0, 0.0], metadata={}),
        ])

        # Query in direction of record 1
        results = store.search([1.0, 0.0], k=5)
        # Should only return record 1 (score 1.0), not record 2 (score -1.0)
        assert len(results) == 1
        assert results[0].record.id == "1"
        assert results[0].score == 1.0


class TestVectorRecord:
    """Tests for VectorRecord dataclass."""

    def test_vector_record_creation(self):
        record = VectorRecord(
            id="test-1",
            text="sample text",
            embedding=[0.1, 0.2, 0.3],
            metadata={"page": 1, "source": "doc.pdf"},
        )
        assert record.id == "test-1"
        assert record.text == "sample text"
        assert record.embedding == [0.1, 0.2, 0.3]
        assert record.metadata == {"page": 1, "source": "doc.pdf"}

    def test_vector_record_empty_metadata(self):
        record = VectorRecord(id="1", text="t", embedding=[0.1], metadata={})
        assert record.metadata == {}


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        record = VectorRecord(id="1", text="t", embedding=[0.1], metadata={})
        result = SearchResult(record=record, score=0.95)
        assert result.record == record
        assert result.score == 0.95