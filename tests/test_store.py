"""Tests for vector store."""

import pytest
import tempfile
import os
import json
from unittest.mock import patch

from app.rag.store import InMemoryVectorStore, PersistentVectorStore, VectorRecord, SearchResult
from app.rag.store_factory import get_vector_store


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


class TestPersistentVectorStore:
    """Tests for PersistentVectorStore."""

    def _make_store(self, path: str = None) -> PersistentVectorStore:
        """Create a PersistentVectorStore with a temporary file."""
        if path is None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
            path = tmp.name
            tmp.close()
        return PersistentVectorStore(path)

    def _cleanup(self, path: str) -> None:
        """Clean up a temporary file."""
        try:
            os.unlink(path)
        except OSError:
            pass

    def test_absent_file_creates_empty_store(self):
        """Non-existent file initializes an empty store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nonexistent.jsonl")
            store = PersistentVectorStore(path)
            assert store.count() == 0
            assert store.dimension is None

    def test_empty_file_creates_empty_store(self):
        """Empty file initializes an empty store."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name
        try:
            store = PersistentVectorStore(path)
            assert store.count() == 0
            assert store.dimension is None
        finally:
            self._cleanup(path)

    def test_add_and_reload_preserves_records(self):
        """Records added are persisted and reloaded."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            # First store instance - add records
            store1 = PersistentVectorStore(path)
            records = [
                VectorRecord(id="1", text="first", embedding=[1.0, 0.0], metadata={"doc": "a"}),
                VectorRecord(id="2", text="second", embedding=[0.0, 1.0], metadata={"doc": "b"}),
            ]
            store1.add(records)
            assert store1.count() == 2
            assert store1.dimension == 2

            # Second store instance - should load persisted records
            store2 = PersistentVectorStore(path)
            assert store2.count() == 2
            assert store2.dimension == 2

            # Verify content
            results = store2.search([1.0, 0.0], k=2)
            assert len(results) == 2
            assert results[0].record.id == "1"
            assert results[0].record.text == "first"
            assert results[0].record.metadata == {"doc": "a"}
            assert results[0].score == 1.0

            assert results[1].record.id == "2"
            assert results[1].record.text == "second"
            assert results[1].record.metadata == {"doc": "b"}

        finally:
            self._cleanup(path)

    def test_search_after_reload_works(self):
        """Similarity search works correctly after reload."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            store1 = PersistentVectorStore(path)
            store1.add([
                VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={}),
                VectorRecord(id="2", text="b", embedding=[0.0, 1.0], metadata={}),
                VectorRecord(id="3", text="c", embedding=[0.707, 0.707], metadata={}),
            ])

            # Reload and search
            store2 = PersistentVectorStore(path)
            results = store2.search([1.0, 0.0], k=3)

            assert len(results) == 3
            assert results[0].record.id == "1"
            assert results[0].score == 1.0
            assert results[1].record.id == "3"
            assert results[2].record.id == "2"
            assert results[2].score == 0.0

        finally:
            self._cleanup(path)

    def test_multiple_records_metadata_preserved(self):
        """Multiple records with various metadata are preserved."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            store1 = PersistentVectorStore(path)
            records = [
                VectorRecord(id="1", text="t1", embedding=[0.1, 0.2], metadata={"page": 1, "source": "doc.pdf"}),
                VectorRecord(id="2", text="t2", embedding=[0.3, 0.4], metadata={"page": 2, "source": "doc.pdf", "tags": ["a", "b"]}),
                VectorRecord(id="3", text="t3", embedding=[0.5, 0.6], metadata={}),
            ]
            store1.add(records)

            store2 = PersistentVectorStore(path)
            assert store2.count() == 3

            results = store2.search([0.1, 0.2], k=3)
            assert len(results) == 3

            # Check all metadata preserved
            for r in results:
                if r.record.id == "1":
                    assert r.record.metadata == {"page": 1, "source": "doc.pdf"}
                elif r.record.id == "2":
                    assert r.record.metadata == {"page": 2, "source": "doc.pdf", "tags": ["a", "b"]}
                elif r.record.id == "3":
                    assert r.record.metadata == {}

        finally:
            self._cleanup(path)

    def test_add_persists(self):
        """Add operation persists to disk."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            store = PersistentVectorStore(path)
            store.add([VectorRecord(id="1", text="test", embedding=[1.0, 0.0], metadata={"k": "v"})])

            # Verify file was written
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            # Reload and verify
            store2 = PersistentVectorStore(path)
            assert store2.count() == 1

        finally:
            self._cleanup(path)

    def test_delete_by_metadata_persists(self):
        """Delete by metadata persists to disk."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            store1 = PersistentVectorStore(path)
            store1.add([
                VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={"doc_id": "doc-1"}),
                VectorRecord(id="2", text="b", embedding=[0.0, 1.0], metadata={"doc_id": "doc-2"}),
            ])
            assert store1.count() == 2

            deleted = store1.delete_by_metadata({"doc_id": "doc-1"})
            assert deleted == 1
            assert store1.count() == 1

            # Reload and verify
            store2 = PersistentVectorStore(path)
            assert store2.count() == 1
            results = store2.search([1.0, 0.0], k=5)
            assert len(results) == 1
            assert results[0].record.metadata["doc_id"] == "doc-2"

        finally:
            self._cleanup(path)

    def test_clear_persists(self):
        """Clear operation persists to disk."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            store1 = PersistentVectorStore(path)
            store1.add([VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={})])
            assert store1.count() == 1

            store1.clear()
            assert store1.count() == 0
            assert store1.dimension is None

            # Reload and verify
            store2 = PersistentVectorStore(path)
            assert store2.count() == 0
            assert store2.dimension is None

        finally:
            self._cleanup(path)

    def test_dimension_validation_on_add(self):
        """Adding records with wrong dimension raises and doesn't persist."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            store = PersistentVectorStore(path)
            store.add([VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={})])
            assert store.dimension == 2

            # Try to add 3D record - should raise
            with pytest.raises(ValueError, match="dimension mismatch"):
                store.add([VectorRecord(id="2", text="b", embedding=[1.0, 0.0, 0.0], metadata={})])

            # Original record should still be there
            assert store.count() == 1
            assert store.dimension == 2

            # Reload and verify original persisted
            store2 = PersistentVectorStore(path)
            assert store2.count() == 1
            assert store2.dimension == 2

        finally:
            self._cleanup(path)

    def test_dimension_validation_on_load(self):
        """Loading records with inconsistent dimensions raises."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp:
            path = tmp.name
            # Write two records with different dimensions
            tmp.write('{"id": "1", "text": "a", "embedding": [1.0, 0.0], "metadata": {}}\n')
            tmp.write('{"id": "2", "text": "b", "embedding": [1.0, 0.0, 0.0], "metadata": {}}\n')

        try:
            with pytest.raises(ValueError, match="Dimension mismatch"):
                PersistentVectorStore(path)
        finally:
            self._cleanup(path)

    def test_malformed_json_raises(self):
        """Malformed JSON in file raises clear error."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp:
            path = tmp.name
            tmp.write('{"id": "1", "text": "a", "embedding": [1.0, 0.0], "metadata": {}}\n')
            tmp.write('not valid json\n')

        try:
            with pytest.raises(ValueError, match="Malformed JSON at line 2"):
                PersistentVectorStore(path)
        finally:
            self._cleanup(path)

    def test_missing_field_raises(self):
        """Missing required field raises clear error."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp:
            path = tmp.name
            tmp.write('{"id": "1", "text": "a", "embedding": [1.0, 0.0]}\n')  # missing metadata

        try:
            with pytest.raises(ValueError, match="Missing required field 'metadata'"):
                PersistentVectorStore(path)
        finally:
            self._cleanup(path)

    def test_empty_embedding_raises(self):
        """Record with empty embedding raises on load."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp:
            path = tmp.name
            tmp.write('{"id": "1", "text": "a", "embedding": [], "metadata": {}}\n')

        try:
            with pytest.raises(ValueError, match="empty or invalid embedding"):
                PersistentVectorStore(path)
        finally:
            self._cleanup(path)

    def test_non_numeric_embedding_raises(self):
        """Record with non-numeric embedding values raises on load."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp:
            path = tmp.name
            tmp.write('{"id": "1", "text": "a", "embedding": [1.0, "x"], "metadata": {}}\n')

        try:
            with pytest.raises(ValueError, match="non-numeric embedding values"):
                PersistentVectorStore(path)
        finally:
            self._cleanup(path)

    def test_invalid_metadata_type_raises(self):
        """Record with non-object metadata raises on load."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tmp:
            path = tmp.name
            tmp.write('{"id": "1", "text": "a", "embedding": [1.0, 0.0], "metadata": "not an object"}\n')

        try:
            with pytest.raises(ValueError, match="invalid metadata"):
                PersistentVectorStore(path)
        finally:
            self._cleanup(path)

    def test_non_serializable_metadata_raises_on_add(self):
        """Adding record with non-JSON-serializable metadata raises."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            store = PersistentVectorStore(path)
            # set is not JSON serializable
            record = VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={"tags": set([1, 2])})

            with pytest.raises(ValueError, match="not JSON serializable"):
                store.add([record])

            # Store should be empty
            assert store.count() == 0

        finally:
            self._cleanup(path)

    def test_failed_save_rolls_back_add(self):
        """Failed save during add rolls back in-memory state."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            store = PersistentVectorStore(path)
            store.add([VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={})])
            assert store.count() == 1

            # Make the file read-only to force save failure
            os.chmod(path, 0o444)

            try:
                with pytest.raises(Exception):
                    store.add([VectorRecord(id="2", text="b", embedding=[0.0, 1.0], metadata={})])
            finally:
                # Restore permissions for cleanup
                os.chmod(path, 0o644)

            # Original record should still be there (rollback worked)
            assert store.count() == 1
            results = store.search([1.0, 0.0], k=5)
            assert len(results) == 1
            assert results[0].record.id == "1"

        finally:
            self._cleanup(path)

    def test_failed_save_rolls_back_delete(self):
        """Failed save during delete rolls back in-memory state."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            store = PersistentVectorStore(path)
            store.add([
                VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={"doc_id": "doc-1"}),
                VectorRecord(id="2", text="b", embedding=[0.0, 1.0], metadata={"doc_id": "doc-2"}),
            ])
            assert store.count() == 2

            # Make the file read-only to force save failure
            os.chmod(path, 0o444)

            try:
                with pytest.raises(Exception):
                    store.delete_by_metadata({"doc_id": "doc-1"})
            finally:
                os.chmod(path, 0o644)

            # Both records should still be there (rollback worked)
            assert store.count() == 2

        finally:
            self._cleanup(path)

    def test_failed_save_rolls_back_clear(self):
        """Failed save during clear rolls back in-memory state."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            store = PersistentVectorStore(path)
            store.add([VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={})])
            assert store.count() == 1

            # Make the file read-only to force save failure
            os.chmod(path, 0o444)

            try:
                with pytest.raises(Exception):
                    store.clear()
            finally:
                os.chmod(path, 0o644)

            # Record should still be there (rollback worked)
            assert store.count() == 1

        finally:
            self._cleanup(path)

    def test_empty_file_behavior_consistent(self):
        """Empty file behavior is consistent with absent file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            store1 = PersistentVectorStore(path)
            assert store1.count() == 0
            assert store1.dimension is None

            # Add a record
            store1.add([VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={})])
            assert store1.count() == 1

            # Truncate file manually
            with open(path, "w") as f:
                pass  # Empty file

            # Reload should give empty store
            store2 = PersistentVectorStore(path)
            assert store2.count() == 0
            assert store2.dimension is None

        finally:
            self._cleanup(path)


class TestVectorStoreFactory:
    """Tests for get_vector_store factory function."""

    def test_default_type_is_memory(self):
        """Default VECTOR_STORE_TYPE is memory."""
        with patch.dict(os.environ, {}, clear=True):
            store = get_vector_store()
            assert isinstance(store, InMemoryVectorStore)

    def test_explicit_memory_type(self):
        """Explicit memory type returns InMemoryVectorStore."""
        with patch.dict(os.environ, {"VECTOR_STORE_TYPE": "memory"}, clear=True):
            store = get_vector_store()
            assert isinstance(store, InMemoryVectorStore)

    def test_persistent_type_returns_persistent_store(self):
        """Persistent type returns PersistentVectorStore with default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"VECTOR_STORE_TYPE": "persistent", "VECTOR_STORE_PATH": os.path.join(tmpdir, "test.jsonl")}, clear=True):
                store = get_vector_store()
                assert isinstance(store, PersistentVectorStore)
                assert store.count() == 0

    def test_custom_persistent_path(self):
        """Custom VECTOR_STORE_PATH is honored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom", "vector_store.jsonl")
            with patch.dict(os.environ, {"VECTOR_STORE_TYPE": "persistent", "VECTOR_STORE_PATH": custom_path}, clear=True):
                store = get_vector_store()
                assert isinstance(store, PersistentVectorStore)
                # Verify the store can be used
                store.add([VectorRecord(id="1", text="test", embedding=[0.1, 0.2], metadata={})])
                assert store.count() == 1

    def test_case_insensitive_store_type(self):
        """Store type is case-insensitive."""
        with patch.dict(os.environ, {"VECTOR_STORE_TYPE": "MEMORY"}, clear=True):
            store = get_vector_store()
            assert isinstance(store, InMemoryVectorStore)

        with patch.dict(os.environ, {"VECTOR_STORE_TYPE": " Persistent "}, clear=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.jsonl")
                with patch.dict(os.environ, {"VECTOR_STORE_TYPE": " Persistent ", "VECTOR_STORE_PATH": path}, clear=True):
                    store = get_vector_store()
                    assert isinstance(store, PersistentVectorStore)

    def test_unsupported_store_type_raises(self):
        """Unsupported store type raises clear error."""
        with patch.dict(os.environ, {"VECTOR_STORE_TYPE": "invalid_type"}, clear=True):
            with pytest.raises(RuntimeError, match="Invalid VECTOR_STORE_TYPE: 'invalid_type'"):
                get_vector_store()

    def test_explicit_store_injection_overrides_env(self):
        """Explicit store argument in create_agent_service overrides env config."""
        from app.rag.store import InMemoryVectorStore
        from app.agent.service import create_agent_service
        from unittest.mock import MagicMock
        from app.ai.provider import AIProvider

        # Even with persistent in env, explicit store should be used
        with patch.dict(os.environ, {"VECTOR_STORE_TYPE": "persistent"}, clear=True):
            mock_provider = MagicMock(spec=AIProvider)
            explicit_store = InMemoryVectorStore()
            service = create_agent_service(provider=mock_provider, store=explicit_store)

            # The service should use the explicitly provided store
            assert service.store is explicit_store
            assert isinstance(service.store, InMemoryVectorStore)

    def test_factory_creates_parent_directory(self):
        """Factory creates parent directory for persistent store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "nested", "dir", "vector_store.jsonl")
            with patch.dict(os.environ, {"VECTOR_STORE_TYPE": "persistent", "VECTOR_STORE_PATH": custom_path}, clear=True):
                # Should not raise even though parent dirs don't exist
                store = get_vector_store()
                assert isinstance(store, PersistentVectorStore)
                # Verify parent dir was created
                assert os.path.exists(os.path.dirname(custom_path))