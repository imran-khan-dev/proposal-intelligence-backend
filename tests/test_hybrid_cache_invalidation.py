"""Regression tests for hybrid retrieval cache invalidation.

These tests verify that BM25 indexes are properly invalidated after
document ingestion, deletion, and updates.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid

import pytest
from sqlalchemy import text

from app.ai.embeddings import EmbeddingProvider
from app.core.database import (
    get_async_session_factory,
    initialize_database,
    shutdown_database,
)
from app.rag.hybrid_retrieval import HybridRetrievalService, get_or_create_hybrid_service, invalidate_hybrid_cache, _service_cache
from app.rag.ingest import async_ingest_document
from app.rag.retrieve import async_retrieve
from app.rag.stores.postgres_vector_store import PostgreSQLVectorStore

pytestmark = pytest.mark.integration

TEST_DATABASE_URL = os.environ.get(
    "DATABASE_TEST_URL",
    "postgresql+asyncpg://postgres:45.86.86.86@localhost:5432/proposal_intelligence_test",
)

WS_A = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
WS_B = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")


@pytest.fixture(scope="session")
def db_engine():
    """Initialize the test database and run migrations via alembic."""
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    os.environ["DATABASE_USE_NULL_POOL"] = "true"
    initialize_database()

    # Apply migrations head using the project's alembic config.
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"alembic upgrade head failed:\n{result.stderr}\n{result.stdout}"
        )
    yield
    # Don't call shutdown_database() here - let process exit handle cleanup
    # shutdown_database() closes the event loop which breaks subsequent tests


@pytest.fixture
async def clean_tables(db_engine):
    """Truncate document tables before each test for isolation."""
    factory = get_async_session_factory()
    async with factory() as session:
        await session.execute(text("DELETE FROM document_chunks"))
        await session.execute(text("DELETE FROM documents"))
        await session.execute(text("DELETE FROM tenants"))
        await session.commit()
    yield
    async with factory() as session:
        await session.execute(text("DELETE FROM document_chunks"))
        await session.execute(text("DELETE FROM documents"))
        await session.execute(text("DELETE FROM tenants"))
        await session.commit()


class FakeEmbedder(EmbeddingProvider):
    """Deterministic embedding provider for tests."""

    dimension = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for t in texts:
            h = hash(t)
            vec = [(h >> (i * 8)) & 0xFF for i in range(384)]
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


async def _add_doc(store: PostgreSQLVectorStore, doc_id: str, text: str, ws_uuid: uuid.UUID, source_type: str = "company_knowledge"):
    """Helper to ingest a document."""
    embedder = FakeEmbedder()
    await async_ingest_document(
        text=text,
        doc_id=doc_id,
        store=store,
        embedder=embedder,
        workspace_id=str(ws_uuid),
        source_type=source_type,
    )


class TestHybridCacheInvalidation:
    """Tests for BM25 cache invalidation after document changes."""

    async def test_ingest_invalidates_cache(self, db_engine, clean_tables):
        """After ingesting a new document, hybrid search finds it without manual cache invalidation."""
        store_a = PostgreSQLVectorStore(workspace_id=WS_A)
        embedder = FakeEmbedder()

        # Initial search - empty index
        results = await async_retrieve(
            query="policy",
            embedder=embedder,
            store=store_a,
            top_k=5,
            source_type="company_knowledge",
            use_hybrid=True,
        )
        assert len(results) == 0

        # Ingest a new company-knowledge document
        await _add_doc(store_a, "doc-1", "Company policy document about refunds", WS_A)

        # Search again - should find the new document without manual invalidation
        results = await async_retrieve(
            query="policy",
            embedder=embedder,
            store=store_a,
            top_k=5,
            source_type="company_knowledge",
            use_hybrid=True,
        )
        assert len(results) == 1
        assert "refunds" in results[0].record.text

    async def test_ingest_invalidates_only_affected_workspace(self, db_engine, clean_tables):
        """Ingesting into workspace A does not affect workspace B's cache."""
        store_a = PostgreSQLVectorStore(workspace_id=WS_A)
        store_b = PostgreSQLVectorStore(workspace_id=WS_B)
        embedder = FakeEmbedder()

        # Add initial docs to both workspaces
        await _add_doc(store_a, "doc-a", "Workspace A policy document", WS_A)
        await _add_doc(store_b, "doc-b", "Workspace B policy document", WS_B)

        # Search both - both should have results
        results_a = await async_retrieve(
            query="policy",
            embedder=embedder,
            store=store_a,
            top_k=5,
            source_type="company_knowledge",
            use_hybrid=True,
        )
        results_b = await async_retrieve(
            query="policy",
            embedder=embedder,
            store=store_b,
            top_k=5,
            source_type="company_knowledge",
            use_hybrid=True,
        )
        assert len(results_a) == 1
        assert len(results_b) == 1

        # Ingest new doc into workspace A only
        await _add_doc(store_a, "doc-a-2", "Another policy for workspace A", WS_A)

        # Search again - workspace A should have 2 results, workspace B still 1
        results_a = await async_retrieve(
            query="policy",
            embedder=embedder,
            store=store_a,
            top_k=5,
            source_type="company_knowledge",
            use_hybrid=True,
        )
        results_b = await async_retrieve(
            query="policy",
            embedder=embedder,
            store=store_b,
            top_k=5,
            source_type="company_knowledge",
            use_hybrid=True,
        )
        assert len(results_a) == 2
        assert len(results_b) == 1
        assert results_b[0].record.metadata.get("workspace_id") == str(WS_B)

    async def test_ingest_invalidates_only_affected_source_type(self, db_engine, clean_tables):
        """Ingesting company_knowledge does not affect RFP source type cache."""
        store = PostgreSQLVectorStore(workspace_id=WS_A)
        embedder = FakeEmbedder()

        # Add both company_knowledge and RFP docs
        await _add_doc(store, "comp-1", "Company knowledge about refunds", WS_A, source_type="company_knowledge")
        await _add_doc(store, "rfp-1", "RFP requirement for refunds", WS_A, source_type="rfp")

        # Search company_knowledge
        results_ck = await async_retrieve(
            query="refunds",
            embedder=embedder,
            store=store,
            top_k=5,
            source_type="company_knowledge",
            use_hybrid=True,
        )
        # Search RFP
        results_rfp = await async_retrieve(
            query="refunds",
            embedder=embedder,
            store=store,
            top_k=5,
            source_type="rfp",
            use_hybrid=True,
        )
        assert len(results_ck) == 1
        assert len(results_rfp) == 1

        # Add new company_knowledge doc
        await _add_doc(store, "comp-2", "Company knowledge about exchanges", WS_A, source_type="company_knowledge")

        # Search for "exchanges" - should find the new company_knowledge doc
        # Hybrid search may return multiple results (BM25 + dense fusion), so verify
        # the new document is present rather than asserting exact count
        results_ck = await async_retrieve(
            query="exchanges",
            embedder=embedder,
            store=store,
            top_k=5,
            source_type="company_knowledge",
            use_hybrid=True,
        )
        assert len(results_ck) >= 1
        assert any("exchanges" in r.record.text for r in results_ck)

        # RFP search for "refunds" should still return the RFP doc (cache not invalidated)
        results_rfp = await async_retrieve(
            query="refunds",
            embedder=embedder,
            store=store,
            top_k=5,
            source_type="rfp",
            use_hybrid=True,
        )
        assert len(results_rfp) == 1
        assert "refunds" in results_rfp[0].record.text

    async def test_empty_ingest_deletes_and_invalidates(self, db_engine, clean_tables):
        """Ingesting empty text deletes existing doc and invalidates cache."""
        store = PostgreSQLVectorStore(workspace_id=WS_A)
        embedder = FakeEmbedder()

        # Add a document
        await _add_doc(store, "doc-1", "Policy document about refunds", WS_A)

        # Search - should find it
        results = await async_retrieve(
            query="refunds",
            embedder=embedder,
            store=store,
            top_k=5,
            source_type="company_knowledge",
            use_hybrid=True,
        )
        assert len(results) == 1

        # Ingest empty text for same doc_id (deletes the document)
        count = await async_ingest_document(
            text="",
            doc_id="doc-1",
            store=store,
            embedder=embedder,
            workspace_id=str(WS_A),
            source_type="company_knowledge",
        )
        assert count == 0

        # Search again - should find nothing
        results = await async_retrieve(
            query="refunds",
            embedder=embedder,
            store=store,
            top_k=5,
            source_type="company_knowledge",
            use_hybrid=True,
        )
        assert len(results) == 0

    async def test_rfp_docs_not_in_company_knowledge_results(self, db_engine, clean_tables):
        """RFP source passages are never returned in company_knowledge searches."""
        store = PostgreSQLVectorStore(workspace_id=WS_A)
        embedder = FakeEmbedder()

        # Add RFP document (source_type="rfp")
        await _add_doc(store, "rfp-1", "The system must support SAML 2.0 SSO", WS_A, source_type="rfp")
        # Add company_knowledge document
        await _add_doc(store, "comp-1", "Our platform supports SAML 2.0 SSO for enterprise", WS_A, source_type="company_knowledge")

        # Search company_knowledge - should only get company_knowledge result
        results = await async_retrieve(
            query="SAML",
            embedder=embedder,
            store=store,
            top_k=5,
            source_type="company_knowledge",
            use_hybrid=True,
        )
        assert len(results) == 1
        assert results[0].record.metadata.get("source_type") == "company_knowledge"
        assert results[0].record.metadata.get("doc_id") == "comp-1"

        # Search without source_type filter - should get both
        results = await async_retrieve(
            query="SAML",
            embedder=embedder,
            store=store,
            top_k=5,
            source_type=None,
            use_hybrid=True,
        )
        assert len(results) == 2
        source_types = {r.record.metadata.get("source_type") for r in results}
        assert source_types == {"company_knowledge", "rfp"}

    async def test_failed_ingestion_does_not_corrupt_cache(self, db_engine, clean_tables):
        """Failed ingestion (e.g., dimension mismatch) does not affect existing cache."""
        store = PostgreSQLVectorStore(workspace_id=WS_A)
        embedder = FakeEmbedder()

        # Add initial document
        await _add_doc(store, "doc-1", "Valid policy document", WS_A)

        # Search - works
        results = await async_retrieve(
            query="policy",
            embedder=embedder,
            store=store,
            top_k=5,
            source_type="company_knowledge",
            use_hybrid=True,
        )
        assert len(results) == 1

        # Try to ingest with a different embedder that produces wrong dimensions
        class BadEmbedder(EmbeddingProvider):
            dimension = 128  # Wrong dimension

            def embed(self, texts):
                return [[0.1] * 128 for _ in texts]

        bad_embedder = BadEmbedder()

        # This should fail
        with pytest.raises(ValueError, match="dimension mismatch"):
            await async_ingest_document(
                text="Another policy",
                doc_id="doc-2",
                store=store,
                embedder=bad_embedder,
                workspace_id=str(WS_A),
                source_type="company_knowledge",
            )

        # Original search should still work (cache not corrupted)
        results = await async_retrieve(
            query="policy",
            embedder=embedder,
            store=store,
            top_k=5,
            source_type="company_knowledge",
            use_hybrid=True,
        )
        assert len(results) == 1
        assert "Valid policy document" in results[0].record.text

    async def test_cache_invalidation_function_directly(self, db_engine, clean_tables):
        """Test the invalidate_hybrid_cache function directly."""
        store = PostgreSQLVectorStore(workspace_id=WS_A)
        embedder = FakeEmbedder()

        # Create a service and populate cache
        service = get_or_create_hybrid_service(embedder, store, str(WS_A), "company_knowledge", False)
        assert service is not None

        # Verify cache has entry - check internal cache
        # Cache key format: (store_id, workspace_id, source_type, enable_reranking)
        store_id = id(store)
        cache_key = (store_id, str(WS_A), "company_knowledge", False)
        assert cache_key in _service_cache

        # Invalidate
        invalidate_hybrid_cache(workspace_id=str(WS_A), source_type="company_knowledge")

        # Cache entry should be removed
        assert cache_key not in _service_cache

        # Next search should rebuild index (verified by other tests above)
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])