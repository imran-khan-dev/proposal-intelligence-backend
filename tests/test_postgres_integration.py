"""PostgreSQL integration tests for the document ingestion milestone.

These tests are marked ``integration`` and require a running PostgreSQL server
at the DATABASE_URL configured in ``conftest.py`` (the
``proposal_intelligence_test`` database). They are NOT run by the default
``pytest`` invocation; run them explicitly with:

    pytest tests/ -m integration

They verify real database persistence, workspace isolation, and source-reference
resolution through ``PostgreSQLVectorStore``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid

import pytest
from sqlalchemy import text

from app.core.database import (
    get_async_session_factory,
    initialize_database,
    shutdown_database,
)
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.rag.store import VectorRecord
from app.rag.stores.postgres_vector_store import PostgreSQLVectorStore

pytestmark = pytest.mark.integration

TEST_DATABASE_URL = os.environ.get(
    "DATABASE_TEST_URL",
    "postgresql+asyncpg://postgres:45.86.86.86@localhost:5432/proposal_intelligence_test",
)

WS_A = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
WS_B = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")


@pytest.fixture(scope="module")
def event_loop():
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def db_engine():
    """Initialize the test database and run migrations via alembic."""
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
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
    await shutdown_database()


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


def _make_records(doc_id, workspace_id, pages):
    records = []
    ws_str = str(workspace_id)
    for i, (page_number, text) in enumerate(pages):
        records.append(
            VectorRecord(
                id=f"{doc_id}_chunk_{i}",
                text=text,
                embedding=[0.1] * 384,
                metadata={
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "page_number": page_number,
                    "character_start": 0,
                    "character_end": len(text),
                    "source_filename": f"{doc_id}.txt",
                    "workspace_id": ws_str,
                },
            )
        )
    return records


class TestPostgresWorkspaceIsolation:
    async def test_queries_cannot_leak_across_workspaces(
        self, db_engine, clean_tables
    ):
        store_ws_a = PostgreSQLVectorStore(workspace_id=WS_A)
        store_ws_b = PostgreSQLVectorStore(workspace_id=WS_B)

        await store_ws_a.async_add(
            _make_records("doc-a", WS_A, [(1, "workspace a content")])
        )
        await store_ws_b.async_add(
            _make_records("doc-b", WS_B, [(1, "workspace b content")])
        )

        # Search in workspace-a must not return workspace-b's chunk.
        results = await store_ws_a.async_search([0.1] * 384, k=5)
        assert all(r.record.metadata["workspace_id"] == str(WS_A) for r in results)

        # Count is scoped.
        assert await store_ws_a.async_count() == 1
        assert await store_ws_b.async_count() == 1

    async def test_delete_by_doc_id_scoped_to_workspace(
        self, db_engine, clean_tables
    ):
        store = PostgreSQLVectorStore(workspace_id=WS_A)
        await store.async_add(
            _make_records("doc-a", WS_A, [(1, "content")])
        )
        assert await store.async_count() == 1

        # Deleting a doc that belongs to another workspace must be a no-op.
        other = PostgreSQLVectorStore(workspace_id=WS_B)
        deleted = await other.async_delete_by_metadata({"doc_id": "doc-a"})
        assert deleted == 0
        assert await store.async_count() == 1

    async def test_document_and_chunk_persist_with_workspace_id(
        self, db_engine, clean_tables
    ):
        store = PostgreSQLVectorStore(workspace_id=WS_A)
        await store.async_add(
            _make_records("doc-a", WS_A, [(1, "content")])
        )

        factory = get_async_session_factory()
        async with factory() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(Document).where(Document.workspace_id == store._workspace_id)
            )
            documents = result.scalars().all()
            assert len(documents) == 1
            assert documents[0].doc_id == "doc-a"

            result = await session.execute(
                select(DocumentChunk).where(
                    DocumentChunk.workspace_id == store._workspace_id
                )
            )
            chunks = result.scalars().all()
            assert len(chunks) == 1
            assert chunks[0].chunk_metadata["page_number"] == 1

    async def test_list_get_get_chunks_source_references(
        self, db_engine, clean_tables
    ):
        store = PostgreSQLVectorStore(workspace_id=WS_A)
        pages = [
            (1, "Requirement one text here."),
            (2, "Requirement two text here."),
        ]
        await store.async_add(_make_records("doc-x", WS_A, pages))

        docs = await store.async_list_documents(workspace_id=WS_A)
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "doc-x"
        assert docs[0]["chunk_count"] == 2
        assert docs[0]["pages"] == [1, 2]

        doc = await store.async_get_document("doc-x", workspace_id=WS_A)
        assert doc is not None
        assert doc["doc_id"] == "doc-x"

        chunks = await store.async_get_document_chunks(
            "doc-x", workspace_id=WS_A
        )
        assert len(chunks) == 2
        assert chunks[0]["page_number"] == 1
        assert chunks[1]["page_number"] == 2
        assert chunks[0]["character_start"] == 0
        assert chunks[1]["source_filename"] == "doc-x.txt"

        # Another workspace sees nothing.
        other = PostgreSQLVectorStore(workspace_id=WS_B)
        assert await other.async_list_documents() == []
        assert await other.async_get_document("doc-x") is None
        assert await other.async_get_document_chunks("doc-x") == []


class TestPostgresReIngestion:
    async def test_re_ingestion_replaces_chunks_without_duplicates(
        self, db_engine, clean_tables
    ):
        store = PostgreSQLVectorStore(workspace_id=WS_A)
        await store.async_add(
            _make_records("doc-a", WS_A, [(1, "first version")])
        )
        assert await store.async_count() == 1

        await store.async_add(
            _make_records("doc-a", WS_A, [(1, "second version"), (2, "more")])
        )
        assert await store.async_count() == 2

        chunks = await store.async_get_document_chunks("doc-a")
        assert [c["text"] for c in chunks] == ["second version", "more"]
