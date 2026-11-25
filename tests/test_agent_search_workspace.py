"""Agent search workspace isolation integration tests.

These tests verify that ``search_company_knowledge`` uses the same
server-resolved workspace as document ingestion, and that knowledge documents
cannot leak across workspaces.
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
from app.core.workspace import resolve_workspace_id
from app.agent_core.builtin_tools import create_search_company_knowledge_tool
from app.modules.agent.service import create_agent_service
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


class FakeEmbedder:
    dimension = 384

    def embed(self, texts):
        return [[0.1] * 384 for _ in texts]


@pytest.fixture
def embedder():
    return FakeEmbedder()


async def _add_doc(store, doc_id: str, text: str, ws_uuid: uuid.UUID):
    """Ingest a single chunk into the store under the given workspace."""
    ws_str = str(ws_uuid)
    await store.async_add(
        [
            VectorRecord(
                id=f"{doc_id}_chunk_0",
                text=text,
                embedding=[0.1] * 384,
                metadata={
                    "doc_id": doc_id,
                    "chunk_index": 0,
                    "page_number": 1,
                    "character_start": 0,
                    "character_end": len(text),
                    "source_filename": f"{doc_id}.txt",
                    "workspace_id": ws_str,
                },
            )
        ]
    )


class TestAgentSearchWorkspaceIsolation:
    async def test_search_uses_same_workspace_as_ingestion(
        self, db_engine, clean_tables, embedder
    ):
        """Document ingestion and agent knowledge search use the same workspace."""
        store_a = PostgreSQLVectorStore(workspace_id=WS_A)
        await _add_doc(store_a, "doc-a", "Secret company policy document", WS_A)

        search_tool = create_search_company_knowledge_tool(
            embedder=embedder, store=store_a
        )
        results = await search_tool(query="company policy", top_k=5)

        assert len(results) == 1
        assert results[0]["text"] == "Secret company policy document"
        assert results[0]["metadata"]["workspace_id"] == str(WS_A)

    async def test_search_does_not_leak_across_workspaces(
        self, db_engine, clean_tables, embedder
    ):
        """Knowledge documents in workspace A are not returned when searching workspace B."""
        store_a = PostgreSQLVectorStore(workspace_id=WS_A)
        store_b = PostgreSQLVectorStore(workspace_id=WS_B)

        await _add_doc(store_a, "doc-a", "Workspace A secret document", WS_A)
        await _add_doc(store_b, "doc-b", "Workspace B secret document", WS_B)

        search_a = create_search_company_knowledge_tool(
            embedder=embedder, store=store_a
        )
        search_b = create_search_company_knowledge_tool(
            embedder=embedder, store=store_b
        )

        results_a = await search_a(query="secret", top_k=5)
        results_b = await search_b(query="secret", top_k=5)

        assert len(results_a) == 1
        assert results_a[0]["metadata"]["workspace_id"] == str(WS_A)

        assert len(results_b) == 1
        assert results_b[0]["metadata"]["workspace_id"] == str(WS_B)

    async def test_agent_service_wires_workspace_scoped_store(
        self, db_engine, clean_tables, embedder
    ):
        """create_agent_service wires the workspace-scoped store to the search tool."""
        from unittest.mock import MagicMock

        store_a = PostgreSQLVectorStore(workspace_id=WS_A)
        await _add_doc(store_a, "doc-a", "Agent searchable document", WS_A)

        mock_provider = MagicMock()
        service = create_agent_service(
            provider=mock_provider, embedder=embedder, store=store_a
        )

        assert service.store is store_a

    async def test_workspace_id_precedence_over_tenant_id(self):
        """WORKSPACE_ID takes precedence over TENANT_ID in resolution."""
        os.environ["WORKSPACE_ID"] = str(WS_A)
        os.environ["TENANT_ID"] = str(WS_B)
        try:
            resolved = resolve_workspace_id()
            assert resolved == WS_A
        finally:
            os.environ.pop("WORKSPACE_ID", None)
            os.environ.pop("TENANT_ID", None)
