"""Focused tests for workspace resolution and isolation.

These tests prove that a client cannot access another workspace's documents
or chunks by supplying a different ``workspace_id``.  Workspace identity is
resolved server-side via :func:`app.core.workspace.resolve_workspace_id` and
never trusted from client input.

Workspace IDs are resolved as :class:`uuid.UUID` values (matching the
``workspace_id`` database column type).  Explicit ``WORKSPACE_ID`` or
``TENANT_ID`` must be valid UUIDs; invalid configuration fails early.
When neither is configured, a fixed deterministic UUID is used.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.workspace import resolve_workspace_id, workspace_id_str
from app.modules.documents.controller import (
    get_document_chunks_controller,
    get_document_controller,
    ingest_document_controller,
    list_documents_controller,
)
from app.modules.documents.schema import ChunkListResponse
from app.rag.store import InMemoryVectorStore

WS_A = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
WS_B = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")


class FakeEmbedder:
    dimension = 8

    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]


# ---------------------------------------------------------------------------
# Workspace resolver
# ---------------------------------------------------------------------------

class TestWorkspaceResolver:
    @contextlib.contextmanager
    def _env(self, **kwargs):
        """Context manager that sets environment variables and restores them."""
        orig = {}
        for key, value in kwargs.items():
            orig[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        try:
            yield
        finally:
            for key, value in orig.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_default_when_no_env(self):
        with self._env(WORKSPACE_ID=None, TENANT_ID=None):
            result = resolve_workspace_id()
        assert isinstance(result, uuid.UUID)
        assert result == uuid.UUID("550e8400-e29b-41d4-a716-446655440000")

    def test_from_workspace_id_env(self):
        with self._env(WORKSPACE_ID="550e8400-e29b-41d4-a716-446655440009"):
            result = resolve_workspace_id()
        assert isinstance(result, uuid.UUID)
        assert result == uuid.UUID("550e8400-e29b-41d4-a716-446655440009")

    def test_from_tenant_id_env_fallback(self):
        with self._env(WORKSPACE_ID=None, TENANT_ID="550e8400-e29b-41d4-a716-446655440003"):
            result = resolve_workspace_id()
        assert isinstance(result, uuid.UUID)
        assert result == uuid.UUID("550e8400-e29b-41d4-a716-446655440003")

    def test_workspace_id_takes_precedence_over_tenant_id(self):
        with self._env(
            WORKSPACE_ID="550e8400-e29b-41d4-a716-446655440009",
            TENANT_ID="550e8400-e29b-41d4-a716-446655440003",
        ):
            result = resolve_workspace_id()
        assert result == uuid.UUID("550e8400-e29b-41d4-a716-446655440009")

    def test_invalid_workspace_id_raises(self):
        with self._env(WORKSPACE_ID="not-a-uuid"):
            with pytest.raises(ValueError, match="must be a valid UUID"):
                resolve_workspace_id()

    def test_invalid_tenant_id_raises(self):
        with self._env(WORKSPACE_ID=None, TENANT_ID="not-a-uuid"):
            with pytest.raises(ValueError, match="must be a valid UUID"):
                resolve_workspace_id()

    def test_workspace_id_str_returns_string(self):
        result = workspace_id_str(WS_A)
        assert result == "550e8400-e29b-41d4-a716-446655440001"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Controller uses server-resolved workspace (isolation)
# ---------------------------------------------------------------------------

def _make_upload_file(filename: str, content: bytes):
    mock = MagicMock()
    mock.filename = filename
    mock.read = AsyncMock(return_value=content)
    mock.close = AsyncMock()
    return mock


class TestControllerWorkspaceIsolation:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_same_doc_id_isolated_across_workspaces(self):
        """Re-ingesting the same doc_id in another workspace must not delete
        the first workspace's chunk."""
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()

        with patch(
            "app.modules.documents.controller._resolve_store_and_embedder",
            return_value=(store, embedder),
        ), patch(
            "app.modules.documents.controller.get_vector_store",
            return_value=store,
        ):
            self._run(
                ingest_document_controller(
                    _make_upload_file("shared.txt", b"ws-a content"),
                    workspace_id=str(WS_A),
                )
            )
            self._run(
                ingest_document_controller(
                    _make_upload_file("shared.txt", b"ws-b content"),
                    workspace_id=str(WS_B),
                )
            )

            chunks_a = asyncio.run(
                get_document_chunks_controller("shared", workspace_id=str(WS_A))
            )
            chunks_b = asyncio.run(
                get_document_chunks_controller("shared", workspace_id=str(WS_B))
            )

        assert len(chunks_a.chunks) == 1
        assert chunks_a.chunks[0].workspace_id == str(WS_A)
        assert len(chunks_b.chunks) == 1
        assert chunks_b.chunks[0].workspace_id == str(WS_B)

    def test_list_respects_workspace(self):
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()

        with patch(
            "app.modules.documents.controller._resolve_store_and_embedder",
            return_value=(store, embedder),
        ), patch(
            "app.modules.documents.controller.get_vector_store",
            return_value=store,
        ):
            self._run(
                ingest_document_controller(
                    _make_upload_file("a.txt", b"ws-a doc"),
                    workspace_id=str(WS_A),
                )
            )
            self._run(
                ingest_document_controller(
                    _make_upload_file("b.txt", b"ws-b doc"),
                    workspace_id=str(WS_B),
                )
            )

            docs_a = asyncio.run(list_documents_controller(workspace_id=str(WS_A)))
            docs_b = asyncio.run(list_documents_controller(workspace_id=str(WS_B)))

        assert [d.doc_id for d in docs_a.documents] == ["a"]
        assert [d.doc_id for d in docs_b.documents] == ["b"]

    def test_get_document_isolated_by_workspace(self):
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()

        with patch(
            "app.modules.documents.controller._resolve_store_and_embedder",
            return_value=(store, embedder),
        ), patch(
            "app.modules.documents.controller.get_vector_store",
            return_value=store,
        ):
            self._run(
                ingest_document_controller(
                    _make_upload_file("only.txt", b"ws-a only"),
                    workspace_id=str(WS_A),
                )
            )

        # ws-b cannot see ws-a's document by doc_id.
        with patch(
            "app.modules.documents.controller.get_vector_store",
            return_value=store,
        ):
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    get_document_controller("only", workspace_id=str(WS_B))
                )
            assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Route ignores client-supplied workspace_id query parameter
# ---------------------------------------------------------------------------

class TestRouteIgnoresClientWorkspaceId:
    def test_route_signature_has_no_workspace_id_param(self):
        from app.modules.documents import route

        import inspect

        for func in (
            route.ingest_document_route,
            route.list_documents_route,
            route.get_document_route,
            route.get_document_chunks_route,
        ):
            params = inspect.signature(func).parameters
            assert "workspace_id" not in params, func.__name__

    def test_client_workspace_id_query_is_ignored_by_server(self):
        """Sending ?workspace_id=evil does not change the workspace used.

        The route resolves workspace_id server-side from the environment, so a
        client-supplied query parameter is not part of the API contract and is
        ignored.
        """
        from fastapi.testclient import TestClient

        import main

        os.environ["WORKSPACE_ID"] = str(WS_A)
        try:
            with patch(
                "app.modules.documents.controller._resolve_store_and_embedder",
                return_value=(MagicMock(), FakeEmbedder()),
            ), patch(
                "app.modules.documents.controller.async_ingest_document_from_text",
                new=AsyncMock(return_value=1),
            ):
                with TestClient(main.app) as client:
                    # Client attempts to select another workspace.
                    r = client.post(
                        "/api/documents/ingest",
                        files={"file": ("notes.txt", b"hello", "text/plain")},
                        params={"workspace_id": "evil"},
                    )
            assert r.status_code == 200, r.json()
            # Response reflects the server-resolved workspace, not "evil".
            assert r.json()["workspace_id"] == str(WS_A)
        finally:
            os.environ.pop("WORKSPACE_ID", None)
