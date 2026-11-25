"""Focused tests for workspace resolution and isolation.

These tests prove that a client cannot access another workspace's documents
or chunks by supplying a different ``workspace_id``.  Workspace identity is
resolved server-side via :func:`app.core.workspace.resolve_workspace_id` and
never trusted from client input.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.workspace import resolve_workspace_id
from app.modules.documents.controller import (
    get_document_chunks_controller,
    get_document_controller,
    ingest_document_controller,
    list_documents_controller,
)
from app.modules.documents.schema import ChunkListResponse
from app.rag.store import InMemoryVectorStore


class FakeEmbedder:
    dimension = 8

    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]


# ---------------------------------------------------------------------------
# Workspace resolver
# ---------------------------------------------------------------------------

class TestWorkspaceResolver:
    def test_default_when_no_env(self):
        for key in ("WORKSPACE_ID", "TENANT_ID"):
            os.environ.pop(key, None)
        assert resolve_workspace_id() == "default-workspace"

    def test_from_workspace_id_env(self):
        os.environ["WORKSPACE_ID"] = "ws-custom"
        try:
            assert resolve_workspace_id() == "ws-custom"
        finally:
            os.environ.pop("WORKSPACE_ID", None)

    def test_from_tenant_id_env_fallback(self):
        os.environ.pop("WORKSPACE_ID", None)
        os.environ["TENANT_ID"] = "legacy-tenant"
        try:
            assert resolve_workspace_id() == "legacy-tenant"
        finally:
            os.environ.pop("TENANT_ID", None)

    def test_workspace_id_takes_precedence_over_tenant_id(self):
        os.environ["WORKSPACE_ID"] = "ws-primary"
        os.environ["TENANT_ID"] = "legacy-tenant"
        try:
            assert resolve_workspace_id() == "ws-primary"
        finally:
            os.environ.pop("WORKSPACE_ID", None)
            os.environ.pop("TENANT_ID", None)


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
        the first workspace's chunk (empty-text and normal paths)."""
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
                    workspace_id="ws-a",
                )
            )
            self._run(
                ingest_document_controller(
                    _make_upload_file("shared.txt", b"ws-b content"),
                    workspace_id="ws-b",
                )
            )

            chunks_a = asyncio.run(
                get_document_chunks_controller("shared", workspace_id="ws-a")
            )
            chunks_b = asyncio.run(
                get_document_chunks_controller("shared", workspace_id="ws-b")
            )

        assert len(chunks_a.chunks) == 1
        assert chunks_a.chunks[0].workspace_id == "ws-a"
        assert len(chunks_b.chunks) == 1
        assert chunks_b.chunks[0].workspace_id == "ws-b"

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
                    workspace_id="ws-a",
                )
            )
            self._run(
                ingest_document_controller(
                    _make_upload_file("b.txt", b"ws-b doc"),
                    workspace_id="ws-b",
                )
            )

            docs_a = asyncio.run(list_documents_controller(workspace_id="ws-a"))
            docs_b = asyncio.run(list_documents_controller(workspace_id="ws-b"))

        assert [d.doc_id for d in docs_a.documents] == ["a"]
        assert [d.doc_id for d in docs_b.documents] == ["b"]

    def test_get_document_isolated_by_workspace(self):
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()

        with patch(
            "app.modules.documents.controller._resolve_store_and_embedder",
            return_value=(store, embedder),
        ):
            self._run(
                ingest_document_controller(
                    _make_upload_file("only.txt", b"ws-a only"),
                    workspace_id="ws-a",
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
                    get_document_controller("only", workspace_id="ws-b")
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

        os.environ["WORKSPACE_ID"] = "server-ws"
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
            assert r.json()["workspace_id"] == "server-ws"
        finally:
            os.environ.pop("WORKSPACE_ID", None)
