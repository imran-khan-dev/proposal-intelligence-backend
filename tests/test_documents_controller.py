"""Unit tests for the document ingestion controller.

The service layer and store are mocked so no database, network, or API key is
required. Covers format validation, PDF/TXT/MD ingestion, empty/invalid
documents, and workspace scoping at the controller boundary.
"""

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile

TEST_WS = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
TEST_WS_STR = str(TEST_WS)
TEST_WS_B = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")
TEST_WS_B_STR = str(TEST_WS_B)

from app.modules.documents.controller import (
    get_document_chunks_controller,
    get_document_controller,
    ingest_document_controller,
    list_documents_controller,
)


def _make_upload_file(filename: str, content: bytes, content_type: str):
    mock = MagicMock(spec=UploadFile)
    mock.filename = filename
    mock.content_type = content_type
    mock.read = AsyncMock(return_value=content)
    mock.close = AsyncMock()
    return mock


class TestIngestDocumentController:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_ingest_txt(self):
        store = MagicMock()
        embedder = MagicMock()
        file = _make_upload_file("notes.txt", b"hello notes content", "text/plain")

        with patch(
            "app.modules.documents.controller._resolve_store_and_embedder",
            return_value=(store, embedder),
        ), patch(
            "app.modules.documents.controller.async_ingest_document_from_text",
            new=AsyncMock(return_value=3),
        ) as ingest_mock:
            resp = self._run(
                ingest_document_controller(file, workspace_id=TEST_WS_STR)
            )

        assert resp.doc_id == "notes"
        assert resp.chunk_count == 3
        assert resp.workspace_id == TEST_WS_STR
        kwargs = ingest_mock.call_args.kwargs
        assert kwargs["text"] == "hello notes content"
        assert kwargs["doc_id"] == "notes"
        assert kwargs["source_filename"] == "notes.txt"
        assert kwargs["workspace_id"] == TEST_WS_STR
        assert kwargs["pages"] is None

    def test_ingest_md(self):
        store = MagicMock()
        embedder = MagicMock()
        file = _make_upload_file("policy.md", b"# Policy\n\nBe safe.", "text/markdown")

        with patch(
            "app.modules.documents.controller._resolve_store_and_embedder",
            return_value=(store, embedder),
        ), patch(
            "app.modules.documents.controller.async_ingest_document_from_text",
            new=AsyncMock(return_value=2),
        ) as ingest_mock:
            resp = self._run(ingest_document_controller(file))

        assert resp.doc_id == "policy"
        kwargs = ingest_mock.call_args.kwargs
        assert kwargs["text"] == "# Policy\n\nBe safe."
        assert kwargs["pages"] is None

    def test_ingest_pdf_passes_pages_through(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fixtures.pdfgen import build_pdf

        store = MagicMock()
        embedder = MagicMock()
        pdf_bytes = build_pdf([["page one text"], ["page two text"]])
        file = _make_upload_file("r.pdf", pdf_bytes, "application/pdf")

        def fake_resolve(workspace_id=None):
            return (store, embedder)

        with patch(
            "app.modules.documents.controller._resolve_store_and_embedder",
            side_effect=fake_resolve,
        ), patch(
            "app.modules.documents.controller.async_ingest_document_from_text",
            new=AsyncMock(return_value=2),
        ) as ingest_mock:
            resp = self._run(ingest_document_controller(file, workspace_id=TEST_WS_B_STR))

        assert resp.doc_id == "r"
        assert resp.workspace_id == TEST_WS_B_STR
        kwargs = ingest_mock.call_args.kwargs
        assert kwargs["pages"] is not None
        assert len(kwargs["pages"]) == 2
        assert kwargs["pages"][0]["page_number"] == 1
        assert "page one text" in kwargs["pages"][0]["text"]
        assert kwargs["source_filename"] == "r.pdf"

    def test_ingest_unsupported_type_rejected(self):
        store = MagicMock()
        embedder = MagicMock()
        file = _make_upload_file("image.png", b"\x89PNG", "image/png")

        with patch(
            "app.modules.documents.controller._resolve_store_and_embedder",
            return_value=(store, embedder),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(ingest_document_controller(file))
        assert exc_info.value.status_code == 415

    def test_ingest_empty_pdf_rejected(self):
        store = MagicMock()
        embedder = MagicMock()
        file = _make_upload_file("empty.pdf", b"%PDF-1.4\n%%EOF", "application/pdf")

        with patch(
            "app.modules.documents.controller._resolve_store_and_embedder",
            return_value=(store, embedder),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(ingest_document_controller(file))
        assert exc_info.value.status_code == 422

    def test_ingest_empty_file_rejected(self):
        store = MagicMock()
        embedder = MagicMock()
        file = _make_upload_file("empty.txt", b"", "text/plain")

        with patch(
            "app.modules.documents.controller._resolve_store_and_embedder",
            return_value=(store, embedder),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(ingest_document_controller(file))
        assert exc_info.value.status_code == 400


class TestListDocumentsController:
    def test_list_documents(self):
        store = MagicMock()
        expected = [
            {"doc_id": "doc-a", "source_filename": "a.txt", "chunk_count": 2, "workspace_id": TEST_WS_STR, "pages": [1]}
        ]
        with patch(
            "app.modules.documents.controller.get_vector_store", return_value=store
        ), patch(
            "app.modules.documents.controller.async_list_documents",
            new=AsyncMock(return_value=expected),
        ):
            resp = asyncio.run(list_documents_controller(workspace_id=TEST_WS_STR))

        assert len(resp.documents) == 1
        assert resp.documents[0].doc_id == "doc-a"


class TestGetDocumentController:
    def test_get_document_found(self):
        store = MagicMock()
        expected = {"doc_id": "doc-a", "source_filename": "a.txt", "chunk_count": 2, "workspace_id": TEST_WS_STR}
        with patch(
            "app.modules.documents.controller.get_vector_store", return_value=store
        ), patch(
            "app.modules.documents.controller.async_get_document",
            new=AsyncMock(return_value=expected),
        ):
            doc = asyncio.run(get_document_controller("doc-a", workspace_id=TEST_WS_STR))
        assert doc["doc_id"] == "doc-a"

    def test_get_document_not_found(self):
        store = MagicMock()
        with patch(
            "app.modules.documents.controller.get_vector_store", return_value=store
        ), patch(
            "app.modules.documents.controller.async_get_document",
            new=AsyncMock(return_value=None),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(get_document_controller("missing"))
        assert exc_info.value.status_code == 404


class TestGetDocumentChunksController:
    def test_get_chunks(self):
        store = MagicMock()
        expected = [
            {
                "chunk_id": "doc-a_chunk_0",
                "document_id": "doc-a",
                "doc_id": "doc-a",
                "chunk_index": 0,
                "text": "chunk one",
                "page_number": 1,
                "character_start": 0,
                "character_end": 7,
                "source_filename": "a.txt",
                "workspace_id": TEST_WS_STR,
                "metadata": {},
            }
        ]
        with patch(
            "app.modules.documents.controller.get_vector_store", return_value=store
        ), patch(
            "app.modules.documents.controller.async_get_document_chunks",
            new=AsyncMock(return_value=expected),
        ):
            resp = asyncio.run(get_document_chunks_controller("doc-a"))
        assert len(resp.chunks) == 1
        assert resp.chunks[0].chunk_id == "doc-a_chunk_0"
        assert resp.chunks[0].page_number == 1
