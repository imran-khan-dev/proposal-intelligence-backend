"""Unit tests for the document ingestion service.

Uses mocked stores and embedding providers so no database, network, or API key
is required. Covers PDF/TXT/MD ingestion, workspace metadata, source-reference
preservation, re-ingestion, and document query operations against the in-memory
store.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.rag.store import InMemoryVectorStore, VectorRecord
from app.modules.documents.service import (
    SUPPORTED_EXTENSIONS,
    async_get_document,
    async_get_document_chunks,
    async_ingest_document_from_text,
    async_list_documents,
    derive_doc_id,
    read_text_content,
)


class TestDeriveDocId:
    def test_stem_without_extension(self):
        assert derive_doc_id("proposal.pdf") == "proposal"

    def test_missing_filename(self):
        assert derive_doc_id(None) == "document"
        assert derive_doc_id("") == "document"

    def test_dotfile_bare_extension(self):
        assert derive_doc_id(".txt") == "document"


class TestReadTextContent:
    def test_utf8(self):
        assert read_text_content("hello".encode("utf-8")) == "hello"

    def test_latin1_fallback(self):
        assert read_text_content(b"\xe9\xf1") == "\xe9\xf1".encode("latin-1").decode("latin-1")


class TestSupportedExtensions:
    def test_pdf_txt_md_supported(self):
        assert SUPPORTED_EXTENSIONS == {".pdf", ".txt", ".md"}


class FakeEmbedder:
    """Deterministic embedding provider for tests."""

    def __init__(self, dimension: int = 8):
        self.dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def __init__(self, dimension: int = 8):
        self._dim = dimension

    def embed(self, texts):
        return [[float(i) / 10.0] * self._dim for i in range(len(texts))]


async def _ingest(
    store,
    text,
    doc_id="doc1",
    pages=None,
    workspace_id="ws-1",
    source_filename="sample.txt",
):
    embedder = FakeEmbedder()
    return await async_ingest_document_from_text(
        text=text,
        doc_id=doc_id,
        store=store,
        embedder=embedder,
        pages=pages,
        source_filename=source_filename,
        workspace_id=workspace_id,
    )


class TestAsyncIngestDocumentFromText:
    @pytest.mark.asyncio
    async def test_ingests_txt_as_single_page(self):
        store = InMemoryVectorStore()
        count = await _ingest(store, "hello world", source_filename="sample.txt")
        assert count == 1
        assert store.count() == 1

        records = store._records
        assert records[0].metadata["doc_id"] == "doc1"
        assert records[0].metadata["page_number"] == 1
        assert records[0].metadata["character_start"] == 0
        assert records[0].metadata["character_end"] == len("hello world")
        assert records[0].metadata["source_filename"] == "sample.txt"
        assert records[0].metadata["workspace_id"] == "ws-1"
        # Offsets refer to normalized text.
        assert records[0].text == "hello world"

    @pytest.mark.asyncio
    async def test_pdf_pages_preserve_page_and_offsets(self):
        store = InMemoryVectorStore()
        pages = [
            {"page_number": 1, "text": "page one alpha beta", "character_start": 0, "character_end": 23},
            {"page_number": 2, "text": "page two gamma delta", "character_start": 25, "character_end": 48},
        ]
        text = "page one alpha beta\n\npage two gamma delta"
        count = await _ingest(store, text, pages=pages, source_filename="r.pdf")
        assert count == 2
        p1, p2 = store._records
        assert p1.metadata["page_number"] == 1
        assert p1.metadata["character_start"] == 0
        assert p2.metadata["page_number"] == 2
        assert p2.metadata["character_start"] == 25

    @pytest.mark.asyncio
    async def test_re_ingestion_replaces_chunks(self):
        store = InMemoryVectorStore()
        await _ingest(store, "first version content", doc_id="doc1")
        assert store.count() == 1
        await _ingest(store, "second version content that is longer", doc_id="doc1")
        assert store.count() == 1
        assert store._records[0].text == "second version content that is longer"

    @pytest.mark.asyncio
    async def test_empty_text_removes_existing_chunks(self):
        store = InMemoryVectorStore()
        await _ingest(store, "some content", doc_id="doc1")
        assert store.count() == 1
        count = await _ingest(store, "", doc_id="doc1")
        assert count == 0
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_workspace_id_stored_in_metadata(self):
        store = InMemoryVectorStore()
        await _ingest(store, "workspace content", workspace_id="ws-alpha")
        assert store._records[0].metadata["workspace_id"] == "ws-alpha"

    @pytest.mark.asyncio
    async def test_dimension_mismatch_raises(self):
        store = InMemoryVectorStore()
        # Pre-seed store with a different dimension via add of one record.
        store.add(
            [
                VectorRecord(
                    id="existing",
                    text="existing",
                    embedding=[0.1, 0.2, 0.3],
                    metadata={"doc_id": "other"},
                )
            ]
        )
        # FakeEmbedder produces 8-dim embeddings; store dimension is 3.
        with pytest.raises(ValueError, match="dimension mismatch"):
            await _ingest(store, "new content", doc_id="new")


class TestDocumentQueriesInMemory:
    @pytest.mark.asyncio
    async def test_list_documents_scoped_by_workspace(self):
        store = InMemoryVectorStore()
        await _ingest(store, "ws-a doc", doc_id="doc-a", workspace_id="ws-a")
        await _ingest(store, "ws-b doc", doc_id="doc-b", workspace_id="ws-b")

        docs_ws_a = await async_list_documents(store, workspace_id="ws-a")
        assert [d["doc_id"] for d in docs_ws_a] == ["doc-a"]

        docs_all = await async_list_documents(store)
        assert {d["doc_id"] for d in docs_all} == {"doc-a", "doc-b"}

    @pytest.mark.asyncio
    async def test_get_document(self):
        store = InMemoryVectorStore()
        await _ingest(store, "document content here", doc_id="doc-x", source_filename="x.txt")
        doc = await async_get_document(store, "doc-x")
        assert doc is not None
        assert doc["doc_id"] == "doc-x"
        assert doc["source_filename"] == "x.txt"
        assert doc["chunk_count"] == 1

        missing = await async_get_document(store, "does-not-exist")
        assert missing is None

    @pytest.mark.asyncio
    async def test_get_document_chunks_include_source_reference(self):
        store = InMemoryVectorStore()
        pages = [
            {"page_number": 1, "text": "alpha beta", "character_start": 0, "character_end": 8},
        ]
        await _ingest(
            store,
            "alpha beta\n\n",
            doc_id="doc-y",
            pages=pages,
            source_filename="y.pdf",
        )
        chunks = await async_get_document_chunks(store, "doc-y")
        assert len(chunks) == 1
        c = chunks[0]
        assert c["doc_id"] == "doc-y"
        assert c["page_number"] == 1
        assert c["character_start"] == 0
        # "alpha beta" is 10 characters.
        assert c["character_end"] == 10
        assert c["source_filename"] == "y.pdf"
        assert c["chunk_id"] == "doc-y_chunk_0"

    @pytest.mark.asyncio
    async def test_get_document_chunks_respects_workspace(self):
        store = InMemoryVectorStore()
        await _ingest(store, "ws-a content", doc_id="shared", workspace_id="ws-a")
        await _ingest(store, "ws-b content", doc_id="shared", workspace_id="ws-b")

        chunks_a = await async_get_document_chunks(store, "shared", workspace_id="ws-a")
        assert len(chunks_a) == 1
        assert chunks_a[0]["workspace_id"] == "ws-a"

        chunks_b = await async_get_document_chunks(store, "shared", workspace_id="ws-b")
        assert len(chunks_b) == 1
        assert chunks_b[0]["workspace_id"] == "ws-b"
