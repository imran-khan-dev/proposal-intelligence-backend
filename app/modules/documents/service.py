"""Document ingestion service.

Reuses the existing chunking and embedding pipeline from ``app.rag.ingest``.
No chunking or embedding logic is duplicated here.

Supported extensions: ``.pdf``, ``.txt``, ``.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from app.ai.embeddings import EmbeddingProvider
from app.rag.ingest import (
    TextChunk,
    async_ingest_document,
    build_normalized_text,
    chunk_pages,
    extract_pdf_pages,
    ingest_document,
)
from app.rag.store import VectorStore


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def derive_doc_id(filename: str | None) -> str:
    """Derive a stable document ID from a filename.

    Uses the filename stem (without extension). Falls back to ``"document"``
    when the filename is missing, empty, or has no meaningful stem
    (e.g. a bare extension like ``.txt``).
    """
    if not filename:
        return "document"
    p = Path(filename)
    stem = p.stem
    if not stem or (stem == p.name and p.name.startswith(".")):
        return "document"
    return stem


def read_text_content(content: bytes) -> str:
    """Decode raw bytes to text, handling encoding issues.

    Attempts UTF-8 first, falling back to latin-1 (which can decode any
    byte sequence) for files saved in legacy encodings.
    """
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _extract_pdf_pages_from_bytes(content: bytes, filename: str) -> List[dict]:
    """Extract pages from an in-memory PDF byte buffer.

    Writes the bytes to a temporary file and delegates to
    :func:`app.rag.ingest.extract_pdf_pages`.

    Returns:
        A list of page dicts (see :func:`extract_pdf_pages`).
    """
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_path = temp_file.name
        temp_file.write(content)

    try:
        return extract_pdf_pages(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def ingest_document_from_text(
    text: str,
    doc_id: str,
    store: VectorStore,
    embedder: EmbeddingProvider,
    chunk_size: int = 800,
    overlap: int = 100,
    metadata: Optional[dict] = None,
    pages: Optional[List[dict]] = None,
    source_filename: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> int:
    """Validate and ingest document text into a vector store (sync version).

    Delegates to :func:`app.rag.ingest.ingest_document` without duplicating
    any chunking or embedding logic. When ``pages`` are provided (e.g. from a
    PDF), chunks are produced per-page with absolute character offsets and
    page numbers so every stored chunk carries a traceable source reference.

    Args:
        text: Document text to ingest.
        doc_id: Unique identifier for the document.
        store: Vector store to write to.
        embedder: Embedding provider to generate vectors.
        chunk_size: Maximum chunk size in characters (default: 800).
        overlap: Overlap between chunks in characters (default: 100).
        metadata: Optional metadata to include with each chunk.
        pages: Optional list of page segment dicts (see
            :func:`app.rag.ingest.chunk_pages`).
        source_filename: Original filename of the source document.
        workspace_id: Workspace identifier for isolation, stored in chunk
            metadata.

    Returns:
        Number of chunks ingested.

    Raises:
        ValueError: If the text is empty or embedding dimensions mismatch.
        RuntimeError: If embedding fails.
    """
    return ingest_document(
        text=text,
        doc_id=doc_id,
        store=store,
        embedder=embedder,
        chunk_size=chunk_size,
        overlap=overlap,
        metadata=metadata,
        pages=pages,
        source_filename=source_filename,
        workspace_id=workspace_id,
    )


async def async_ingest_document_from_text(
    text: str,
    doc_id: str,
    store: VectorStore,
    embedder: EmbeddingProvider,
    chunk_size: int = 800,
    overlap: int = 100,
    metadata: Optional[dict] = None,
    pages: Optional[List[dict]] = None,
    source_filename: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> int:
    """Validate and ingest document text into a vector store (async version).

    Delegates to :func:`app.rag.ingest.async_ingest_document` without
    duplicating any chunking or embedding logic.

    Args:
        text: Document text to ingest.
        doc_id: Unique identifier for the document.
        store: Vector store to write to.
        embedder: Embedding provider to generate vectors.
        chunk_size: Maximum chunk size in characters (default: 800).
        overlap: Overlap between chunks in characters (default: 100).
        metadata: Optional metadata to include with each chunk.
        pages: Optional list of page segment dicts (see
            :func:`app.rag.ingest.chunk_pages`).
        source_filename: Original filename of the source document.
        workspace_id: Workspace identifier for isolation, stored in chunk
            metadata.

    Returns:
        Number of chunks ingested.

    Raises:
        ValueError: If the text is empty or embedding dimensions mismatch.
        RuntimeError: If embedding fails.
    """
    return await async_ingest_document(
        text=text,
        doc_id=doc_id,
        store=store,
        embedder=embedder,
        chunk_size=chunk_size,
        overlap=overlap,
        metadata=metadata,
        pages=pages,
        source_filename=source_filename,
        workspace_id=workspace_id,
    )


# ---------------------------------------------------------------------------
# Document query operations (list / get / get-chunks)
# ---------------------------------------------------------------------------

def list_documents(
    store: VectorStore, workspace_id: Optional[str] = None
) -> List[dict]:
    """List ingested documents, optionally scoped to a workspace.

    Args:
        store: Vector store to query.
        workspace_id: If set, only documents in this workspace are returned.

    Returns:
        A list of document summary dicts:
        ``doc_id``, ``source_filename``, ``title``, ``chunk_count``,
        ``workspace_id``.
    """
    if hasattr(store, "list_documents"):
        return store.list_documents(workspace_id=workspace_id)  # type: ignore[attr-defined]
    if hasattr(store, "async_list_documents"):
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                store.async_list_documents(workspace_id=workspace_id)  # type: ignore[attr-defined]
            )
        raise RuntimeError("Use async_list_documents() from an async context.")
    raise NotImplementedError(
        f"Store {type(store).__name__} does not support list_documents()."
    )


async def async_list_documents(
    store: VectorStore, workspace_id: Optional[str] = None
) -> List[dict]:
    """Async variant of :func:`list_documents`."""
    if hasattr(store, "async_list_documents"):
        return await store.async_list_documents(workspace_id=workspace_id)  # type: ignore[attr-defined]
    if hasattr(store, "list_documents"):
        return store.list_documents(workspace_id=workspace_id)  # type: ignore[attr-defined]
    raise NotImplementedError(
        f"Store {type(store).__name__} does not support async_list_documents()."
    )


def get_document(
    store: VectorStore, doc_id: str, workspace_id: Optional[str] = None
) -> Optional[dict]:
    """Get metadata for a single document, optionally scoped to a workspace."""
    if hasattr(store, "get_document"):
        return store.get_document(doc_id, workspace_id=workspace_id)  # type: ignore[attr-defined]
    if hasattr(store, "async_get_document"):
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                store.async_get_document(doc_id, workspace_id=workspace_id)  # type: ignore[attr-defined]
            )
        raise RuntimeError("Use async_get_document() from an async context.")
    raise NotImplementedError(
        f"Store {type(store).__name__} does not support get_document()."
    )


async def async_get_document(
    store: VectorStore, doc_id: str, workspace_id: Optional[str] = None
) -> Optional[dict]:
    """Async variant of :func:`get_document`."""
    if hasattr(store, "async_get_document"):
        return await store.async_get_document(doc_id, workspace_id=workspace_id)  # type: ignore[attr-defined]
    if hasattr(store, "get_document"):
        return store.get_document(doc_id, workspace_id=workspace_id)  # type: ignore[attr-defined]
    raise NotImplementedError(
        f"Store {type(store).__name__} does not support async_get_document()."
    )


def get_document_chunks(
    store: VectorStore, doc_id: str, workspace_id: Optional[str] = None
) -> List[dict]:
    """Get all chunks for a document with their source references."""
    if hasattr(store, "get_document_chunks"):
        return store.get_document_chunks(doc_id, workspace_id=workspace_id)  # type: ignore[attr-defined]
    if hasattr(store, "async_get_document_chunks"):
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                store.async_get_document_chunks(doc_id, workspace_id=workspace_id)  # type: ignore[attr-defined]
            )
        raise RuntimeError("Use async_get_document_chunks() from an async context.")
    raise NotImplementedError(
        f"Store {type(store).__name__} does not support get_document_chunks()."
    )


async def async_get_document_chunks(
    store: VectorStore, doc_id: str, workspace_id: Optional[str] = None
) -> List[dict]:
    """Async variant of :func:`get_document_chunks`."""
    if hasattr(store, "async_get_document_chunks"):
        return await store.async_get_document_chunks(doc_id, workspace_id=workspace_id)  # type: ignore[attr-defined]
    if hasattr(store, "get_document_chunks"):
        return store.get_document_chunks(doc_id, workspace_id=workspace_id)  # type: ignore[attr-defined]
    raise NotImplementedError(
        f"Store {type(store).__name__} does not support async_get_document_chunks()."
    )
