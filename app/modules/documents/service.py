"""Document ingestion service.

Reuses the existing chunking and embedding pipeline from ``app.rag.ingest``.
No chunking or embedding logic is duplicated here.
"""

from pathlib import Path
from typing import Optional

from app.ai.embeddings import EmbeddingProvider
from app.rag.ingest import async_ingest_document, ingest_document
from app.rag.store import VectorStore


SUPPORTED_EXTENSIONS = {".txt", ".md"}


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


def ingest_document_from_text(
    text: str,
    doc_id: str,
    store: VectorStore,
    embedder: EmbeddingProvider,
    chunk_size: int = 800,
    overlap: int = 100,
    metadata: Optional[dict] = None,
) -> int:
    """Validate and ingest document text into a vector store (sync version).

    Delegates to :func:`app.rag.ingest.ingest_document` without duplicating
    any chunking or embedding logic.

    Args:
        text: Document text to ingest.
        doc_id: Unique identifier for the document.
        store: Vector store to write to.
        embedder: Embedding provider to generate vectors.
        chunk_size: Maximum chunk size in characters (default: 800).
        overlap: Overlap between chunks in characters (default: 100).
        metadata: Optional metadata to include with each chunk.

    Returns:
        Number of chunks ingested.

    Raises:
        ValueError: If the text is empty or embedding dimensions mismatch.
        RuntimeError: If embedding fails.
    """
    if not text or not text.strip():
        raise ValueError("Document contains no text content")

    return ingest_document(
        text=text,
        doc_id=doc_id,
        store=store,
        embedder=embedder,
        chunk_size=chunk_size,
        overlap=overlap,
        metadata=metadata,
    )


async def async_ingest_document_from_text(
    text: str,
    doc_id: str,
    store: VectorStore,
    embedder: EmbeddingProvider,
    chunk_size: int = 800,
    overlap: int = 100,
    metadata: Optional[dict] = None,
) -> int:
    """Validate and ingest document text into a vector store (async version).

    Delegates to :func:`app.rag.ingest.async_ingest_document` without duplicating
    any chunking or embedding logic.

    Args:
        text: Document text to ingest.
        doc_id: Unique identifier for the document.
        store: Vector store to write to.
        embedder: Embedding provider to generate vectors.
        chunk_size: Maximum chunk size in characters (default: 800).
        overlap: Overlap between chunks in characters (default: 100).
        metadata: Optional metadata to include with each chunk.

    Returns:
        Number of chunks ingested.

    Raises:
        ValueError: If the text is empty or embedding dimensions mismatch.
        RuntimeError: If embedding fails.
    """
    if not text or not text.strip():
        raise ValueError("Document contains no text content")

    return await async_ingest_document(
        text=text,
        doc_id=doc_id,
        store=store,
        embedder=embedder,
        chunk_size=chunk_size,
        overlap=overlap,
        metadata=metadata,
    )
