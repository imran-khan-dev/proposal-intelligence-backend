"""Text chunking utilities for RAG ingestion with source-reference metadata.

Offset convention
-----------------
All character offsets (``character_start``/``character_end``) refer to the
*original normalized document text*, never to chunk-relative positions.

The normalized document text is built by concatenating per-page text segments
with a ``"\\n\\n"`` separator.  Page ``N`` occupies the character range
``[page_start, page_end)`` within that normalized text.  A chunk produced from
page ``N`` therefore carries ``character_start`` and ``character_end`` that are
absolute offsets into the normalized document text, plus ``page_number = N``.

For single-page documents (TXT, MD, or a PDF with no extractable pages) the
whole document is treated as one segment and ``page_number`` is ``1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pypdf import PdfReader

from app.ai.embeddings import EmbeddingProvider
from app.rag.store import VectorRecord, VectorStore


@dataclass
class TextChunk:
    """A chunk of document text with source-reference metadata.

    Attributes:
        text: The chunk text.
        page_number: 1-based page number the chunk originates from, or ``None``
            for documents without pages.
        section_path: Optional dotted section path (e.g. "3.2.1") when the
            section structure is known. Not auto-populated in this milestone.
        character_start: Absolute character offset (inclusive) of the chunk's
            first character within the normalized document text.
        character_end: Absolute character offset (exclusive) of the chunk's
            last character within the normalized document text.
        metadata: Extra metadata merged into the stored record.
    """

    text: str
    page_number: Optional[int] = None
    section_path: Optional[str] = None
    character_start: int = 0
    character_end: int = 0
    metadata: Optional[dict] = None


# ---------------------------------------------------------------------------
# Legacy chunker (preserved unchanged for backward compatibility)
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum size of each chunk in characters. Must be positive.
        overlap: Number of characters to overlap between consecutive chunks.
            Must be non-negative and less than chunk_size.

    Returns:
        A list of text chunks. Empty list if input is empty or whitespace-only.

    Raises:
        ValueError: If chunk_size is not positive, or overlap is negative,
            or overlap >= chunk_size.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


# ---------------------------------------------------------------------------
# Page-aware chunking with absolute character offsets
# ---------------------------------------------------------------------------

def chunk_text_with_offsets(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
    page_number: Optional[int] = None,
    base_offset: int = 0,
    section_path: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> List[TextChunk]:
    """Split text into overlapping chunks with absolute source offsets.

    Offsets are absolute positions within the normalized document text.  The
    caller supplies ``base_offset`` so that chunks from different pages can be
    positioned correctly within the full document.

    Args:
        text: The input text to chunk (already the page/segment text).
        chunk_size: Maximum size of each chunk in characters.
        overlap: Number of characters to overlap between consecutive chunks.
        page_number: 1-based page number for all chunks produced here.
        base_offset: Absolute offset of ``text``'s first character within the
            normalized document text.
        section_path: Optional section path recorded on every chunk.
        metadata: Extra metadata merged into every chunk.

    Returns:
        A list of :class:`TextChunk`. Empty list if input is empty or
        whitespace-only.

    Raises:
        ValueError: If chunk_size is not positive, or overlap is negative,
            or overlap >= chunk_size.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    if len(text) <= chunk_size:
        return [
            TextChunk(
                text=text,
                page_number=page_number,
                section_path=section_path,
                character_start=base_offset,
                character_end=base_offset + len(text),
                metadata=metadata,
            )
        ]

    chunks: List[TextChunk] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_slice = text[start:end]
        abs_start = base_offset + start
        abs_end = base_offset + end
        chunks.append(
            TextChunk(
                text=chunk_text_slice,
                page_number=page_number,
                section_path=section_path,
                character_start=abs_start,
                character_end=abs_end,
                metadata=metadata,
            )
        )

        if end == len(text):
            break

        start = end - overlap

    return chunks


def build_normalized_text(
    pages: List[dict],
) -> str:
    """Build the normalized document text from per-page segments.

    Each entry in ``pages`` must have a ``"text"`` key.  Pages are joined with
    ``"\\n\\n"`` so that character offsets computed against the returned string
    line up with the per-page ``character_start``/``character_end`` offsets
    produced by :func:`chunk_pages`.

    Args:
        pages: List of page dicts, each containing at least ``"text"``.

    Returns:
        The normalized full-document text.
    """
    return "\n\n".join(page.get("text", "") for page in pages)


def chunk_pages(
    pages: List[dict],
    chunk_size: int = 800,
    overlap: int = 100,
    base_metadata: Optional[dict] = None,
) -> List[TextChunk]:
    """Chunk a list of page segments, preserving absolute source offsets.

    Each page dict may contain:
        - ``"text"`` (required): page text.
        - ``"page_number"`` (optional, defaults to 1-based index).
        - ``"character_start"``/``"character_end"`` (optional): absolute offsets
          within the normalized document text. If omitted, offsets are computed
          by concatenating page texts with ``"\\n\\n"`` separators.

    Chunks never cross a page boundary: each page is chunked independently.

    Args:
        pages: List of page segment dicts.
        chunk_size: Maximum size of each chunk in characters.
        overlap: Number of characters to overlap between consecutive chunks.
        base_metadata: Metadata merged into every chunk.

    Returns:
        A list of :class:`TextChunk` with absolute offsets and page numbers.
    """
    chunks: List[TextChunk] = []
    running_offset = 0

    for index, page in enumerate(pages):
        page_text = page.get("text", "") or ""
        page_number = page.get("page_number", index + 1)

        if "character_start" in page and "character_end" in page:
            base_offset = int(page["character_start"])
        else:
            base_offset = running_offset

        page_chunks = chunk_text_with_offsets(
            text=page_text,
            chunk_size=chunk_size,
            overlap=overlap,
            page_number=page_number,
            base_offset=base_offset,
            section_path=page.get("section_path"),
            metadata=base_metadata,
        )
        chunks.extend(page_chunks)

        # Advance the running offset by the page text length plus the "\n\n"
        # separator used by build_normalized_text.
        running_offset += len(page_text) + 2

    return chunks


def extract_pdf_pages(file_path: str) -> List[dict]:
    """Extract text from a PDF file page by page with character offsets.

    Each returned dict describes one page:

        - ``"page_number"``: 1-based page index.
        - ``"text"``: extracted page text (empty string for scanned/empty pages).
        - ``"character_start"``: absolute offset (inclusive) of the page's first
          character within the normalized document text.
        - ``"character_end"``: absolute offset (exclusive) of the page's last
          character within the normalized document text.

    Page boundaries are preserved: pages are never merged across the boundary.
    Empty pages yield an empty ``"text"`` and a zero-width range.  Scanned PDFs
    with no extractable text layer are *not* OCR'd; such pages simply produce
    empty text.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of page dicts, one per PDF page, in document order.

    Raises:
        ValueError: If the PDF contains no pages or cannot be read.
    """
    try:
        reader = PdfReader(file_path)
    except Exception as exc:
        raise ValueError(f"Unable to read PDF: {exc}") from exc

    if len(reader.pages) == 0:
        raise ValueError("PDF contains no pages")

    pages: List[dict] = []
    offset = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(
            {
                "page_number": page_number,
                "text": text,
                "character_start": offset,
                "character_end": offset + len(text),
            }
        )
        offset += len(text) + 2  # account for the "\n\n" page separator

    return pages


# ---------------------------------------------------------------------------
# Document ingestion (chunking + embedding + store)
# ---------------------------------------------------------------------------

def _build_record_metadata(
    doc_id: str,
    chunk_index: int,
    page_number: Optional[int],
    character_start: int,
    character_end: int,
    source_filename: Optional[str],
    section_path: Optional[str],
    workspace_id: Optional[str],
    extra: Optional[dict] = None,
) -> dict:
    metadata: dict = {
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "page_number": page_number,
        "character_start": character_start,
        "character_end": character_end,
    }
    if source_filename is not None:
        metadata["source_filename"] = source_filename
    if section_path is not None:
        metadata["section_path"] = section_path
    if workspace_id is not None:
        metadata["workspace_id"] = str(workspace_id)
    if extra:
        metadata.update(extra)
    return metadata


def ingest_document(
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
    """Ingest a document into the vector store (synchronous version).

    Splits text into chunks, generates embeddings, and stores them with
    stable IDs and source-reference metadata. Re-ingesting the same
    ``doc_id`` replaces existing chunks for that document.

    Args:
        text: Document text to ingest.
        doc_id: Unique identifier for the document.
        store: Vector store to write to.
        embedder: Embedding provider to generate vectors.
        chunk_size: Maximum chunk size in characters (default: 800).
        overlap: Overlap between chunks in characters (default: 100).
        metadata: Optional metadata to include with each chunk.
        pages: Optional list of page segment dicts (see :func:`chunk_pages`).
            When provided, chunks are produced per-page with absolute
            character offsets and page numbers.
        source_filename: Original filename of the source document.
        workspace_id: Workspace identifier for isolation, stored in chunk
            metadata.

    Returns:
        Number of chunks ingested (0 if text is empty/whitespace).

    Raises:
        ValueError: If embedding count doesn't match chunk count.
        RuntimeError: If embedding fails (existing chunks preserved).
    """
    # Handle empty/whitespace text - remove existing chunks and return 0
    if not text or not text.strip():
        delete_filter = {"doc_id": doc_id}
        if workspace_id is not None:
            delete_filter["workspace_id"] = workspace_id
        if store.count() > 0:
            store.delete_by_metadata(delete_filter)
        return 0

    # Build page-aware chunks with absolute offsets when pages are provided.
    if pages:
        text_chunks = chunk_pages(
            pages=pages,
            chunk_size=chunk_size,
            overlap=overlap,
            base_metadata=metadata,
        )
    else:
        # Single segment: treat the whole document as one logical page.
        text_chunks = chunk_text_with_offsets(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
            page_number=1,
            base_offset=0,
            section_path=None,
            metadata=metadata,
        )

    if not text_chunks:
        delete_filter = {"doc_id": doc_id}
        if workspace_id is not None:
            delete_filter["workspace_id"] = workspace_id
        if store.count() > 0:
            store.delete_by_metadata(delete_filter)
        return 0

    chunk_texts = [chunk.text for chunk in text_chunks]

    # Step 2: Generate embeddings (validate before deleting old data)
    try:
        embeddings = embedder.embed(chunk_texts)
    except Exception as exc:
        raise RuntimeError(f"Embedding failed: {exc}") from exc

    # Step 3: Validate embedding count matches chunk count
    if len(embeddings) != len(chunk_texts):
        raise ValueError(
            f"Embedding count mismatch: got {len(embeddings)} embeddings "
            f"for {len(chunk_texts)} chunks"
        )

    # Step 3b: Validate embedding dimensions
    if not embeddings:
        raise ValueError("No embeddings generated")
    new_dim = len(embeddings[0])
    for i, emb in enumerate(embeddings[1:], 1):
        if len(emb) != new_dim:
            raise ValueError(
                f"Inconsistent embedding dimensions: embedding 0 has {new_dim}, "
                f"embedding {i} has {len(emb)}"
            )
    if store.dimension is not None and new_dim != store.dimension:
        raise ValueError(
            f"Embedding dimension mismatch: store has {store.dimension}, "
            f"new embeddings have {new_dim}"
        )

    # Step 4: Build VectorRecord objects with source-reference metadata
    records: List[VectorRecord] = []
    for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
        record_metadata = _build_record_metadata(
            doc_id=doc_id,
            chunk_index=i,
            page_number=chunk.page_number,
            character_start=chunk.character_start,
            character_end=chunk.character_end,
            source_filename=source_filename,
            section_path=chunk.section_path,
            workspace_id=workspace_id,
            extra=chunk.metadata,
        )
        records.append(
            VectorRecord(
                id=f"{doc_id}_chunk_{i}",
                text=chunk.text,
                embedding=embedding,
                metadata=record_metadata,
            )
        )

    # Step 5: Delete old chunks for this doc_id (only after successful embedding).
    # Scope by workspace_id so re-ingesting the same doc_id in another workspace
    # cannot delete that workspace's chunks.
    delete_filter = {"doc_id": doc_id}
    if workspace_id is not None:
        delete_filter["workspace_id"] = workspace_id
    store.delete_by_metadata(delete_filter)

    # Step 6: Add new chunks
    store.add(records)

    return len(records)


async def async_ingest_document(
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
    """Ingest a document into the vector store (async version for FastAPI).

    See :func:`ingest_document` for argument and behavior details.
    """
    # Handle empty/whitespace text - remove existing chunks and return 0
    if not text or not text.strip():
        delete_filter = {"doc_id": doc_id}
        if workspace_id is not None:
            delete_filter["workspace_id"] = workspace_id
        if await store.async_count() > 0:
            await store.async_delete_by_metadata(delete_filter)
        return 0

    # Build page-aware chunks with absolute offsets when pages are provided.
    if pages:
        text_chunks = chunk_pages(
            pages=pages,
            chunk_size=chunk_size,
            overlap=overlap,
            base_metadata=metadata,
        )
    else:
        text_chunks = chunk_text_with_offsets(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
            page_number=1,
            base_offset=0,
            section_path=None,
            metadata=metadata,
        )

    if not text_chunks:
        delete_filter = {"doc_id": doc_id}
        if workspace_id is not None:
            delete_filter["workspace_id"] = workspace_id
        if await store.async_count() > 0:
            await store.async_delete_by_metadata(delete_filter)
        return 0

    chunk_texts = [chunk.text for chunk in text_chunks]

    # Step 2: Generate embeddings (validate before deleting old data)
    try:
        embeddings = embedder.embed(chunk_texts)
    except Exception as exc:
        raise RuntimeError(f"Embedding failed: {exc}") from exc

    # Step 3: Validate embedding count matches chunk count
    if len(embeddings) != len(chunk_texts):
        raise ValueError(
            f"Embedding count mismatch: got {len(embeddings)} embeddings "
            f"for {len(chunk_texts)} chunks"
        )

    # Step 3b: Validate embedding dimensions
    if not embeddings:
        raise ValueError("No embeddings generated")
    new_dim = len(embeddings[0])
    for i, emb in enumerate(embeddings[1:], 1):
        if len(emb) != new_dim:
            raise ValueError(
                f"Inconsistent embedding dimensions: embedding 0 has {new_dim}, "
                f"embedding {i} has {len(emb)}"
            )
    if store.dimension is not None and new_dim != store.dimension:
        raise ValueError(
            f"Embedding dimension mismatch: store has {store.dimension}, "
            f"new embeddings have {new_dim}"
        )

    # Step 4: Build VectorRecord objects with source-reference metadata
    records: List[VectorRecord] = []
    for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
        record_metadata = _build_record_metadata(
            doc_id=doc_id,
            chunk_index=i,
            page_number=chunk.page_number,
            character_start=chunk.character_start,
            character_end=chunk.character_end,
            source_filename=source_filename,
            section_path=chunk.section_path,
            workspace_id=workspace_id,
            extra=chunk.metadata,
        )
        records.append(
            VectorRecord(
                id=f"{doc_id}_chunk_{i}",
                text=chunk.text,
                embedding=embedding,
                metadata=record_metadata,
            )
        )

    # Step 5: Delete old chunks for this doc_id (only after successful embedding).
    # Scope by workspace_id so re-ingesting the same doc_id in another workspace
    # cannot delete that workspace's chunks.
    delete_filter = {"doc_id": doc_id}
    if workspace_id is not None:
        delete_filter["workspace_id"] = workspace_id
    await store.async_delete_by_metadata(delete_filter)

    # Step 6: Add new chunks
    await store.async_add(records)

    return len(records)
