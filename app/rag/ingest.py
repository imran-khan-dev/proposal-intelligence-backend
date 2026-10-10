"""Text chunking utilities for RAG ingestion."""

from typing import List, Optional

from app.ai.embeddings import EmbeddingProvider
from app.rag.store import VectorRecord, VectorStore


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


def ingest_document(
    text: str,
    doc_id: str,
    store: VectorStore,
    embedder: EmbeddingProvider,
    chunk_size: int = 800,
    overlap: int = 100,
    metadata: Optional[dict] = None,
) -> int:
    """Ingest a document into the vector store.

    Splits text into chunks, generates embeddings, and stores them with
    stable IDs and metadata. Re-ingesting the same doc_id replaces
    existing chunks for that document.

    Args:
        text: Document text to ingest.
        doc_id: Unique identifier for the document.
        store: Vector store to write to.
        embedder: Embedding provider to generate vectors.
        chunk_size: Maximum chunk size in characters (default: 800).
        overlap: Overlap between chunks in characters (default: 100).
        metadata: Optional metadata to include with each chunk.

    Returns:
        Number of chunks ingested (0 if text is empty/whitespace).

    Raises:
        ValueError: If embedding count doesn't match chunk count.
        RuntimeError: If embedding fails (existing chunks preserved).
    """
    # Handle empty/whitespace text - remove existing chunks and return 0
    if not text or not text.strip():
        if store.count() > 0:
            store.delete_by_metadata({"doc_id": doc_id})
        return 0

    # Step 1: Chunk the text
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return 0

    # Step 2: Generate embeddings (validate before deleting old data)
    try:
        embeddings = embedder.embed(chunks)
    except Exception as exc:
        raise RuntimeError(f"Embedding failed: {exc}") from exc

    # Step 3: Validate embedding count matches chunk count
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Embedding count mismatch: got {len(embeddings)} embeddings "
            f"for {len(chunks)} chunks"
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

    # Step 4: Build VectorRecord objects
    records: List[VectorRecord] = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        record_metadata = {
            **(metadata or {}),
            "doc_id": doc_id,
            "chunk_index": i,
        }

        records.append(
            VectorRecord(
                id=f"{doc_id}_chunk_{i}",
                text=chunk,
                embedding=embedding,
                metadata=record_metadata,
            )
        )

    # Step 5: Delete old chunks for this doc_id (only after successful embedding)
    store.delete_by_metadata({"doc_id": doc_id})

    # Step 6: Add new chunks
    store.add(records)

    return len(records)