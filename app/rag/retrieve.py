"""RAG retrieval pipeline."""

from typing import List

from app.ai.embeddings import EmbeddingProvider
from app.rag.store import SearchResult, VectorStore


def retrieve(
    query: str,
    embedder: EmbeddingProvider,
    store: VectorStore,
    top_k: int = 5,
) -> List[SearchResult]:
    """Retrieve relevant chunks for a query.

    Args:
        query: The search query text.
        embedder: Embedding provider to generate query vector.
        store: Vector store to search.
        top_k: Maximum number of results to return (default: 5).

    Returns:
        List of SearchResult objects sorted by similarity (highest first).
        Returns empty list for empty query, empty store, or top_k <= 0.

    Raises:
        ValueError: If query is blank or top_k is not positive.
        RuntimeError: If embedding fails.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # Embed the query
    try:
        query_embeddings = embedder.embed([query])
    except Exception as exc:
        raise RuntimeError(f"Query embedding failed: {exc}") from exc

    if not query_embeddings:
        raise RuntimeError("Embedding provider returned no embeddings")

    query_embedding = query_embeddings[0]

    # Search the store
    return store.search(query_embedding, k=top_k)