"""RAG retrieval pipeline with hybrid search support.

This module provides the `retrieve` function for backward compatibility,
which now uses the HybridRetrievalService (built on Retrieval Stack components)
when available, with fallback to the original vector-only search.
"""

from typing import List, Optional

from app.ai.embeddings import EmbeddingProvider
from app.rag.hybrid_retrieval import HybridRetrievalService, create_hybrid_retrieval_service
from app.rag.store import SearchResult, VectorStore
from app.rag.stores.postgres_vector_store import PostgreSQLVectorStore


def _create_hybrid_service(
    embedder: EmbeddingProvider,
    store: VectorStore,
    enable_reranking: bool = False,
) -> Optional[HybridRetrievalService]:
    """Create a hybrid retrieval service for the given store.

    Returns None if the store is not a PostgreSQLVectorStore (hybrid search
    requires PostgreSQL for BM25 index building).
    """
    if not isinstance(store, PostgreSQLVectorStore):
        return None

    return create_hybrid_retrieval_service(
        store=store,
        embedder=embedder,
        enable_reranking=enable_reranking,
        cache_bm25_index=True,
    )


def retrieve(
    query: str,
    embedder: EmbeddingProvider,
    store: VectorStore,
    top_k: int = 5,
    source_type: Optional[str] = None,
    use_hybrid: bool = True,
    enable_reranking: bool = False,
) -> List[SearchResult]:
    """Retrieve relevant chunks for a query using hybrid search when available.

    This function maintains backward compatibility with the original API while
    optionally using the Retrieval Stack's hybrid search (BM25 + Dense + RRF)
    and reranking capabilities when a PostgreSQLVectorStore is used.

    Args:
        query: The search query text.
        embedder: Embedding provider to generate query vector.
        store: Vector store to search.
        top_k: Maximum number of results to return (default: 5).
        source_type: Optional document category filter. When set, only chunks
            whose metadata ``source_type`` matches are searched (e.g.
            ``"company_knowledge"`` to exclude RFP source passages).
        use_hybrid: Whether to use hybrid search (BM25 + Dense + RRF) when
            available. Defaults to True. Falls back to dense-only if hybrid
            is not available.
        enable_reranking: Whether to enable cross-encoder reranking of results.
            Defaults to False. Only applies when hybrid search is used.

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

    # Try hybrid search if requested and available
    if use_hybrid:
        hybrid_service = _create_hybrid_service(embedder, store, enable_reranking)
        if hybrid_service is not None:
            try:
                # Extract workspace_id from PostgreSQLVectorStore if available
                workspace_id = None
                if hasattr(store, '_workspace_id_str'):
                    workspace_id = store._workspace_id_str
                elif hasattr(store, '_workspace_id'):
                    workspace_id = str(store._workspace_id)
                
                return hybrid_service.search_hybrid(
                    query=query,
                    k=top_k,
                    workspace_id=workspace_id,
                    source_type=source_type,
                    bm25_top_k=20,
                    dense_top_k=20,
                    hybrid_top_k=20,
                )
            except Exception as exc:
                # Log and fall back to dense-only search
                import logging
                logging.getLogger(__name__).warning(
                    "Hybrid search failed, falling back to dense-only: %s", exc
                )

    # Fallback to original dense-only vector search
    try:
        query_embeddings = embedder.embed([query])
    except Exception as exc:
        raise RuntimeError(f"Query embedding failed: {exc}") from exc

    if not query_embeddings:
        raise RuntimeError("Embedding provider returned no embeddings")

    query_embedding = query_embeddings[0]

    return store.search(query_embedding, k=top_k, source_type=source_type)


async def async_retrieve(
    query: str,
    embedder: EmbeddingProvider,
    store: VectorStore,
    top_k: int = 5,
    source_type: Optional[str] = None,
    use_hybrid: bool = True,
    enable_reranking: bool = False,
) -> List[SearchResult]:
    """Async version of retrieve using hybrid search when available.

    Args:
        query: The search query text.
        embedder: Embedding provider to generate query vector.
        store: Vector store to search.
        top_k: Maximum number of results to return (default: 5).
        source_type: Optional document category filter.
        use_hybrid: Whether to use hybrid search when available.
        enable_reranking: Whether to enable cross-encoder reranking.

    Returns:
        List of SearchResult objects sorted by similarity (highest first).

    Raises:
        ValueError: If query is blank or top_k is not positive.
        RuntimeError: If embedding fails.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # Try hybrid search if requested and available
    if use_hybrid:
        hybrid_service = _create_hybrid_service(embedder, store, enable_reranking)
        if hybrid_service is not None:
            try:
                # Extract workspace_id from PostgreSQLVectorStore if available
                workspace_id = None
                if hasattr(store, '_workspace_id_str'):
                    workspace_id = store._workspace_id_str
                elif hasattr(store, '_workspace_id'):
                    workspace_id = str(store._workspace_id)
                
                return await hybrid_service.async_search_hybrid(
                    query=query,
                    k=top_k,
                    workspace_id=workspace_id,
                    source_type=source_type,
                    bm25_top_k=20,
                    dense_top_k=20,
                    hybrid_top_k=20,
                )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "Async hybrid search failed, falling back to dense-only: %s", exc
                )

    # Fallback to original dense-only vector search
    try:
        import asyncio
        query_embeddings = await asyncio.to_thread(embedder.embed, [query])
    except Exception as exc:
        raise RuntimeError(f"Query embedding failed: {exc}") from exc

    if not query_embeddings:
        raise RuntimeError("Embedding provider returned no embeddings")

    query_embedding = query_embeddings[0]

    if hasattr(store, "async_search"):
        return await store.async_search(query_embedding, k=top_k, source_type=source_type)
    else:
        import asyncio
        return await asyncio.to_thread(store.search, query_embedding, top_k, source_type)