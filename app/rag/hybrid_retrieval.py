"""Hybrid retrieval service using Retrieval Stack components with PostgreSQL backend.

This module adapts the Retrieval Stack's BM25, Hybrid Search (RRF), and Reranking
components to work with the PostgreSQL vector store, preserving workspace isolation
and source-type filtering at the database level.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, List, Optional, Sequence

from retrieval_stack.bm25 import BM25Searcher, InvertedIndex, Tokenizer as BM25Tokenizer
from retrieval_stack.chunking.models import Chunk
from retrieval_stack.hybrid import HybridSearcher, HybridSearchResult, reciprocal_rank_fusion
from retrieval_stack.reranking import Reranker, RerankerProvider, SentenceTransformerCrossEncoder

from app.ai.embeddings import EmbeddingProvider
from app.rag.store import SearchResult, VectorRecord, VectorStore
from app.rag.stores.postgres_vector_store import PostgreSQLVectorStore

logger = logging.getLogger(__name__)


def _vector_record_to_chunk(record: VectorRecord) -> Chunk:
    """Convert a VectorRecord to a Retrieval Stack Chunk."""
    metadata = dict(record.metadata)
    doc_id = metadata.pop("doc_id", record.id)
    chunk_index = metadata.pop("chunk_index", 0)
    return Chunk(
        id=record.id,
        document_id=doc_id,
        index=chunk_index,
        content=record.text,
        token_count=len(record.text.split()),
        metadata=metadata,
        start_char=metadata.get("character_start"),
        end_char=metadata.get("character_end"),
    )


class HybridRetrievalService:
    """Hybrid retrieval combining BM25 keyword search with dense vector search.

    Uses Retrieval Stack components for BM25 indexing, RRF fusion, and reranking,
    while leveraging PostgreSQL for persistence, workspace isolation, and
    source-type filtering.

    The service builds an in-memory BM25 index from PostgreSQL data on demand
    (or caches it) and combines it with pgvector dense search via RRF.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        reranker_provider: Optional[RerankerProvider] = None,
        bm25_tokenizer: Optional[BM25Tokenizer] = None,
        cache_bm25_index: bool = True,
    ):
        """Initialize the hybrid retrieval service.

        Args:
            store: Vector store (must be PostgreSQLVectorStore for full functionality).
            embedder: Embedding provider for query embedding.
            reranker_provider: Optional reranker provider. If None, no reranking is applied.
            bm25_tokenizer: Optional BM25 tokenizer. Uses default if not provided.
            cache_bm25_index: Whether to cache the BM25 index between searches.
                If True, the index is built once and reused. If False, rebuilt each search.
        """
        self._store = store
        self._embedder = embedder
        self._bm25_tokenizer = bm25_tokenizer or BM25Tokenizer()
        self._cache_bm25_index = cache_bm25_index

        self._bm25_index: Optional[InvertedIndex] = None
        self._bm25_searcher: Optional[BM25Searcher] = None
        self._dense_searcher = None  # Uses PostgreSQL directly
        self._hybrid_searcher: Optional[HybridSearcher] = None
        self._reranker: Optional[Reranker] = None
        self._indexed_workspace_id: Optional[str] = None
        self._indexed_source_type: Optional[str] = None

        if reranker_provider is not None:
            self._reranker = Reranker(reranker_provider)

    def _ensure_bm25_index(
        self,
        workspace_id: Optional[str] = None,
        source_type: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> None:
        """Build or refresh the BM25 index from PostgreSQL data.

        Args:
            workspace_id: Workspace to scope the index to.
            source_type: Source type filter (e.g., "company_knowledge").
            force_rebuild: Force rebuilding the index even if cached.
        """
        # Check if we can reuse cached index
        if (
            not force_rebuild
            and self._cache_bm25_index
            and self._bm25_index is not None
            and self._indexed_workspace_id == workspace_id
            and self._indexed_source_type == source_type
        ):
            return

        # Fetch records from PostgreSQL
        if not isinstance(self._store, PostgreSQLVectorStore):
            logger.warning(
                "HybridRetrievalService: BM25 index only supported with PostgreSQLVectorStore. "
                "Falling back to dense-only search."
            )
            self._bm25_index = None
            self._bm25_searcher = None
            self._hybrid_searcher = None
            return

        # Use the store's internal method to get all chunks for the workspace/source_type
        # We need to access the database directly to fetch all chunks
        import asyncio
        import uuid
        from sqlalchemy import select

        from app.core.database import get_async_session_factory
        from app.models.document import Document
        from app.models.document_chunk import DocumentChunk

        async def _fetch_chunks() -> List[VectorRecord]:
            factory = get_async_session_factory()
            target_workspace_id = None
            if workspace_id:
                try:
                    target_workspace_id = uuid.UUID(workspace_id) if isinstance(workspace_id, str) else workspace_id
                except (ValueError, TypeError):
                    target_workspace_id = None

            async with factory() as session:
                stmt = select(DocumentChunk).join(Document, DocumentChunk.document_id == Document.id)
                if target_workspace_id:
                    stmt = stmt.where(Document.workspace_id == target_workspace_id)
                if source_type:
                    stmt = stmt.where(Document.source_type == source_type)

                result = await session.execute(stmt)
                chunks = result.scalars().all()

                records = []
                for chunk in chunks:
                    metadata = dict(chunk.chunk_metadata or {})
                    if chunk.workspace_id and "workspace_id" not in metadata:
                        metadata["workspace_id"] = str(chunk.workspace_id)
                    records.append(
                        VectorRecord(
                            id=str(chunk.id),
                            text=chunk.text,
                            embedding=list(chunk.embedding),
                            metadata=metadata,
                        )
                    )
                return records

        try:
            # Run async fetch in sync context
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                records = asyncio.run(_fetch_chunks())
            else:
                # We're in an async context - can't run async here
                # This should not happen in normal FastAPI operation if callers use async methods
                raise RuntimeError(
                    "BM25 index build requires async context. "
                    "Use async_search_hybrid() instead."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            logger.exception("Failed to fetch chunks for BM25 index: %s", exc)
            self._bm25_index = None
            self._bm25_searcher = None
            self._hybrid_searcher = None
            return

        if not records:
            self._bm25_index = InvertedIndex(tokenizer=self._bm25_tokenizer)
            self._bm25_searcher = BM25Searcher(self._bm25_index)
            self._hybrid_searcher = None  # No BM25 data, hybrid not possible
            return

        # Build BM25 index from records
        chunks = [_vector_record_to_chunk(r) for r in records]
        self._bm25_index = InvertedIndex(tokenizer=self._bm25_tokenizer)
        self._bm25_index.build(chunks)
        self._bm25_searcher = BM25Searcher(self._bm25_index)

        # Create hybrid searcher (dense search will use PostgreSQL directly)
        self._hybrid_searcher = HybridSearcher(
            bm25_searcher=self._bm25_searcher,
            dense_searcher=None,  # We'll handle dense search via PostgreSQL
            rrf_k=60,
        )

        self._indexed_workspace_id = workspace_id
        self._indexed_source_type = source_type

        logger.info(
            "Built BM25 index with %d chunks (workspace=%s, source_type=%s)",
            len(chunks),
            workspace_id,
            source_type,
        )

    def search_hybrid(
        self,
        query: str,
        k: int = 5,
        workspace_id: Optional[str] = None,
        source_type: Optional[str] = None,
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        hybrid_top_k: int = 20,
        force_rebuild_index: bool = False,
    ) -> List[SearchResult]:
        """Perform hybrid search combining BM25 and dense vector search.

        Args:
            query: Search query text.
            k: Number of final results to return.
            workspace_id: Workspace to search within.
            source_type: Source type filter (e.g., "company_knowledge").
            bm25_top_k: Number of BM25 candidates to retrieve.
            dense_top_k: Number of dense candidates to retrieve from PostgreSQL.
            hybrid_top_k: Number of hybrid candidates before final top-k.
            force_rebuild_index: Force rebuilding the BM25 index.

        Returns:
            List of SearchResult objects sorted by fused score (highest first).
        """
        if k <= 0:
            return []

        # Ensure BM25 index is built
        self._ensure_bm25_index(
            workspace_id=workspace_id,
            source_type=source_type,
            force_rebuild=force_rebuild_index,
        )

        # If no BM25 index (no data or non-PostgreSQL store), fall back to dense-only
        if self._bm25_searcher is None or self._bm25_index.total_docs == 0:
            logger.debug("No BM25 index available, falling back to dense-only search")
            return self._store.search(
                query_embedding=self._embedder.embed([query])[0],
                k=k,
                source_type=source_type,
            )

        # BM25 search
        bm25_results = self._bm25_searcher.search(query, top_k=bm25_top_k)

        # Dense search via PostgreSQL
        query_embedding = self._embedder.embed([query])[0]
        dense_results = self._store.search(
            query_embedding=query_embedding,
            k=dense_top_k,
            source_type=source_type,
        )

        # Convert dense results to HybridSearchResult-like objects for RRF
        dense_search_results = []
        for rank, result in enumerate(dense_results, 1):
            chunk = _vector_record_to_chunk(result.record)
            dense_search_results.append(
                HybridSearchResult(
                    chunk_id=result.record.id,
                    chunk=chunk,
                    rrf_score=0.0,  # Will be computed by RRF
                    sources=["dense"],
                    dense_rank=rank,
                    metadata=dict(result.record.metadata),
                )
            )

        # Convert BM25 results
        bm25_search_results = []
        for rank, result in enumerate(bm25_results, 1):
            bm25_metadata = dict(result.metadata)
            bm25_metadata["doc_id"] = result.chunk.document_id
            bm25_search_results.append(
                HybridSearchResult(
                    chunk_id=result.chunk_id,
                    chunk=result.chunk,
                    rrf_score=0.0,
                    sources=["bm25"],
                    bm25_rank=rank,
                    metadata=bm25_metadata,
                )
            )

        # Apply RRF fusion
        fused = reciprocal_rank_fusion(
            [bm25_search_results, dense_search_results],
            source_names=["bm25", "dense"],
            k=60,
        )

        # Take top hybrid_top_k for potential reranking
        candidates = fused[:hybrid_top_k]

        # Rerank if reranker is available
        if self._reranker is not None and candidates:
            reranked = self._reranker.rerank(query, candidates, top_k=k, candidate_count=hybrid_top_k)
        else:
            reranked = candidates[:k]

        # Convert back to SearchResult
        search_results = []
        for item in reranked:
            # Preserve metadata and add doc_id from chunk.document_id
            metadata = dict(item.metadata)
            metadata["doc_id"] = item.chunk.document_id
            record = VectorRecord(
                id=item.chunk_id,
                text=item.chunk.content,
                embedding=[],  # Not needed for results
                metadata=metadata,
            )
            # Score: use relevance_score if reranked, else rrf_score
            score = getattr(item, "relevance_score", item.rrf_score)
            search_results.append(SearchResult(record=record, score=float(score)))

        return search_results

    async def async_search_hybrid(
        self,
        query: str,
        k: int = 5,
        workspace_id: Optional[str] = None,
        source_type: Optional[str] = None,
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        hybrid_top_k: int = 20,
        force_rebuild_index: bool = False,
    ) -> List[SearchResult]:
        """Async version of hybrid search.

        Args:
            query: Search query text.
            k: Number of final results to return.
            workspace_id: Workspace to search within.
            source_type: Source type filter (e.g., "company_knowledge").
            bm25_top_k: Number of BM25 candidates to retrieve.
            dense_top_k: Number of dense candidates to retrieve from PostgreSQL.
            hybrid_top_k: Number of hybrid candidates before final top-k.
            force_rebuild_index: Force rebuilding the BM25 index.

        Returns:
            List of SearchResult objects sorted by fused score (highest first).
        """
        if k <= 0:
            return []

        # Ensure BM25 index is built (async version)
        await self._async_ensure_bm25_index(
            workspace_id=workspace_id,
            source_type=source_type,
            force_rebuild=force_rebuild_index,
        )

        # If no BM25 index, fall back to dense-only
        if self._bm25_searcher is None or self._bm25_index.total_docs == 0:
            logger.debug("No BM25 index available, falling back to dense-only search")
            query_embeddings = await self._embedder.async_embed([query]) if hasattr(self._embedder, "async_embed") else self._embedder.embed([query])
            return await self._store.async_search(
                query_embedding=query_embeddings[0],
                k=k,
                source_type=source_type,
            )

        # BM25 search (sync, fast)
        bm25_results = self._bm25_searcher.search(query, top_k=bm25_top_k)

        # Dense search via PostgreSQL (async)
        # Run sync embed in thread pool to avoid blocking event loop
        import asyncio
        query_embeddings = await asyncio.to_thread(self._embedder.embed, [query])
        dense_results = await self._store.async_search(
            query_embedding=query_embeddings[0],
            k=dense_top_k,
            source_type=source_type,
        )

        # Convert to hybrid results
        dense_search_results = []
        for rank, result in enumerate(dense_results, 1):
            chunk = _vector_record_to_chunk(result.record)
            dense_search_results.append(
                HybridSearchResult(
                    chunk_id=result.record.id,
                    chunk=chunk,
                    rrf_score=0.0,
                    sources=["dense"],
                    dense_rank=rank,
                    metadata=dict(result.record.metadata),
                )
            )

        bm25_search_results = []
        for rank, result in enumerate(bm25_results, 1):
            bm25_metadata = dict(result.metadata)
            bm25_metadata["doc_id"] = result.chunk.document_id
            bm25_search_results.append(
                HybridSearchResult(
                    chunk_id=result.chunk_id,
                    chunk=result.chunk,
                    rrf_score=0.0,
                    sources=["bm25"],
                    bm25_rank=rank,
                    metadata=bm25_metadata,
                )
            )

        # Apply RRF fusion
        fused = reciprocal_rank_fusion(
            [bm25_search_results, dense_search_results],
            source_names=["bm25", "dense"],
            k=60,
        )

        candidates = fused[:hybrid_top_k]

        # Rerank if available
        if self._reranker is not None and candidates:
            reranked = self._reranker.rerank(query, candidates, top_k=k, candidate_count=hybrid_top_k)
        else:
            reranked = candidates[:k]

        # Convert to SearchResult
        search_results = []
        for item in reranked:
            # Preserve metadata and add doc_id from chunk.document_id
            metadata = dict(item.metadata)
            metadata["doc_id"] = item.chunk.document_id
            record = VectorRecord(
                id=item.chunk_id,
                text=item.chunk.content,
                embedding=[],
                metadata=metadata,
            )
            score = getattr(item, "relevance_score", item.rrf_score)
            search_results.append(SearchResult(record=record, score=float(score)))

        return search_results

    async def _async_ensure_bm25_index(
        self,
        workspace_id: Optional[str] = None,
        source_type: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> None:
        """Build or refresh the BM25 index from PostgreSQL data (async)."""
        if (
            not force_rebuild
            and self._cache_bm25_index
            and self._bm25_index is not None
            and self._indexed_workspace_id == workspace_id
            and self._indexed_source_type == source_type
        ):
            return

        if not isinstance(self._store, PostgreSQLVectorStore):
            logger.warning(
                "HybridRetrievalService: BM25 index only supported with PostgreSQLVectorStore. "
                "Falling back to dense-only search."
            )
            self._bm25_index = None
            self._bm25_searcher = None
            self._hybrid_searcher = None
            return

        import uuid
        from sqlalchemy import select

        from app.core.database import get_async_session_factory
        from app.models.document import Document
        from app.models.document_chunk import DocumentChunk

        factory = get_async_session_factory()
        target_workspace_id = None
        if workspace_id:
            try:
                target_workspace_id = uuid.UUID(workspace_id) if isinstance(workspace_id, str) else workspace_id
            except (ValueError, TypeError):
                target_workspace_id = None

        async with factory() as session:
            stmt = select(DocumentChunk).join(Document, DocumentChunk.document_id == Document.id)
            if target_workspace_id:
                stmt = stmt.where(Document.workspace_id == target_workspace_id)
            if source_type:
                stmt = stmt.where(Document.source_type == source_type)

            result = await session.execute(stmt)
            chunks = result.scalars().all()

            records = []
            for chunk in chunks:
                metadata = dict(chunk.chunk_metadata or {})
                if chunk.workspace_id and "workspace_id" not in metadata:
                    metadata["workspace_id"] = str(chunk.workspace_id)
                records.append(
                    VectorRecord(
                        id=str(chunk.id),
                        text=chunk.text,
                        embedding=list(chunk.embedding),
                        metadata=metadata,
                    )
                )

        if not records:
            self._bm25_index = InvertedIndex(tokenizer=self._bm25_tokenizer)
            self._bm25_searcher = BM25Searcher(self._bm25_index)
            self._hybrid_searcher = None
            return

        chunks_rs = [_vector_record_to_chunk(r) for r in records]
        self._bm25_index = InvertedIndex(tokenizer=self._bm25_tokenizer)
        self._bm25_index.build(chunks_rs)
        self._bm25_searcher = BM25Searcher(self._bm25_index)
        self._hybrid_searcher = HybridSearcher(
            bm25_searcher=self._bm25_searcher,
            dense_searcher=None,
            rrf_k=60,
        )

        self._indexed_workspace_id = workspace_id
        self._indexed_source_type = source_type

        logger.info(
            "Built BM25 index with %d chunks (workspace=%s, source_type=%s)",
            len(chunks_rs),
            workspace_id,
            source_type,
        )

    def invalidate_index(self) -> None:
        """Invalidate the cached BM25 index, forcing rebuild on next search."""
        self._bm25_index = None
        self._bm25_searcher = None
        self._hybrid_searcher = None
        self._indexed_workspace_id = None
        self._indexed_source_type = None


# Module-level cache of HybridRetrievalService instances
# Key: (store_id, workspace_id, source_type, enable_reranking)
_service_cache: dict[tuple, HybridRetrievalService] = {}
_cache_lock = threading.Lock()


def _get_cache_key(
    store: VectorStore,
    workspace_id: Optional[str],
    source_type: Optional[str],
    enable_reranking: bool,
) -> tuple:
    """Generate a cache key for the service."""
    store_id = id(store)
    return (store_id, workspace_id, source_type, enable_reranking)


def get_or_create_hybrid_service(
    embedder: EmbeddingProvider,
    store: VectorStore,
    workspace_id: Optional[str] = None,
    source_type: Optional[str] = None,
    enable_reranking: bool = False,
) -> Optional[HybridRetrievalService]:
    """Get or create a cached hybrid retrieval service.

    Returns None if the store is not a PostgreSQLVectorStore (hybrid search
    requires PostgreSQL for BM25 index building).
    """
    if not isinstance(store, PostgreSQLVectorStore):
        return None

    key = _get_cache_key(store, workspace_id, source_type, enable_reranking)

    with _cache_lock:
        if key in _service_cache:
            return _service_cache[key]

        service = create_hybrid_retrieval_service(
            store=store,
            embedder=embedder,
            enable_reranking=enable_reranking,
            cache_bm25_index=True,
        )
        _service_cache[key] = service
        return service


def invalidate_hybrid_cache(
    workspace_id: Optional[str] = None,
    source_type: Optional[str] = None,
) -> None:
    """Invalidate cached BM25 indexes for the given workspace/source_type.

    This should be called after document ingestion, deletion, or update
    to ensure the next search rebuilds the index with the latest data.

    Args:
        workspace_id: Workspace to invalidate. If None, invalidates all workspaces.
        source_type: Source type to invalidate. If None, invalidates all source types.
    """
    with _cache_lock:
        keys_to_remove = []
        for key, service in _service_cache.items():
            cached_workspace_id = key[1]
            cached_source_type = key[2]
            if (workspace_id is None or cached_workspace_id == workspace_id) and \
               (source_type is None or cached_source_type == source_type):
                service.invalidate_index()
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del _service_cache[key]
        if keys_to_remove:
            logger.info("Invalidated hybrid retrieval cache for workspace=%s, source_type=%s", workspace_id, source_type)


def create_hybrid_retrieval_service(
    store: VectorStore,
    embedder: EmbeddingProvider,
    enable_reranking: bool = False,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    cache_bm25_index: bool = True,
) -> HybridRetrievalService:
    """Factory function to create a HybridRetrievalService with optional reranking.

    Args:
        store: Vector store (PostgreSQLVectorStore recommended).
        embedder: Embedding provider.
        enable_reranking: Whether to enable cross-encoder reranking.
        reranker_model: Model name for SentenceTransformerCrossEncoder.
        cache_bm25_index: Whether to cache the BM25 index.

    Returns:
        Configured HybridRetrievalService instance.
    """
    reranker_provider = None
    if enable_reranking:
        try:
            reranker_provider = SentenceTransformerCrossEncoder(reranker_model)
        except Exception as exc:
            logger.warning("Failed to initialize reranker: %s. Proceeding without reranking.", exc)

    return HybridRetrievalService(
        store=store,
        embedder=embedder,
        reranker_provider=reranker_provider,
        cache_bm25_index=cache_bm25_index,
    )