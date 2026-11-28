"""Evidence retrieval service for requirement-level company knowledge lookup.

For each extracted requirement, builds a focused retrieval query and searches
the company knowledge base (workspace-scoped) for relevant evidence passages.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

from app.ai.embeddings import EmbeddingProvider
from app.ai.embedding_factory import get_embedding_provider
from app.core.workspace import resolve_workspace_id
from app.rag.retrieve import retrieve, async_retrieve
from app.rag.store import SearchResult, VectorStore
from app.rag.store_factory import get_vector_store
from app.rag.stores.postgres_vector_store import PostgreSQLVectorStore

logger = logging.getLogger(__name__)

# Document category that requirement drafting is allowed to retrieve from.
# RFP source passages (source_type ``"rfp"``) must never be used as evidence.
COMPANY_KNOWLEDGE_SOURCE_TYPE = "company_knowledge"


@dataclass
class EvidencePassage:
    """A retrieved evidence passage with score and source references."""

    # The text of the evidence passage
    text: str
    # Similarity score (0.0 to 1.0, higher is more similar)
    score: float
    # Rank in retrieval results (1-based)
    rank: int
    # Source document ID in company knowledge base
    source_doc_id: str
    # Original filename
    source_filename: Optional[str]
    # Chunk ID in the vector store
    chunk_id: Optional[str]
    # Chunk index within the document
    chunk_index: Optional[int]
    # Page number (1-based)
    page_number: Optional[int]
    # Character start offset (absolute, in normalized document text)
    character_start: Optional[int]
    # Character end offset (absolute, in normalized document text)
    character_end: Optional[int]
    # Section path (e.g., "3.2.1")
    section_path: Optional[str]
    # Full metadata from the chunk
    metadata: dict


def _scoped_vector_store(workspace_id: uuid.UUID) -> VectorStore:
    """Return a workspace-scoped vector store for the company knowledge base."""
    store = get_vector_store()
    if isinstance(store, PostgreSQLVectorStore):
        store = PostgreSQLVectorStore(workspace_id=workspace_id)
    return store


def _build_retrieval_query(
    requirement_text: str,
    section_title: Optional[str] = None,
    section_path: Optional[str] = None,
    category: Optional[str] = None,
) -> str:
    """Build a focused retrieval query from requirement text and context.

    Combines the requirement text with relevant section context to improve
    retrieval relevance for company knowledge documents.
    """
    parts: List[str] = []

    # Primary signal: the requirement text itself
    parts.append(requirement_text)

    # Secondary signal: section context
    if section_title:
        parts.append(f"Section: {section_title}")
    if section_path:
        parts.append(f"Path: {section_path}")
    if category:
        parts.append(f"Category: {category}")

    # Join with clear delimiters
    return " | ".join(parts)


def _search_result_to_passage(
    result: SearchResult, rank: int
) -> EvidencePassage:
    """Convert a SearchResult to an EvidencePassage with source references."""
    metadata = result.record.metadata
    return EvidencePassage(
        text=result.record.text,
        score=result.score,
        rank=rank,
        source_doc_id=metadata.get("doc_id", ""),
        source_filename=metadata.get("source_filename"),
        chunk_id=result.record.id,
        chunk_index=metadata.get("chunk_index"),
        page_number=metadata.get("page_number"),
        character_start=metadata.get("character_start"),
        character_end=metadata.get("character_end"),
        section_path=metadata.get("section_path"),
        metadata=metadata,
    )


def retrieve_evidence_for_requirement(
    requirement_text: str,
    workspace_id: uuid.UUID,
    embedder: EmbeddingProvider,
    store: VectorStore,
    section_title: Optional[str] = None,
    section_path: Optional[str] = None,
    category: Optional[str] = None,
    top_k: int = 5,
) -> List[EvidencePassage]:
    """Retrieve evidence passages from company knowledge for a requirement.

    Args:
        requirement_text: The requirement text to find evidence for.
        workspace_id: The workspace UUID (scopes the search).
        embedder: Embedding provider for query embedding.
        store: Workspace-scoped vector store.
        section_title: Optional section title for context.
        section_path: Optional section path for context.
        category: Optional requirement category for context.
        top_k: Maximum number of evidence passages to return.

    Returns:
        List of EvidencePassage objects sorted by score (highest first).
        Empty list if no evidence found or on error.

    Note:
        Only searches the company knowledge base. RFP source passages are
        NOT included - they are kept separate per M3 requirements.
    """
    if not requirement_text or not requirement_text.strip():
        logger.warning("Empty requirement text provided for retrieval")
        return []

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # Build focused query
    query = _build_retrieval_query(
        requirement_text=requirement_text,
        section_title=section_title,
        section_path=section_path,
        category=category,
    )

    try:
        # Retrieve using the existing retrieval pipeline. The source-type
        # filter guarantees only company knowledge is searched, independent of
        # document ID naming, so RFP source passages can never be returned.
        results = retrieve(
            query=query,
            embedder=embedder,
            store=store,
            top_k=top_k,
            source_type=COMPANY_KNOWLEDGE_SOURCE_TYPE,
        )

        # Convert to evidence passages with preserved source references
        passages = [
            _search_result_to_passage(result, rank=i + 1)
            for i, result in enumerate(results)
        ]

        logger.info(
            "Retrieved %d evidence passages for requirement (workspace=%s)",
            len(passages),
            workspace_id,
        )
        return passages

    except Exception as exc:
        logger.exception("Evidence retrieval failed for requirement: %s", exc)
        # Return empty list on error - caller can decide how to handle
        return []


async def retrieve_evidence_for_requirement_async(
    requirement_text: str,
    workspace_id: uuid.UUID,
    embedder: Optional[EmbeddingProvider] = None,
    store: Optional[VectorStore] = None,
    section_title: Optional[str] = None,
    section_path: Optional[str] = None,
    category: Optional[str] = None,
    top_k: int = 5,
) -> List[EvidencePassage]:
    """Async version of retrieve_evidence_for_requirement.

    Uses async hybrid search when available (PostgreSQLVectorStore) to avoid blocking.
    """
    if embedder is None:
        embedder = get_embedding_provider()
    if store is None:
        store = _scoped_vector_store(workspace_id)

    if not requirement_text or not requirement_text.strip():
        logger.warning("Empty requirement text provided for retrieval")
        return []

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # Build focused query
    query = _build_retrieval_query(
        requirement_text=requirement_text,
        section_title=section_title,
        section_path=section_path,
        category=category,
    )

    try:
        # Retrieve using the async hybrid retrieval pipeline.
        # The source-type filter guarantees only company knowledge is searched.
        results = await async_retrieve(
            query=query,
            embedder=embedder,
            store=store,
            top_k=top_k,
            source_type=COMPANY_KNOWLEDGE_SOURCE_TYPE,
            use_hybrid=True,
        )

        # Convert to evidence passages with preserved source references
        passages = [
            _search_result_to_passage(result, rank=i + 1)
            for i, result in enumerate(results)
        ]

        logger.info(
            "Retrieved %d evidence passages for requirement (workspace=%s)",
            len(passages),
            workspace_id,
        )
        return passages

    except Exception as exc:
        logger.exception("Evidence retrieval failed for requirement: %s", exc)
        # Return empty list on error - caller can decide how to handle
        return []