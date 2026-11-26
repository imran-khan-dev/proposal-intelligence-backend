"""Drafting service orchestrating evidence retrieval and draft generation.

Provides the main entry point for M3: for each extracted requirement,
retrieve company evidence and generate an evidence-backed draft response.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

from app.ai.embeddings import EmbeddingProvider
from app.ai.embedding_factory import get_embedding_provider
from app.core.workspace import resolve_workspace_id
from app.modules.drafting.generation import (
    RequirementDraft,
    UncertaintyLevel,
    generate_draft_for_requirement,
)
from app.modules.drafting.retrieval import (
    EvidencePassage,
    retrieve_evidence_for_requirement,
)
from app.modules.requirements.schema import RequirementRecord
from app.rag.store import VectorStore
from app.rag.store_factory import get_vector_store
from app.rag.stores.postgres_vector_store import PostgreSQLVectorStore

logger = logging.getLogger(__name__)


@dataclass
class DraftingResult:
    """Result of drafting for a single requirement."""

    requirement: RequirementRecord
    draft: RequirementDraft
    retrieval_query: str
    evidence_count: int


def _scoped_vector_store(workspace_id: uuid.UUID) -> VectorStore:
    """Return a workspace-scoped vector store for the company knowledge base."""
    store = get_vector_store()
    if isinstance(store, PostgreSQLVectorStore):
        store = PostgreSQLVectorStore(workspace_id=workspace_id)
    return store


async def draft_for_requirement(
    requirement: RequirementRecord,
    workspace_id: uuid.UUID,
    embedder: Optional[EmbeddingProvider] = None,
    store: Optional[VectorStore] = None,
    provider=None,  # AIProvider for draft generation
    top_k: int = 5,
) -> DraftingResult:
    """Retrieve evidence and generate a draft for a single requirement.

    This is the main M3 entry point for a single requirement.

    Args:
        requirement: The extracted requirement (from M2).
        workspace_id: Workspace UUID for scoping.
        embedder: Embedding provider (optional, uses singleton if None).
        store: Vector store (optional, creates workspace-scoped if None).
        provider: AI provider for draft generation (optional, uses RFP provider if None).
        top_k: Maximum evidence passages to retrieve.

    Returns:
        DraftingResult with the draft, evidence, and metadata.
    """
    # Resolve dependencies
    if embedder is None:
        embedder = get_embedding_provider()
    if store is None:
        store = _scoped_vector_store(workspace_id)

    # Build retrieval query (same logic as retrieval module)
    from app.modules.drafting.retrieval import _build_retrieval_query
    retrieval_query = _build_retrieval_query(
        requirement_text=requirement.text,
        section_title=requirement.section_title,
        section_path=requirement.section_path,
        category=requirement.category.value if hasattr(requirement.category, 'value') else str(requirement.category),
    )

    # Retrieve evidence
    passages = retrieve_evidence_for_requirement(
        requirement_text=requirement.text,
        workspace_id=workspace_id,
        embedder=embedder,
        store=store,
        section_title=requirement.section_title,
        section_path=requirement.section_path,
        category=requirement.category.value if hasattr(requirement.category, 'value') else str(requirement.category),
        top_k=top_k,
    )

    # Generate draft
    draft = generate_draft_for_requirement(
        requirement_text=requirement.text,
        requirement_key=requirement.requirement_key,
        requirement_id=uuid.UUID(requirement.requirement_key) if False else None,  # We'll fix this
        passages=passages,
        section_title=requirement.section_title,
        category=requirement.category.value if hasattr(requirement.category, 'value') else str(requirement.category),
        provider=provider,
    )

    return DraftingResult(
        requirement=requirement,
        draft=draft,
        retrieval_query=retrieval_query,
        evidence_count=len(passages),
    )


async def draft_for_all_requirements(
    requirements: List[RequirementRecord],
    workspace_id: uuid.UUID,
    embedder: Optional[EmbeddingProvider] = None,
    store: Optional[VectorStore] = None,
    provider=None,
    top_k: int = 5,
) -> List[DraftingResult]:
    """Generate drafts for all requirements in a document.

    Sequential execution (no background jobs). Each requirement is processed
    independently with its own retrieval and generation.

    Args:
        requirements: List of extracted requirements from M2.
        workspace_id: Workspace UUID for scoping.
        embedder: Embedding provider (optional).
        store: Vector store (optional).
        provider: AI provider for draft generation (optional).
        top_k: Maximum evidence passages per requirement.

    Returns:
        List of DraftingResult, one per requirement.
    """
    results = []
    for req in requirements:
        try:
            result = await draft_for_requirement(
                requirement=req,
                workspace_id=workspace_id,
                embedder=embedder,
                store=store,
                provider=provider,
                top_k=top_k,
            )
            results.append(result)
        except Exception as exc:
            logger.exception("Drafting failed for requirement %s: %s", req.requirement_key, exc)
            # Create a failed result
            from app.modules.drafting.generation import RequirementDraft, EvidenceCitation
            failed_draft = RequirementDraft(
                requirement_id=uuid.uuid4(),
                requirement_key=req.requirement_key,
                requirement_text=req.text,
                draft_answer=None,
                citations=[],
                evidence_found=False,
                uncertainty=UncertaintyLevel.NONE,
                uncertainty_notes=[f"Drafting error: {type(exc).__name__}: {exc}"],
                evidence_passages=[],
            )
            results.append(DraftingResult(
                requirement=req,
                draft=failed_draft,
                retrieval_query="",
                evidence_count=0,
            ))

    return results