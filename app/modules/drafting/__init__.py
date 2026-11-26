"""Drafting module for M3: requirement-level evidence retrieval and draft generation."""

from app.modules.drafting.generation import (
    EvidenceCitation,
    RequirementDraft,
    UncertaintyLevel,
    generate_draft_for_requirement,
)
from app.modules.drafting.retrieval import (
    EvidencePassage,
    retrieve_evidence_for_requirement,
    retrieve_evidence_for_requirement_async,
)
from app.modules.drafting.service import (
    DraftingResult,
    draft_for_all_requirements,
    draft_for_requirement,
)

__all__ = [
    # Retrieval
    "EvidencePassage",
    "retrieve_evidence_for_requirement",
    "retrieve_evidence_for_requirement_async",
    # Generation
    "EvidenceCitation",
    "RequirementDraft",
    "UncertaintyLevel",
    "generate_draft_for_requirement",
    # Service
    "DraftingResult",
    "draft_for_all_requirements",
    "draft_for_requirement",
]