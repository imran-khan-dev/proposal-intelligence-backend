"""Draft generation service for requirement-level evidence-backed responses.

Generates draft responses for each requirement using only retrieved company
evidence. Distinguishes supported evidence from missing/weak evidence.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from app.ai.factory import get_rfp_provider
from app.ai.provider import AIProvider
from app.modules.drafting.retrieval import EvidencePassage, retrieve_evidence_for_requirement
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DraftResponseModel(BaseModel):
    """Pydantic model for structured draft response from LLM."""

    draft_answer: Optional[str] = Field(None, description="Generated draft answer")
    citations: List[dict] = Field(default_factory=list, description="Evidence citations")
    evidence_found: bool = Field(False, description="Whether evidence was found")
    uncertainty: str = Field("none", description="Uncertainty level")
    uncertainty_notes: List[str] = Field(default_factory=list, description="Uncertainty notes")


class UncertaintyLevel(str, Enum):
    """Uncertainty level for a draft response."""

    SUPPORTED = "supported"      # Strong evidence found
    WEAK = "weak"                # Some evidence but insufficient
    NONE = "none"                # No relevant evidence found
    CONFLICTING = "conflicting"  # Evidence contradicts or is inconsistent


@dataclass
class EvidenceCitation:
    """A citation linking draft text to an evidence passage."""

    evidence_id: str  # The evidence passage ID (chunk_id or generated)
    excerpt: str      # Exact excerpt from the evidence used
    score: float      # Similarity score of the evidence


@dataclass
class RequirementDraft:
    """A draft response for a single requirement with evidence citations."""

    requirement_id: uuid.UUID
    requirement_key: str
    requirement_text: str
    draft_answer: Optional[str]
    citations: List[EvidenceCitation]
    evidence_found: bool
    uncertainty: UncertaintyLevel
    uncertainty_notes: List[str]
    # The evidence passages used (for persistence)
    evidence_passages: List[EvidencePassage]


# System prompt for draft generation
DRAFT_SYSTEM_PROMPT = """You are a proposal writer generating evidence-backed responses to RFP requirements.

Your task: Given an RFP requirement and retrieved company knowledge passages, write a concise draft response that:
1. Uses ONLY the provided evidence passages - do not invent company capabilities
2. Cites specific evidence passages for every claim made
3. Clearly indicates when evidence is missing, weak, or conflicting

Rules:
- If evidence supports the requirement, write a compliant response citing the evidence.
- If evidence is weak/partial, write a partial response and note the gaps.
- If no relevant evidence exists, state "No supporting evidence found in company knowledge base."
- If evidence conflicts, note the conflict and do not make a definitive claim.
- Never hallucinate certifications, experience, commitments, or capabilities.
- Keep responses concise and professional.
- Do not include meta-commentary in the answer - output only the draft response.

Format your response as JSON with these fields:
{
  "draft_answer": "The draft response text, or null if no evidence",
  "citations": [
    {"evidence_id": "chunk_id", "excerpt": "exact text from evidence", "score": 0.85}
  ],
  "evidence_found": true/false,
  "uncertainty": "supported|weak|none|conflicting",
  "uncertainty_notes": ["Note about gaps or conflicts"]
}"""


def _format_evidence_for_prompt(passages: List[EvidencePassage]) -> str:
    """Format evidence passages for the LLM prompt."""
    if not passages:
        return "No evidence passages retrieved."

    lines = ["RETRIEVED EVIDENCE PASSAGES:"]
    for i, p in enumerate(passages, 1):
        lines.append(f"\n--- Evidence {i} (score: {p.score:.3f}) ---")
        lines.append(f"Source: {p.source_filename or p.source_doc_id}")
        if p.page_number:
            lines.append(f"Page: {p.page_number}")
        if p.section_path:
            lines.append(f"Section: {p.section_path}")
        lines.append(f"Text: {p.text}")
    return "\n".join(lines)


def _build_draft_prompt(
    requirement_text: str,
    requirement_key: str,
    section_title: Optional[str],
    category: Optional[str],
    passages: List[EvidencePassage],
) -> str:
    """Build the user prompt for draft generation."""
    parts = [
        f"REQUIREMENT ({requirement_key}):",
        requirement_text,
    ]
    if section_title:
        parts.append(f"Section: {section_title}")
    if category:
        parts.append(f"Category: {category}")
    parts.append("")
    parts.append(_format_evidence_for_prompt(passages))
    parts.append("""
Generate a draft response using ONLY the above evidence. Return JSON as specified.""")
    return "\n".join(parts)


def _ground_citations(
    raw_citations: list, passages: List[EvidencePassage]
) -> List[EvidenceCitation]:
    """Ground model-produced citations against the retrieved passages.

    A citation is kept only when BOTH of the following hold:

    * its ``evidence_id`` matches a retrieved passage's ``chunk_id`` (no
      fabricated evidence IDs), and
    * its ``excerpt`` is an exact substring of that passage's text (no
      invented quotations).

    The citation score is taken from the matched passage rather than the model
    so a fabricated score cannot be introduced. This is lightweight grounding
    against evidence already retrieved for this requirement, not independent
    citation verification (a later milestone).
    """
    by_chunk_id = {p.chunk_id: p for p in passages if p.chunk_id}

    grounded: List[EvidenceCitation] = []
    for c in raw_citations:
        if not isinstance(c, dict):
            continue
        evidence_id = c.get("evidence_id")
        excerpt = c.get("excerpt")
        if not evidence_id or not excerpt:
            continue
        passage = by_chunk_id.get(evidence_id)
        if passage is None:
            # Fabricated evidence ID - not among retrieved passages.
            continue
        if excerpt not in passage.text:
            # Fabricated quotation - not present in the retrieved passage.
            continue
        grounded.append(
            EvidenceCitation(
                evidence_id=evidence_id,
                excerpt=excerpt,
                score=float(passage.score),
            )
        )
    return grounded


def _parse_draft_response(response_text: str) -> dict:
    """Parse the LLM's JSON response, with fallback for malformed output."""
    import json

    # Try to extract JSON from the response
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to find JSON block
        import re
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        # Fallback: return a structured error response
        logger.warning("Failed to parse LLM draft response as JSON: %s", response_text)
        return {
            "draft_answer": None,
            "citations": [],
            "evidence_found": False,
            "uncertainty": "none",
            "uncertainty_notes": ["Failed to parse LLM response"],
        }


def generate_draft_for_requirement(
    requirement_text: str,
    requirement_key: str,
    requirement_id: uuid.UUID,
    passages: List[EvidencePassage],
    section_title: Optional[str] = None,
    category: Optional[str] = None,
    provider: Optional[AIProvider] = None,
) -> RequirementDraft:
    """Generate an evidence-backed draft for a single requirement.

    Args:
        requirement_text: The requirement text.
        requirement_key: Stable requirement identifier.
        requirement_id: Database UUID of the requirement.
        passages: Retrieved evidence passages (company knowledge only).
        section_title: Optional section title.
        category: Optional requirement category.
        provider: AI provider for generation. If None, uses get_rfp_provider().

    Returns:
        RequirementDraft with answer, citations, and uncertainty assessment.

    Note:
        Does not assign compliance status - that is a later milestone.
        Does not verify citations independently - that is a later milestone.
    """
    if provider is None:
        try:
            provider = get_rfp_provider()
        except Exception as exc:
            logger.warning("AI provider unavailable for draft generation: %s", exc)
            return RequirementDraft(
                requirement_id=requirement_id,
                requirement_key=requirement_key,
                requirement_text=requirement_text,
                draft_answer=None,
                citations=[],
                evidence_found=False,
                uncertainty=UncertaintyLevel.NONE,
                uncertainty_notes=["AI provider unavailable"],
                evidence_passages=passages,
            )

    if not passages:
        # No evidence found - return evidence-gap response
        return RequirementDraft(
            requirement_id=requirement_id,
            requirement_key=requirement_key,
            requirement_text=requirement_text,
            draft_answer="No supporting evidence found in company knowledge base.",
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=["No relevant company knowledge passages retrieved"],
            evidence_passages=passages,
        )

    # Build prompt and generate
    prompt = _build_draft_prompt(
        requirement_text=requirement_text,
        requirement_key=requirement_key,
        section_title=section_title,
        category=category,
        passages=passages,
    )

    try:
        response = provider.generate_structured(
            system_prompt=DRAFT_SYSTEM_PROMPT,
            user_text=prompt,
            response_model=DraftResponseModel,
        )

        parsed = _parse_draft_response(response.model_dump_json())

        # Ground citations against the retrieved passages: keep only citations
        # whose evidence_id is a retrieved chunk and whose excerpt is an exact
        # substring of that passage (drops fabricated IDs and quotations).
        citations = _ground_citations(parsed.get("citations", []), passages)

        # Map uncertainty string to enum
        uncertainty_str = parsed.get("uncertainty", "none").lower()
        try:
            uncertainty = UncertaintyLevel(uncertainty_str)
        except ValueError:
            uncertainty = UncertaintyLevel.NONE

        # Determine if evidence was actually found and used
        evidence_found = parsed.get("evidence_found", False) and len(citations) > 0

        draft_answer = parsed.get("draft_answer")
        if not evidence_found:
            draft_answer = "No supporting evidence found in company knowledge base."

        return RequirementDraft(
            requirement_id=requirement_id,
            requirement_key=requirement_key,
            requirement_text=requirement_text,
            draft_answer=draft_answer,
            citations=citations,
            evidence_found=evidence_found,
            uncertainty=uncertainty,
            uncertainty_notes=parsed.get("uncertainty_notes", []),
            evidence_passages=passages,
        )

    except Exception as exc:
        logger.exception("Draft generation failed for requirement %s: %s", requirement_key, exc)
        return RequirementDraft(
            requirement_id=requirement_id,
            requirement_key=requirement_key,
            requirement_text=requirement_text,
            draft_answer=None,
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=[f"Generation error: {type(exc).__name__}: {exc}"],
            evidence_passages=passages,
        )