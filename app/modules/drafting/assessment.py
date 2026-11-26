"""Requirement-level compliance and gap assessment (M3.4).

Runs *after* draft generation and independent citation verification (M3.3). It
assigns a conservative, explainable compliance status to a requirement using
ONLY citations that passed verification. It does not retrieve evidence,
generate the draft, or re-verify citations; it reuses those results.

Status model
------------
``compliant``            Verified evidence adequately demonstrates the company
                         meets the requirement.
``partially_compliant``  Verified evidence demonstrates only partial coverage or
                         identifies limitations.
``non_compliant``        Verified evidence explicitly establishes the company
                         cannot meet the requirement.
``needs_review``         Evidence is insufficient, ambiguous, conflicting, or
                         otherwise cannot support a reliable conclusion.

Guarantees
----------
* Only successfully verified citations may support an assessment. Failed,
  fabricated, wrong-workspace, missing-workspace, or RFP-source citations never
  count.
* No verified citations always yields ``needs_review`` (never
  ``non_compliant``).
* Conflicting evidence is preserved as ``needs_review`` rather than forced into
  a definitive class.
* The original requirement, draft answer, citations, and verification results
  are left untouched; only assessment fields are added.

This is assessment only: no human-review workflow, export, dashboard, or agent
loop. The existing AI provider abstraction is reused for the structured
assessment; no new provider or direct model calls are introduced.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from app.ai.provider import AIProvider
from app.modules.drafting.generation import RequirementDraft, UncertaintyLevel

logger = logging.getLogger(__name__)


class ComplianceStatus(str, Enum):
    """Requirement-level compliance status."""

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"


# Statuses that require verified evidence to be asserted.
_EVIDENCE_BACKED_STATUSES = {
    ComplianceStatus.COMPLIANT,
    ComplianceStatus.PARTIALLY_COMPLIANT,
    ComplianceStatus.NON_COMPLIANT,
}


class ComplianceAssessmentModel(BaseModel):
    """Structured assessment response from the AI provider."""

    status: str = Field(..., description="One of: compliant|partially_compliant|non_compliant|needs_review")
    rationale: str = Field(..., description="Concise explanation of the assessment")


@dataclass
class ComplianceAssessment:
    """Requirement-level compliance assessment result.

    Attributes:
        status: The assigned :class:`ComplianceStatus`.
        rationale: Concise, explainable rationale.
        supporting_citation_count: Number of verified citations that may support
            the assessment.
        total_citation_count: Total citations the draft carried.
    """

    status: ComplianceStatus
    rationale: str
    supporting_citation_count: int
    total_citation_count: int


ASSESSMENT_SYSTEM_PROMPT = """You assess whether a company meets a single RFP requirement, using ONLY the verified evidence provided.

You are conservative and explainable. Rules:
- Use only the VERIFIED citations supplied. Never invent evidence.
- "compliant": verified evidence adequately demonstrates the requirement is met.
- "partially_compliant": verified evidence shows only partial coverage or limitations.
- "non_compliant": verified evidence explicitly establishes the requirement cannot be met.
- "needs_review": evidence is insufficient, ambiguous, or conflicting.
- Never conclude "non_compliant" merely because evidence is missing; use "needs_review".
- Preserve uncertainty and conflicts rather than forcing a definitive class.

Return JSON: {"status": "...", "rationale": "concise reason"}"""


def _build_assessment_prompt(
    requirement_key: str,
    requirement_text: str,
    draft_answer: Optional[str],
    uncertainty: UncertaintyLevel,
    verified_citations: List,
) -> str:
    """Build the user prompt using only verified citations."""
    parts = [
        f"REQUIREMENT ({requirement_key}):",
        requirement_text,
        "",
        f"DRAFT ANSWER: {draft_answer or '(none)'}",
        f"DRAFT UNCERTAINTY: {uncertainty.value}",
        "",
        "VERIFIED CITATIONS (the only evidence you may use):",
    ]
    if not verified_citations:
        parts.append("(none)")
    else:
        for i, v in enumerate(verified_citations, 1):
            parts.append(f"{i}. evidence_id={v.evidence_id}: {v.excerpt}")
    parts.append("")
    parts.append("Assess compliance using only the verified citations above.")
    return "\n".join(parts)


def _needs_review(rationale: str, supporting: int, total: int) -> ComplianceAssessment:
    return ComplianceAssessment(
        status=ComplianceStatus.NEEDS_REVIEW,
        rationale=rationale,
        supporting_citation_count=supporting,
        total_citation_count=total,
    )


def assess_requirement_compliance(
    draft: RequirementDraft,
    provider: Optional[AIProvider] = None,
) -> ComplianceAssessment:
    """Assign a compliance status to a requirement from verified evidence only.

    Args:
        draft: The generated draft (with ``citation_verifications`` populated).
        provider: Existing AI provider used for the structured assessment. When
            ``None`` or unavailable, the result is ``needs_review``.

    Returns:
        A :class:`ComplianceAssessment`. The draft is not mutated.
    """
    citations = draft.citations or []
    verifications = draft.citation_verifications or []
    verified = [v for v in verifications if v.verified]
    total = len(citations)

    # No verified evidence -> needs_review (never non_compliant).
    if not verified:
        return _needs_review(
            "No verified evidence supports a compliance determination.",
            supporting=0,
            total=total,
        )

    # Preserve conflicting evidence rather than forcing a definitive class.
    if draft.uncertainty == UncertaintyLevel.CONFLICTING:
        return _needs_review(
            "Conflicting evidence; a reliable compliance conclusion requires review.",
            supporting=len(verified),
            total=total,
        )

    if provider is None:
        return _needs_review(
            "Assessment provider unavailable; compliance could not be determined.",
            supporting=len(verified),
            total=total,
        )

    prompt = _build_assessment_prompt(
        requirement_key=draft.requirement_key,
        requirement_text=draft.requirement_text,
        draft_answer=draft.draft_answer,
        uncertainty=draft.uncertainty,
        verified_citations=verified,
    )

    try:
        response = provider.generate_structured(
            system_prompt=ASSESSMENT_SYSTEM_PROMPT,
            user_text=prompt,
            response_model=ComplianceAssessmentModel,
        )
        parsed = response.model_dump()
    except Exception as exc:  # provider/parse failure -> conservative
        logger.warning("Compliance assessment failed: %s", exc)
        return _needs_review(
            f"Assessment provider error: {type(exc).__name__}: {exc}",
            supporting=len(verified),
            total=total,
        )

    raw_status = str(parsed.get("status", "")).strip().lower()
    rationale = str(parsed.get("rationale", "")).strip()

    # A meaningful rationale is mandatory: never assert a definitive status
    # without an explanation.
    if not rationale:
        return _needs_review(
            "Assessment returned no rationale; defaulting to needs_review.",
            supporting=len(verified),
            total=total,
        )

    try:
        status = ComplianceStatus(raw_status)
    except ValueError:
        return _needs_review(
            f"Unrecognized assessment status '{raw_status}'; defaulting to needs_review.",
            supporting=len(verified),
            total=total,
        )

    # Defensive guard: an evidence-backed status requires verified evidence.
    if status in _EVIDENCE_BACKED_STATUSES and not verified:
        return _needs_review(
            "Assessment claimed a definitive status without verified evidence.",
            supporting=0,
            total=total,
        )

    # ``non_compliant`` requires explicit, adequate evidence. When the draft's
    # own uncertainty is weak/none, the evidence is not explicit enough to
    # establish that the requirement cannot be met -> needs_review.
    if status == ComplianceStatus.NON_COMPLIANT and draft.uncertainty in (
        UncertaintyLevel.WEAK,
        UncertaintyLevel.NONE,
    ):
        return _needs_review(
            "Verified evidence is too weak to establish non-compliance; needs review.",
            supporting=len(verified),
            total=total,
        )

    return ComplianceAssessment(
        status=status,
        rationale=rationale,
        supporting_citation_count=len(verified),
        total_citation_count=total,
    )