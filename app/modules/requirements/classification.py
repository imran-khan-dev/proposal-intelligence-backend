"""Hybrid requirement classification.

Two independently testable stages:

1. Rule-based signals (``classify_by_rules``): deterministic keyword/pattern
   matching for modality (mandatory / optional / informational / uncertain)
   and category.  No network or model required.
2. LLM-assisted fallback (``classify_with_llm``): used only when the
   rule-based pass is uncertain.  Delegates to the existing AI provider
   abstraction and is fully mockable.

Uncertainty is preserved rather than forced: when rules are inconclusive and
no LLM is available (or the LLM fails), the candidate stays ``uncertain`` with
an explanatory note.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel, Field

from app.ai.provider import AIProvider


class RequirementModality(str, Enum):
    """How binding a requirement is."""

    MANDATORY = "mandatory"
    OPTIONAL = "optional"
    INFORMATIONAL = "informational"
    UNCERTAIN = "uncertain"


class RequirementCategory(str, Enum):
    """High-level category of a requirement."""

    FUNCTIONAL = "functional"
    TECHNICAL = "technical"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    INTEGRATION = "integration"
    IMPLEMENTATION = "implementation"
    SUPPORT_MAINTENANCE = "support_and_maintenance"
    VENDOR_QUALIFICATIONS = "vendor_qualifications"
    SUBMISSION = "submission"
    OTHER = "other"
    UNCERTAIN = "uncertain"


class RuleClassification(BaseModel):
    """Result of the rule-based classifier."""

    modality: RequirementModality
    category: RequirementCategory
    confidence: float
    notes: List[str] = Field(default_factory=list)
    is_requirement: bool = Field(
        True, description="Whether the text reads as a requirement at all"
    )


class LLMClassification(BaseModel):
    """Structured response expected from the LLM classifier."""

    is_requirement: bool = Field(
        ..., description="True if the text is an actual requirement"
    )
    modality: RequirementModality = Field(..., description="Bindingness")
    category: RequirementCategory = Field(..., description="Category")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the classification"
    )
    notes: List[str] = Field(
        default_factory=list, description="Optional clarifying notes"
    )


class _ClassificationsResponse(BaseModel):
    """Batch wrapper returned by the LLM for multiple candidates."""

    items: List[LLMClassification] = Field(default_factory=list)


T = TypeVar("T", bound=BaseModel)


# --- Rule lexicons ---------------------------------------------------------

_MANDATORY_PATTERNS = [
    r"\bmust\b",
    r"\bshall\b",
    r"\bis required\b",
    r"\bare required\b",
    r"\brequired to\b",
    r"\brequirement that\b",
    r"\bmandatory\b",
    r"\bmust not\b",
    r"\bshall not\b",
    r"\bwill be required\b",
    r"\bhas to\b",
    r"\bhave to\b",
    r"\bneeds? to\b",
    r"\bis responsible for\b",
    r"\bwill provide\b",
    r"\bwill be responsible\b",
]

_OPTIONAL_PATTERNS = [
    r"\bmay\b",
    r"\bshould\b",
    r"\bmight\b",
    r"\bcan\b",
    r"\bcould\b",
    r"\boptional\b",
    r"\bif desired\b",
    r"\bat the (?:contractor'?s|vendor'?s|responder'?s) discretion\b",
    r"\bpreferably\b",
    r"\bwhere possible\b",
    r"\brecommended\b",
    r"\bdesirable\b",
    r"\bnot required\b",
]

# Category keyword -> patterns.  Order matters only for tie-breaking; the
# category with the most distinct pattern hits wins.
_CATEGORY_PATTERNS: dict[RequirementCategory, list[str]] = {
    RequirementCategory.SECURITY: [
        r"\bsecurity\b", r"\bauthentication\b", r"\bauthorization\b",
        r"\bencryption\b", r"\bsso\b", r"\bsaml\b", r"\boauth\b",
        r"\bvulnerabilit", r"\bpen(?:etration)? test", r"\bsoc ?2\b",
        r"\biso ?27001\b", r"\baccess control\b", r"\bpii\b",
        r"\bdata protection\b", r"\bsecure\b", r"\bcyber",
    ],
    RequirementCategory.COMPLIANCE: [
        r"\bcomplian", r"\bregulat", r"\bhipaa\b", r"\bferpa\b",
        r"\bgdpr\b", r"\bccpa\b", r"\bwcag\b", r"\bada\b",
        r"\bsection 508\b", r"\baudit\b", r"\bcertif", r"\bstatut",
    ],
    RequirementCategory.INTEGRATION: [
        r"\bintegrat", r"\binterface\b", r"\binteroperab",
        r"\bdata exchange\b", r"\bsftp\b", r"\bedi\b", r"\bconnect to\b",
        r"\bthird[- ]party\b", r"\bweb ?service\b", r"\brest\b",
    ],
    RequirementCategory.TECHNICAL: [
        r"\btechnical\b", r"\barchitect", r"\binfrastructur",
        r"\bcloud\b", r"\bdatabase\b", r"\blatency\b", r"\buptime\b",
        r"\bsla\b", r"\bperformance\b", r"\bscalab", r"\bstorage\b",
        r"\bnetwork\b", r"\bbandwidth\b", r"\bavailability\b",
        r"\bplatform\b", r"\bserver\b",
    ],
    RequirementCategory.FUNCTIONAL: [
        r"\bfunctional\b", r"\bfeature\b", r"\bcapabilit",
        r"\buser interface\b", r"\bworkflow\b", r"\breport\b",
        r"\bdashboard\b", r"\bsearch\b", r"\bmodule\b",
    ],
    RequirementCategory.IMPLEMENTATION: [
        r"\bimplementation\b", r"\bdeployment\b", r"\bmigration\b",
        r"\btimeline\b", r"\bproject plan\b", r"\btraining\b",
        r"\bgo[- ]live\b", r"\brollout\b", r"\bonboarding\b",
    ],
    RequirementCategory.SUPPORT_MAINTENANCE: [
        r"\bsupport\b", r"\bmaintenance\b", r"\bwarranty\b",
        r"\bhelp ?desk\b", r"\bservice desk\b", r"\bpatch\b",
        r"\bmonitoring\b", r"\b24/7\b", r"\bservice level\b",
    ],
    RequirementCategory.VENDOR_QUALIFICATIONS: [
        r"\bvendor\b", r"\bcontractor\b", r"\bqualification",
        r"\bexperience\b", r"\breferences?\b", r"\bpast performance\b",
        r"\binsurance\b", r"\bbonding\b", r"\byears in business\b",
        r"\bproposer\b",
    ],
    RequirementCategory.SUBMISSION: [
        r"\bsubmit\b", r"\bsubmission\b", r"\bproposal\b",
        r"\bdeadline\b", r"\bdue date\b", r"\bformat\b", r"\bportal\b",
        r"\bcontact\b", r"\bsignature\b", r"\bappendix\b",
        r"\battachment\b", r"\bexhibit\b",
    ],
}

_COMPILED_MANDATORY = [re.compile(p, re.IGNORECASE) for p in _MANDATORY_PATTERNS]
_COMPILED_OPTIONAL = [re.compile(p, re.IGNORECASE) for p in _OPTIONAL_PATTERNS]
_COMPILED_CATEGORY = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in _CATEGORY_PATTERNS.items()
}


def _count_hits(patterns: list[re.Pattern], text: str) -> int:
    return sum(1 for p in patterns if p.search(text))


def classify_category_by_rules(text: str) -> tuple[RequirementCategory, float]:
    """Return the best category and a confidence in ``[0, 1]``."""
    scores = {
        cat: _count_hits(pats, text) for cat, pats in _COMPILED_CATEGORY.items()
    }
    best_cat = max(scores, key=lambda c: scores[c])
    best = scores[best_cat]
    if best == 0:
        return RequirementCategory.UNCERTAIN, 0.3
    total = sum(scores.values())
    confidence = 0.5 + 0.5 * (best / total) if total else 0.3
    return best_cat, min(confidence, 0.95)


def classify_by_rules(text: str) -> RuleClassification:
    """Deterministic rule-based modality + category classification."""
    mandatory_hits = _count_hits(_COMPILED_MANDATORY, text)
    optional_hits = _count_hits(_COMPILED_OPTIONAL, text)
    notes: List[str] = []

    if mandatory_hits and optional_hits:
        modality = RequirementModality.UNCERTAIN
        confidence = 0.4
        notes.append(
            "Conflicting mandatory and optional signals in the same text"
        )
    elif mandatory_hits:
        modality = RequirementModality.MANDATORY
        confidence = 0.9
    elif optional_hits:
        modality = RequirementModality.OPTIONAL
        confidence = 0.8
    else:
        modality = RequirementModality.UNCERTAIN
        confidence = 0.3
        notes.append("No modal signal (must/shall/should/may) detected")

    category, cat_conf = classify_category_by_rules(text)
    if category is RequirementCategory.UNCERTAIN:
        notes.append("Category could not be determined from keywords")

    # A requirement must read as a requirement: it needs some signal.
    is_requirement = bool(mandatory_hits or optional_hits)

    return RuleClassification(
        modality=modality,
        category=category,
        confidence=round(min(confidence, cat_conf + 0.1), 3),
        notes=notes,
        is_requirement=is_requirement,
    )


_LLM_SYSTEM_PROMPT = """You are an expert analyst for US state and municipal IT/software RFPs.

For each numbered candidate text, decide whether it is an actual requirement
(not descriptive prose), classify how binding it is (mandatory, optional,
informational, uncertain), and assign exactly one category from:
functional, technical, security, compliance, integration, implementation,
support_and_maintenance, vendor_qualifications, submission, other, uncertain.

Return only a JSON object of the form {"items": [...]} with one item per
candidate, in the same order. Each item must contain: is_requirement (bool),
modality, category, confidence (0..1), notes (array of strings).
Do not invent requirements that are not present in the candidate text."""


def classify_with_llm(
    candidates: List[str],
    provider: Optional[AIProvider],
) -> List[Optional[LLMClassification]]:
    """Classify candidates with the LLM, returning ``None`` per failure.

    Failures (no provider, provider error, malformed output, length mismatch)
    yield ``None`` for the affected candidate so callers can preserve the
    rule-based ``uncertain`` result instead of forcing a classification.
    """
    if provider is None or not candidates:
        return [None] * len(candidates)

    numbered = "\n".join(
        f"{i + 1}. {text}" for i, text in enumerate(candidates)
    )
    try:
        response = provider.generate_structured(
            system_prompt=_LLM_SYSTEM_PROMPT,
            user_text=(
                "Classify each candidate requirement below.\n\n"
                f"{numbered}"
            ),
            response_model=_ClassificationsResponse,
        )
    except Exception:
        return [None] * len(candidates)

    items = getattr(response, "items", None)
    if not items or len(items) != len(candidates):
        return [None] * len(candidates)
    return list(items)