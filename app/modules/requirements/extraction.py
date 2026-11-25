"""Requirement extraction from a structured document.

Turns a :class:`~app.modules.requirements.structured_document.StructuredDocument`
into individually identifiable, validated requirements.

Pipeline:
1. Collect candidate text units (list items, table rows, sentences from
   paragraphs and list items).
2. Keep only candidates that read as requirements (rule-based signal or an
   LLM ``is_requirement`` verdict).  Pure descriptive prose is dropped.
3. Classify modality + category (rules first, LLM fallback for the uncertain).
4. Extract cross-references, assign stable IDs, and de-duplicate repeats.

Extraction and classification are independent: this module consumes the
:class:`~app.modules.requirements.classification.RuleClassification` /
``LLMClassification`` results and never calls a model directly.
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field

from app.modules.requirements.classification import (
    LLMClassification,
    RequirementCategory,
    RequirementModality,
    RuleClassification,
    classify_by_rules,
    classify_with_llm,
)
from app.modules.requirements.structured_document import (
    DocumentElement,
    StructuredDocument,
)

# Sentence terminator followed by whitespace / end.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Cross-reference patterns: "Section 3.2", "see 4.1", "REQ-001", "Attachment A".
_CROSSREF_PATTERNS = [
    re.compile(r"\b(?:see|refer to|per|pursuant to)\s+(section\s+)?\d+(?:\.\d+)*", re.IGNORECASE),
    re.compile(r"\bsection\s+\d+(?:\.\d+)*", re.IGNORECASE),
    re.compile(r"\b(?:attachment|exhibit|appendix|addendum)\s+[A-Z0-9]+", re.IGNORECASE),
    re.compile(r"\b[A-Z]{2,5}-\d{2,}\b"),
]


class ExtractedRequirement(BaseModel):
    """A single extracted, validated requirement."""

    requirement_key: str = Field(
        ..., description="Stable identifier derived from doc + source span"
    )
    text: str = Field(..., description="Original requirement text")
    section_title: Optional[str] = Field(
        None, description="Section title the requirement appears under"
    )
    section_path: Optional[str] = Field(
        None, description="Dotted section path the requirement appears under"
    )
    page_number: int = Field(..., description="1-based page number")
    character_start: int = Field(
        ..., description="Absolute start offset (inclusive)"
    )
    character_end: int = Field(
        ..., description="Absolute end offset (exclusive)"
    )
    modality: RequirementModality = Field(..., description="Bindingness")
    category: RequirementCategory = Field(..., description="Category")
    cross_references: List[str] = Field(
        default_factory=list, description="Referenced sections/ids"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    uncertainty_notes: List[str] = Field(default_factory=list)
    extraction_method: str = Field(
        ..., description="rule | llm | hybrid"
    )


class _Candidate(BaseModel):
    text: str
    page_number: int
    character_start: int
    character_end: int
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    # Set when a list item inherits a mandatory lead-in ("...shall provide
    # the following:") and therefore reads as a requirement without its own
    # modal verb.
    lead_in_mandatory: bool = False


def _iter_candidates(doc: StructuredDocument) -> List[_Candidate]:
    """Collect candidate requirement text units in reading order."""
    candidates: List[_Candidate] = []
    lead_in_mandatory = False

    for el in doc.elements:
        if el.kind == "list_item":
            candidates.append(
                _Candidate(
                    text=el.text,
                    page_number=el.span.page_number,
                    character_start=el.span.char_start,
                    character_end=el.span.char_end,
                    section_title=_last_title(el.section_path),
                    section_path=el.section_path,
                    lead_in_mandatory=lead_in_mandatory,
                )
            )
        elif el.kind == "table":
            assert el.table is not None
            lead_in_mandatory = False
            for row in el.table.rows:
                cell_texts = [c.text for c in row if c.text]
                if not cell_texts:
                    continue
                row_text = " | ".join(cell_texts)
                span = row[0].span
                candidates.append(
                    _Candidate(
                        text=row_text,
                        page_number=span.page_number if span else 1,
                        character_start=span.char_start if span else 0,
                        character_end=row[-1].span.char_end
                        if row[-1].span
                        else (span.char_end if span else 0),
                        section_title=_last_title(el.section_path),
                        section_path=el.section_path,
                    )
                )
        elif el.kind == "paragraph":
            lead_in_mandatory = _paragraph_is_mandatory_lead_in(el)
            for sent, s_start, s_end in _split_sentences(el):
                candidates.append(
                    _Candidate(
                        text=sent,
                        page_number=el.span.page_number,
                        character_start=s_start,
                        character_end=s_end,
                        section_title=_last_title(el.section_path),
                        section_path=el.section_path,
                    )
                )
        elif el.kind == "heading":
            lead_in_mandatory = False

    return candidates


def _paragraph_is_mandatory_lead_in(el: DocumentElement) -> bool:
    """True if ``el`` is a paragraph ending with ":" that is mandatory."""
    if el.kind != "paragraph":
        return False
    if not el.text.rstrip().endswith(":"):
        return False
    return classify_by_rules(el.text).modality is RequirementModality.MANDATORY


def _split_sentences(el: DocumentElement):
    """Split a paragraph element into sentences with absolute offsets."""
    text = el.text
    base = el.span.char_start
    pos = 0
    for raw in _SENTENCE_SPLIT_RE.split(text):
        start = base + pos
        end = start + len(raw)
        stripped = raw.strip()
        if stripped:
            lead = len(raw) - len(raw.lstrip())
            yield stripped, start + lead, start + lead + len(stripped)
        pos += len(raw) + 1


def _last_title(section_path: Optional[str]) -> Optional[str]:
    if not section_path:
        return None
    return section_path.split(".")[-1]


def _extract_cross_references(text: str) -> List[str]:
    refs: List[str] = []
    seen = set()
    for pat in _CROSSREF_PATTERNS:
        for m in pat.finditer(text):
            token = m.group(0).strip()
            key = token.lower()
            if key not in seen:
                seen.add(key)
                refs.append(token)
    return refs


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _requirement_key(doc_id: str, page: int, start: int, end: int) -> str:
    """Deterministic, stable requirement key from doc + source span."""
    return f"{doc_id}:p{page}:{start}-{end}"


def extract_requirements(
    doc: StructuredDocument,
    provider=None,
) -> List[ExtractedRequirement]:
    """Extract and classify requirements from a structured document.

    Args:
        doc: The parsed structured document.
        provider: Optional AI provider for LLM-assisted classification of
            candidates the rules leave uncertain.  When ``None`` or on
            failure, uncertain candidates are preserved as ``uncertain``.

    Returns:
        A list of validated :class:`ExtractedRequirement`, de-duplicated by
        normalized text (first occurrence wins).
    """
    candidates = _iter_candidates(doc)
    if not candidates:
        return []

    # Stage 1: rule-based classification for every candidate.
    rule_results: List[RuleClassification] = [
        classify_by_rules(c.text) for c in candidates
    ]

    # Stage 2: LLM fallback only for candidates rules left uncertain.
    uncertain_idx = [
        i
        for i, (c, r) in enumerate(zip(candidates, rule_results))
        if not r.is_requirement or r.modality is RequirementModality.UNCERTAIN
    ]
    llm_results: List[Optional[LLMClassification]] = [None] * len(candidates)
    if uncertain_idx:
        llm_texts = [candidates[i].text for i in uncertain_idx]
        classified = classify_with_llm(llm_texts, provider)
        for slot, result in zip(uncertain_idx, classified):
            llm_results[slot] = result

    # Stage 3: assemble, dedupe, and build requirements.
    requirements: List[ExtractedRequirement] = []
    seen_texts: set[str] = set()

    for cand, rule, llm in zip(candidates, rule_results, llm_results):
        method = "rule"
        modality = rule.modality
        category = rule.category
        confidence = rule.confidence
        notes = list(rule.notes)
        is_requirement = rule.is_requirement

        if llm is not None:
            method = "hybrid" if rule.is_requirement else "llm"
            is_requirement = llm.is_requirement
            modality = llm.modality
            category = llm.category
            confidence = llm.confidence
            notes = list(llm.notes)
        elif cand.lead_in_mandatory and not rule.is_requirement:
            # List item under a mandatory lead-in with no modal of its own.
            method = "rule"
            is_requirement = True
            modality = RequirementModality.MANDATORY
            confidence = 0.75
            notes = list(rule.notes) + [
                "Inherits mandatory modality from the preceding lead-in"
            ]

        if not is_requirement:
            continue

        norm = _normalize(cand.text)
        if not norm:
            continue
        if norm in seen_texts:
            # Duplicate / repeated requirement: keep first occurrence only.
            continue
        seen_texts.add(norm)

        requirements.append(
            ExtractedRequirement(
                requirement_key=_requirement_key(
                    doc.document_id,
                    cand.page_number,
                    cand.character_start,
                    cand.character_end,
                ),
                text=cand.text,
                section_title=cand.section_title,
                section_path=cand.section_path,
                page_number=cand.page_number,
                character_start=cand.character_start,
                character_end=cand.character_end,
                modality=modality,
                category=category,
                cross_references=_extract_cross_references(cand.text),
                confidence=round(float(confidence), 3),
                uncertainty_notes=notes,
                extraction_method=method,
            )
        )

    return requirements