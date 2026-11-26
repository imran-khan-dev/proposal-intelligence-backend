"""Independent citation verification for RFP draft answers (M3.3).

This module runs *after* draft generation. It does not retrieve evidence or
generate text; it only checks the citations a draft already carries against the
evidence passages that were retrieved for that requirement.

For every citation it verifies all of the following:

* the ``evidence_id`` belongs to the evidence retrieved for the requirement;
* the cited ``excerpt`` is an exact substring of the stored source passage;
* the source passage belongs to the current workspace (it must carry workspace
  metadata and that metadata must match the current workspace);
* the source passage is company knowledge, not an RFP source.

A per-citation result (``verified`` / ``failed`` plus a concise reason) is
returned for *every* input citation. Failed citations are reported, never
silently discarded. The generated draft answer is left untouched.

This is verification only: it does not assign compliance status, trigger human
review, export anything, or start an agent loop.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

from app.modules.drafting.generation import EvidenceCitation
from app.modules.drafting.retrieval import (
    COMPANY_KNOWLEDGE_SOURCE_TYPE,
    EvidencePassage,
)

logger = logging.getLogger(__name__)

# Concise, stable failure reasons (asserted by tests).
REASON_VERIFIED = "verified"
REASON_EVIDENCE_ID_NOT_FOUND = "evidence_id not among retrieved evidence"
REASON_EXCERPT_NOT_IN_PASSAGE = "excerpt not found in source passage"
REASON_WORKSPACE_UNKNOWN = "evidence has no workspace metadata"
REASON_WRONG_WORKSPACE = "evidence belongs to another workspace"
REASON_RFP_SOURCE = "citation points to an RFP source, not company knowledge"
REASON_NOT_COMPANY_KNOWLEDGE = "source is not company knowledge"


@dataclass
class CitationVerificationResult:
    """Per-citation verification outcome.

    Attributes:
        evidence_id: The citation's evidence identifier (chunk id).
        excerpt: The cited excerpt.
        verified: ``True`` when every check passed, ``False`` otherwise.
        reason: Concise machine-readable reason (see ``REASON_*`` constants).
        source_doc_id: Source document id of the matched passage, if any.
        workspace_id: Workspace id recorded on the matched passage, if any.
    """

    evidence_id: str
    excerpt: str
    verified: bool
    reason: str
    source_doc_id: Optional[str] = None
    workspace_id: Optional[str] = None


def _verify_single(
    citation: EvidenceCitation,
    by_chunk_id: dict,
    workspace_id: uuid.UUID,
) -> CitationVerificationResult:
    """Verify one citation against the retrieved passages."""
    evidence_id = citation.evidence_id
    excerpt = citation.excerpt

    passage = by_chunk_id.get(evidence_id)
    if passage is None:
        return CitationVerificationResult(
            evidence_id=evidence_id,
            excerpt=excerpt,
            verified=False,
            reason=REASON_EVIDENCE_ID_NOT_FOUND,
        )

    source_doc_id = passage.source_doc_id
    passage_ws = passage.metadata.get("workspace_id")

    # Excerpt must be an exact substring of the stored source passage.
    if excerpt not in passage.text:
        return CitationVerificationResult(
            evidence_id=evidence_id,
            excerpt=excerpt,
            verified=False,
            reason=REASON_EXCERPT_NOT_IN_PASSAGE,
            source_doc_id=source_doc_id,
            workspace_id=passage_ws,
        )

    # Workspace isolation: the passage must record a workspace and it must
    # match the current one. A passage without workspace metadata cannot be
    # confirmed to belong to this workspace, so it fails verification.
    if passage_ws is None:
        return CitationVerificationResult(
            evidence_id=evidence_id,
            excerpt=excerpt,
            verified=False,
            reason=REASON_WORKSPACE_UNKNOWN,
            source_doc_id=source_doc_id,
            workspace_id=passage_ws,
        )
    if str(passage_ws) != str(workspace_id):
        return CitationVerificationResult(
            evidence_id=evidence_id,
            excerpt=excerpt,
            verified=False,
            reason=REASON_WRONG_WORKSPACE,
            source_doc_id=source_doc_id,
            workspace_id=passage_ws,
        )

    # Source type: only company knowledge may back a draft citation.
    source_type = passage.metadata.get("source_type")
    if source_type == "rfp":
        return CitationVerificationResult(
            evidence_id=evidence_id,
            excerpt=excerpt,
            verified=False,
            reason=REASON_RFP_SOURCE,
            source_doc_id=source_doc_id,
            workspace_id=passage_ws,
        )
    if source_type != COMPANY_KNOWLEDGE_SOURCE_TYPE:
        return CitationVerificationResult(
            evidence_id=evidence_id,
            excerpt=excerpt,
            verified=False,
            reason=REASON_NOT_COMPANY_KNOWLEDGE,
            source_doc_id=source_doc_id,
            workspace_id=passage_ws,
        )

    return CitationVerificationResult(
        evidence_id=evidence_id,
        excerpt=excerpt,
        verified=True,
        reason=REASON_VERIFIED,
        source_doc_id=source_doc_id,
        workspace_id=passage_ws,
    )


def verify_citations(
    citations: List[EvidenceCitation],
    passages: List[EvidencePassage],
    workspace_id: uuid.UUID,
) -> List[CitationVerificationResult]:
    """Verify every citation against the evidence retrieved for a requirement.

    Args:
        citations: The draft's citations (one result is returned per citation).
        passages: The evidence passages retrieved for this requirement.
        workspace_id: The current workspace UUID.

    Returns:
        A list of :class:`CitationVerificationResult`, one per input citation,
        in the same order. Never drops a citation: failures are reported with a
        reason.
    """
    by_chunk_id = {p.chunk_id: p for p in passages if p.chunk_id}
    return [
        _verify_single(citation, by_chunk_id, workspace_id)
        for citation in citations
    ]