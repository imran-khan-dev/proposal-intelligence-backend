"""Unit tests for independent citation verification (M3.3).

Verifies that ``verify_citations`` reports a per-citation result for every
citation and never silently discards failures. No database or AI API required.
"""

from __future__ import annotations

import uuid
from typing import List

import pytest

from app.ai.provider import AIProvider
from app.modules.drafting.generation import (
    DraftResponseModel,
    EvidenceCitation,
    generate_draft_for_requirement,
)
from app.modules.drafting.retrieval import EvidencePassage
from app.modules.drafting.service import draft_for_requirement
from app.modules.drafting.verification import (
    REASON_EVIDENCE_ID_NOT_FOUND,
    REASON_EXCERPT_NOT_IN_PASSAGE,
    REASON_NOT_COMPANY_KNOWLEDGE,
    REASON_RFP_SOURCE,
    REASON_VERIFIED,
    REASON_WRONG_WORKSPACE,
    verify_citations,
)
from app.modules.requirements.schema import RequirementRecord
from app.rag.store import InMemoryVectorStore, VectorRecord

WS_A = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
WS_B = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")


def _passage(
    chunk_id="chunk_1",
    text="Our platform supports SAML 2.0 single sign-on for enterprise customers.",
    workspace_id="550e8400-e29b-41d4-a716-446655440001",
    source_type="company_knowledge",
    doc_id="comp_doc",
) -> EvidencePassage:
    metadata = {"doc_id": doc_id, "source_type": source_type}
    if workspace_id is not None:
        metadata["workspace_id"] = workspace_id
    return EvidencePassage(
        text=text,
        score=0.9,
        rank=1,
        source_doc_id=doc_id,
        source_filename="company.pdf",
        chunk_id=chunk_id,
        chunk_index=0,
        page_number=1,
        character_start=0,
        character_end=len(text),
        section_path="2.1",
        metadata=metadata,
    )


def _citation(evidence_id="chunk_1", excerpt="SAML 2.0 single sign-on") -> EvidenceCitation:
    return EvidenceCitation(evidence_id=evidence_id, excerpt=excerpt, score=0.9)


class TestVerifyCitations:
    def test_valid_citation_verified(self):
        results = verify_citations([_citation()], [_passage()], WS_A)
        assert len(results) == 1
        assert results[0].verified is True
        assert results[0].reason == REASON_VERIFIED
        assert results[0].source_doc_id == "comp_doc"

    def test_fabricated_evidence_id_fails(self):
        results = verify_citations([_citation(evidence_id="chunk_999")], [_passage()], WS_A)
        assert len(results) == 1
        assert results[0].verified is False
        assert results[0].reason == REASON_EVIDENCE_ID_NOT_FOUND

    def test_altered_excerpt_fails(self):
        altered = _citation(excerpt="We are ISO 9001 certified")
        results = verify_citations([altered], [_passage()], WS_A)
        assert results[0].verified is False
        assert results[0].reason == REASON_EXCERPT_NOT_IN_PASSAGE

    def test_wrong_workspace_fails(self):
        # Passage belongs to WS_B but we verify against WS_A.
        results = verify_citations([_citation()], [_passage(workspace_id=str(WS_B))], WS_A)
        assert results[0].verified is False
        assert results[0].reason == REASON_WRONG_WORKSPACE

    def test_rfp_source_fails(self):
        results = verify_citations(
            [_citation()], [_passage(source_type="rfp")], WS_A
        )
        assert results[0].verified is False
        assert results[0].reason == REASON_RFP_SOURCE

    def test_missing_source_type_fails(self):
        results = verify_citations(
            [_citation()], [_passage(source_type=None)], WS_A
        )
        assert results[0].verified is False
        assert results[0].reason == REASON_NOT_COMPANY_KNOWLEDGE

    def test_absent_workspace_metadata_passes(self):
        # No workspace metadata -> accepted (retrieval is workspace-scoped).
        results = verify_citations(
            [_citation()], [_passage(workspace_id=None)], WS_A
        )
        assert results[0].verified is True

    def test_no_citations_returns_empty(self):
        assert verify_citations([], [_passage()], WS_A) == []

    def test_one_result_per_citation_in_order(self):
        citations = [
            _citation(evidence_id="chunk_1", excerpt="SAML 2.0 single sign-on"),
            _citation(evidence_id="chunk_999", excerpt="anything"),
        ]
        results = verify_citations(citations, [_passage()], WS_A)
        assert len(results) == 2
        assert results[0].evidence_id == "chunk_1"
        assert results[0].verified is True
        assert results[1].evidence_id == "chunk_999"
        assert results[1].verified is False

    def test_does_not_discard_failed_citations(self):
        # A failed citation is still present in the results (not dropped).
        results = verify_citations(
            [_citation(evidence_id="ghost"), _citation()], [_passage()], WS_A
        )
        assert [r.evidence_id for r in results] == ["ghost", "chunk_1"]
        assert [r.verified for r in results] == [False, True]


class _StubProvider(AIProvider):
    """Returns a fixed structured draft response."""

    def __init__(self, response: dict):
        self._response = response

    def generate_structured(self, system_prompt, user_text, response_model):
        return DraftResponseModel(**self._response)

    def chat_with_tools(self, messages, tools):
        raise NotImplementedError


def _make_requirement() -> RequirementRecord:
    return RequirementRecord(
        requirement_key="doc1:p1:100-150",
        doc_id="doc1",
        text="The system must support SAML 2.0 single sign-on.",
        section_title="Authentication",
        section_path="2.1",
        page_number=1,
        character_start=100,
        character_end=150,
        modality="mandatory",
        category="security",
        cross_references=[],
        confidence=0.95,
        uncertainty_notes=[],
        extraction_method="rule",
    )


class _ControlledEmbedder:
    dimension = 4

    def __init__(self, mapping: dict):
        self._mapping = mapping

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._mapping.get(t, [0.0, 0.0, 0.0, 1.0]) for t in texts]


class TestServiceAttachesVerification:
    @pytest.mark.asyncio
    async def test_draft_for_requirement_runs_verification(self):
        query = "The system must support SAML 2.0 single sign-on."
        store = InMemoryVectorStore()
        store.add([
            VectorRecord(
                id="chunk_1",
                text="Our platform supports SAML 2.0 single sign-on for enterprise customers.",
                embedding=[0.0, 1.0, 0.0, 0.0],
                metadata={
                    "doc_id": "comp_doc",
                    "chunk_index": 0,
                    "source_filename": "company.pdf",
                    "workspace_id": str(WS_A),
                    "source_type": "company_knowledge",
                },
            )
        ])
        embedder = _ControlledEmbedder({query: [1.0, 0.0, 0.0, 0.0]})
        provider = _StubProvider({
            "draft_answer": "We support SAML 2.0 single sign-on.",
            "citations": [
                {"evidence_id": "chunk_1",
                 "excerpt": "SAML 2.0 single sign-on", "score": 0.9}
            ],
            "evidence_found": True,
            "uncertainty": "supported",
            "uncertainty_notes": [],
        })

        result = await draft_for_requirement(
            requirement=_make_requirement(),
            workspace_id=WS_A,
            embedder=embedder,
            store=store,
            provider=provider,
            top_k=5,
        )

        assert len(result.draft.citation_verifications) == 1
        assert result.draft.citation_verifications[0].verified is True
        # Draft answer is preserved by verification.
        assert result.draft.draft_answer == "We support SAML 2.0 single sign-on."

    @pytest.mark.asyncio
    async def test_verification_flags_rfp_source_in_flow(self):
        query = "The system must support SAML 2.0 single sign-on."
        store = InMemoryVectorStore()
        # Only an RFP-source chunk is present; retrieval filter excludes it, so
        # generation produces no grounded citations and verification is empty.
        store.add([
            VectorRecord(
                id="rfp_chunk",
                text=query,
                embedding=[0.0, 1.0, 0.0, 0.0],
                metadata={
                    "doc_id": "rfp_doc",
                    "chunk_index": 0,
                    "workspace_id": str(WS_A),
                    "source_type": "rfp",
                },
            )
        ])
        embedder = _ControlledEmbedder({query: [1.0, 0.0, 0.0, 0.0]})
        provider = _StubProvider({
            "draft_answer": "x",
            "citations": [],
            "evidence_found": False,
            "uncertainty": "none",
            "uncertainty_notes": [],
        })

        result = await draft_for_requirement(
            requirement=_make_requirement(),
            workspace_id=WS_A,
            embedder=embedder,
            store=store,
            provider=provider,
            top_k=5,
        )

        assert result.draft.citation_verifications == []
        assert result.draft.evidence_found is False