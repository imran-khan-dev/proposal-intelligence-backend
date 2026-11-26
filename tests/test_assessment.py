"""Unit tests for requirement-level compliance assessment (M3.4).

No database or real AI API required: a stub provider returns structured
assessments. Verifies that only verified citations can support a status, that
no-evidence/conflicting/failed-citation cases are conservative (needs_review),
and that the original draft is preserved.
"""

from __future__ import annotations

import uuid
from typing import List

import pytest

from app.ai.provider import AIProvider
from app.modules.drafting.assessment import (
    ComplianceAssessmentModel,
    ComplianceStatus,
    assess_requirement_compliance,
)
from app.modules.drafting.generation import (
    DraftResponseModel,
    EvidenceCitation,
    RequirementDraft,
    UncertaintyLevel,
)
from app.modules.drafting.repository import InMemoryDraftingRepository
from app.modules.drafting.retrieval import EvidencePassage
from app.modules.drafting.service import draft_for_requirement
from app.modules.drafting.verification import (
    REASON_EVIDENCE_ID_NOT_FOUND,
    REASON_VERIFIED,
    REASON_WRONG_WORKSPACE,
    CitationVerificationResult,
)
from app.modules.requirements.schema import RequirementRecord
from app.rag.store import InMemoryVectorStore, VectorRecord

WS_A = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")


def _passage(chunk_id="chunk_1", text="Our platform supports SAML 2.0 single sign-on."):
    return EvidencePassage(
        text=text,
        score=0.9,
        rank=1,
        source_doc_id="comp_doc",
        source_filename="company.pdf",
        chunk_id=chunk_id,
        chunk_index=0,
        page_number=1,
        character_start=0,
        character_end=len(text),
        section_path="2.1",
        metadata={"doc_id": "comp_doc", "source_type": "company_knowledge",
                  "workspace_id": str(WS_A)},
    )


def _verified(evidence_id="chunk_1"):
    return CitationVerificationResult(
        evidence_id=evidence_id, excerpt="SAML 2.0 single sign-on",
        verified=True, reason=REASON_VERIFIED, source_doc_id="comp_doc",
        workspace_id=str(WS_A),
    )


def _failed(evidence_id="chunk_1", reason=REASON_EVIDENCE_ID_NOT_FOUND):
    return CitationVerificationResult(
        evidence_id=evidence_id, excerpt="x", verified=False, reason=reason,
    )


def _draft(
    uncertainty=UncertaintyLevel.SUPPORTED,
    citations=None,
    verifications=None,
    answer="We support SAML 2.0 single sign-on.",
):
    citations = citations if citations is not None else [
        EvidenceCitation("chunk_1", "SAML 2.0 single sign-on", 0.9)
    ]
    verifications = verifications if verifications is not None else [_verified()]
    return RequirementDraft(
        requirement_id=uuid.uuid4(),
        requirement_key="doc1:p1:1-10",
        requirement_text="The system must support SAML 2.0 single sign-on.",
        draft_answer=answer,
        citations=citations,
        evidence_found=True,
        uncertainty=uncertainty,
        uncertainty_notes=[],
        evidence_passages=[_passage()],
        citation_verifications=verifications,
    )


class _AssessmentProvider(AIProvider):
    def __init__(self, status, rationale="rationale"):
        self._status = status
        self._rationale = rationale
        self.calls = 0

    def generate_structured(self, system_prompt, user_text, response_model):
        self.calls += 1
        return ComplianceAssessmentModel(status=self._status, rationale=self._rationale)

    def chat_with_tools(self, messages, tools):
        raise NotImplementedError


class _RaisingProvider(AIProvider):
    def generate_structured(self, system_prompt, user_text, response_model):
        raise RuntimeError("provider boom")

    def chat_with_tools(self, messages, tools):
        raise NotImplementedError


class TestAssessmentStatuses:
    def test_adequately_supported_is_compliant(self):
        provider = _AssessmentProvider("compliant")
        result = assess_requirement_compliance(_draft(), provider)
        assert result.status == ComplianceStatus.COMPLIANT
        assert result.supporting_citation_count == 1
        assert provider.calls == 1

    def test_partial_is_partially_compliant(self):
        provider = _AssessmentProvider("partially_compliant")
        result = assess_requirement_compliance(_draft(), provider)
        assert result.status == ComplianceStatus.PARTIALLY_COMPLIANT

    def test_explicit_unsupported_with_evidence_is_non_compliant(self):
        provider = _AssessmentProvider("non_compliant")
        result = assess_requirement_compliance(_draft(), provider)
        assert result.status == ComplianceStatus.NON_COMPLIANT

    def test_no_verified_evidence_is_needs_review(self):
        provider = _AssessmentProvider("compliant")
        draft = _draft(verifications=[_failed()], citations=[
            EvidenceCitation("chunk_1", "SAML 2.0 single sign-on", 0.9)
        ])
        result = assess_requirement_compliance(draft, provider)
        assert result.status == ComplianceStatus.NEEDS_REVIEW
        assert provider.calls == 0  # provider not consulted

    def test_failed_citation_cannot_support_compliance(self):
        # Only a failed (fabricated-id) verification is present.
        provider = _AssessmentProvider("compliant")
        draft = _draft(
            citations=[EvidenceCitation("ghost", "anything", 0.9)],
            verifications=[_failed("ghost")],
        )
        result = assess_requirement_compliance(draft, provider)
        assert result.status == ComplianceStatus.NEEDS_REVIEW
        assert provider.calls == 0

    def test_conflicting_evidence_is_needs_review(self):
        provider = _AssessmentProvider("compliant")
        draft = _draft(uncertainty=UncertaintyLevel.CONFLICTING)
        result = assess_requirement_compliance(draft, provider)
        assert result.status == ComplianceStatus.NEEDS_REVIEW
        assert provider.calls == 0

    def test_wrong_workspace_citation_cannot_support_compliance(self):
        provider = _AssessmentProvider("compliant")
        draft = _draft(
            citations=[EvidenceCitation("chunk_1", "SAML 2.0 single sign-on", 0.9)],
            verifications=[_failed("chunk_1", REASON_WRONG_WORKSPACE)],
        )
        result = assess_requirement_compliance(draft, provider)
        assert result.status == ComplianceStatus.NEEDS_REVIEW
        assert provider.calls == 0

    def test_invalid_status_defaults_needs_review(self):
        provider = _AssessmentProvider("maybe")
        result = assess_requirement_compliance(_draft(), provider)
        assert result.status == ComplianceStatus.NEEDS_REVIEW

    def test_provider_error_is_needs_review(self):
        result = assess_requirement_compliance(_draft(), _RaisingProvider())
        assert result.status == ComplianceStatus.NEEDS_REVIEW

    def test_provider_none_is_needs_review(self):
        result = assess_requirement_compliance(_draft(), None)
        assert result.status == ComplianceStatus.NEEDS_REVIEW

    def test_original_draft_preserved(self):
        draft = _draft()
        before_answer = draft.draft_answer
        before_citations = list(draft.citations)
        before_verifications = list(draft.citation_verifications)
        assess_requirement_compliance(draft, _AssessmentProvider("compliant"))
        # Assessment does not mutate the draft.
        assert draft.draft_answer == before_answer
        assert draft.citations == before_citations
        assert draft.citation_verifications == before_verifications
        assert draft.compliance_status is None


class TestAssessmentPersistence:
    @pytest.mark.asyncio
    async def test_compliance_persisted_and_readable(self):
        draft = _draft()
        draft.compliance_status = ComplianceStatus.COMPLIANT
        draft.compliance_rationale = "Verified evidence meets the requirement."

        repo = InMemoryDraftingRepository()
        run = await repo.create_draft_run(WS_A, uuid.uuid4(), None)
        # Older/unassessed record is readable (NULL compliance).
        assert run.compliance_status is None

        await repo.complete_draft_run(run.run_id, draft)
        stored = await repo.get_draft_run(run.run_id)
        assert stored is not None
        assert stored.compliance_status == "compliant"
        assert stored.compliance_rationale == "Verified evidence meets the requirement."
        # Original answer and citations preserved.
        assert stored.draft_answer == draft.draft_answer
        assert len(stored.citations) == 1


class _CombinedProvider(AIProvider):
    """Returns a draft for DraftResponseModel and an assessment otherwise."""

    def __init__(self, draft: dict, assessment: dict):
        self._draft = draft
        self._assessment = assessment

    def generate_structured(self, system_prompt, user_text, response_model):
        if response_model is DraftResponseModel:
            return DraftResponseModel(**self._draft)
        if response_model is ComplianceAssessmentModel:
            return ComplianceAssessmentModel(**self._assessment)
        raise ValueError(response_model)

    def chat_with_tools(self, messages, tools):
        raise NotImplementedError


class _ControlledEmbedder:
    dimension = 4

    def __init__(self, mapping: dict):
        self._mapping = mapping

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._mapping.get(t, [0.0, 0.0, 0.0, 1.0]) for t in texts]


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


class TestServiceAssessment:
    @pytest.mark.asyncio
    async def test_service_sets_compliance_and_preserves(self):
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
        provider = _CombinedProvider(
            draft={
                "draft_answer": "We support SAML 2.0 single sign-on.",
                "citations": [
                    {"evidence_id": "chunk_1",
                     "excerpt": "SAML 2.0 single sign-on", "score": 0.9}
                ],
                "evidence_found": True,
                "uncertainty": "supported",
                "uncertainty_notes": [],
            },
            assessment={"status": "compliant", "rationale": "meets requirement"},
        )

        result = await draft_for_requirement(
            requirement=_make_requirement(),
            workspace_id=WS_A,
            embedder=embedder,
            store=store,
            provider=provider,
            top_k=5,
        )

        assert result.draft.compliance_status == ComplianceStatus.COMPLIANT
        assert result.draft.compliance_rationale == "meets requirement"
        # Preservation.
        assert result.draft.draft_answer == "We support SAML 2.0 single sign-on."
        assert len(result.draft.citations) == 1
        assert len(result.draft.citation_verifications) == 1
        assert result.draft.citation_verifications[0].verified is True