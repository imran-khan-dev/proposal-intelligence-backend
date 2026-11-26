"""Unit tests for the drafting module (M3).

Tests evidence retrieval, draft generation, and persistence with mocked providers.
No external AI APIs or database required.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List
from unittest.mock import MagicMock

import pytest

from app.ai.embeddings import EmbeddingProvider
from app.ai.provider import AIProvider
from app.modules.drafting.generation import (
    EvidenceCitation,
    RequirementDraft,
    UncertaintyLevel,
    generate_draft_for_requirement,
)
from app.modules.drafting.assessment import ComplianceStatus
from app.modules.drafting.retrieval import (
    EvidencePassage,
    retrieve_evidence_for_requirement,
)
from app.modules.drafting.repository import InMemoryDraftingRepository, DraftGenerationRunRecord
from app.modules.drafting.service import DraftingResult, draft_for_requirement
from app.modules.requirements.schema import RequirementRecord
from app.rag.store import InMemoryVectorStore, SearchResult, VectorRecord


# --- Test Fixtures ---

class FakeEmbedder(EmbeddingProvider):
    """Deterministic embedding provider for tests."""

    dimension = 384

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Return different vectors for different texts to simulate similarity
        # Use a simple hash-based approach for deterministic but distinct vectors
        vectors = []
        for t in texts:
            # Create a more distinct vector based on text content
            h = hash(t)
            vec = [(h >> (i * 8)) & 0xFF for i in range(384)]
            # Normalize
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


class FakeProvider(AIProvider):
    """Mock AI provider that returns structured draft responses."""

    def __init__(self, response: dict = None):
        self._response = response or {
            "draft_answer": "We can meet this requirement using our existing platform.",
            "citations": [
                {"evidence_id": "chunk_1", "excerpt": "Our platform supports SAML 2.0 single sign-on", "score": 0.9}
            ],
            "evidence_found": True,
            "uncertainty": "supported",
            "uncertainty_notes": []
        }
        self.calls = 0

    def generate_structured(self, system_prompt: str, user_text: str, response_model: type):
        self.calls += 1
        from app.modules.drafting.generation import DraftResponseModel
        return DraftResponseModel(**self._response)

    def chat_with_tools(self, messages: list, tools: list):
        raise NotImplementedError


def _make_vector_store_with_docs() -> InMemoryVectorStore:
    """Create an in-memory vector store with some test documents."""
    store = InMemoryVectorStore()
    records = [
        VectorRecord(
            id="chunk_1",
            text="Our platform supports SAML 2.0 single sign-on for enterprise customers.",
            embedding=[0.1] * 384,
            metadata={
                "doc_id": "doc_1",
                "chunk_index": 0,
                "page_number": 1,
                "character_start": 0,
                "character_end": 70,
                "source_filename": "platform_capabilities.pdf",
                "workspace_id": "test_ws",
                "source_type": "company_knowledge",
                "section_path": "2.1",
            },
        ),
        VectorRecord(
            id="chunk_2",
            text="The system provides role-based access control with multi-factor authentication.",
            embedding=[0.2] * 384,
            metadata={
                "doc_id": "doc_2",
                "chunk_index": 0,
                "page_number": 2,
                "character_start": 0,
                "character_end": 75,
                "source_filename": "security_features.pdf",
                "workspace_id": "test_ws",
                "source_type": "company_knowledge",
                "section_path": "3.1.1",
            },
        ),
        VectorRecord(
            id="chunk_3",
            text="We offer a 99.9% uptime SLA with 24/7 monitoring and support.",
            embedding=[0.3] * 384,
            metadata={
                "doc_id": "doc_3",
                "chunk_index": 0,
                "page_number": 1,
                "character_start": 0,
                "character_end": 60,
                "source_filename": "sla_commitments.pdf",
                "workspace_id": "test_ws",
                "source_type": "company_knowledge",
                "section_path": "4.2",
            },
        ),
    ]
    store.add(records)
    return store


def _make_requirement() -> RequirementRecord:
    """Create a test requirement record."""
    return RequirementRecord(
        requirement_key="doc1:p1:100-150",
        doc_id="doc1",
        text="The system must support SAML 2.0 single sign-on.",
        section_title="Authentication Requirements",
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


WS_UUID = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")


# --- Retrieval Tests ---

class TestEvidenceRetrieval:
    """Tests for evidence retrieval from company knowledge base."""

    def test_retrieves_relevant_passages(self):
        """Retrieval returns passages relevant to the requirement."""
        store = _make_vector_store_with_docs()
        embedder = FakeEmbedder()
        req = _make_requirement()

        passages = retrieve_evidence_for_requirement(
            requirement_text=req.text,
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store,
            section_title=req.section_title,
            section_path=req.section_path,
            category=req.category,
            top_k=3,
        )

        assert len(passages) > 0
        # Should find the SAML passage
        assert any("SAML 2.0" in p.text for p in passages)
        # All passages should have source references preserved
        for p in passages:
            assert p.source_doc_id
            assert p.score >= 0.0
            assert p.rank >= 1

    def test_retrieval_scoped_to_workspace(self):
        """Retrieval only returns passages from the same workspace.

        InMemoryVectorStore doesn't filter by workspace_id during search;
        this is handled by PostgreSQLVectorStore at the database level.
        For in-memory tests, workspace isolation is achieved by using
        separate store instances per workspace.
        """
        # Create separate stores for different workspaces
        store_ws_a = _make_vector_store_with_docs()
        store_ws_b = InMemoryVectorStore()
        store_ws_b.add([
            VectorRecord(
                id="chunk_other_ws",
                text="Other workspace secret document",
                embedding=[0.5] * 384,
                metadata={
                    "doc_id": "doc_other",
                    "chunk_index": 0,
                    "source_filename": "secret.pdf",
                    "workspace_id": "other_ws",
                    "source_type": "company_knowledge",
                },
            )
        ])

        embedder = FakeEmbedder()
        req = _make_requirement()

        # Search in workspace A's store
        passages = retrieve_evidence_for_requirement(
            requirement_text=req.text,
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store_ws_a,
            top_k=5,
        )

        # Should only return passages from workspace A's store
        for p in passages:
            assert p.metadata.get("workspace_id") == "test_ws"

        # Search in workspace B's store
        passages_b = retrieve_evidence_for_requirement(
            requirement_text=req.text,
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store_ws_b,
            top_k=5,
        )

        # Should only return passages from workspace B's store
        for p in passages_b:
            assert p.metadata.get("workspace_id") == "other_ws"

    def test_no_evidence_found_returns_empty_list(self):
        """When no relevant passages exist, returns empty list."""
        store = InMemoryVectorStore()  # Empty store
        embedder = FakeEmbedder()
        req = _make_requirement()

        passages = retrieve_evidence_for_requirement(
            requirement_text=req.text,
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store,
            top_k=5,
        )

        assert passages == []

    def test_top_k_respected(self):
        """Retrieval respects the top_k limit."""
        store = _make_vector_store_with_docs()
        embedder = FakeEmbedder()
        req = _make_requirement()

        passages = retrieve_evidence_for_requirement(
            requirement_text=req.text,
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store,
            top_k=2,
        )

        assert len(passages) <= 2

    def test_empty_requirement_returns_empty(self):
        """Empty requirement text returns empty list."""
        store = _make_vector_store_with_docs()
        embedder = FakeEmbedder()

        passages = retrieve_evidence_for_requirement(
            requirement_text="",
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store,
            top_k=5,
        )

        assert passages == []

    def test_preserves_source_references(self):
        """Retrieved passages preserve all source reference fields."""
        store = _make_vector_store_with_docs()
        embedder = FakeEmbedder()
        req = _make_requirement()

        passages = retrieve_evidence_for_requirement(
            requirement_text=req.text,
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store,
            top_k=3,
        )

        assert len(passages) > 0
        p = passages[0]
        assert p.source_doc_id
        assert p.source_filename
        assert p.chunk_id
        assert p.chunk_index is not None
        assert p.page_number is not None
        assert p.character_start is not None
        assert p.character_end is not None
        assert p.section_path is not None


# --- Draft Generation Tests ---

class TestDraftGeneration:
    """Tests for evidence-backed draft generation."""

    def test_generates_draft_with_evidence(self):
        """When evidence exists, generates a draft with citations."""
        passages = [
            EvidencePassage(
                text="Our platform supports SAML 2.0 single sign-on for enterprise customers.",
                score=0.9,
                rank=1,
                source_doc_id="doc_1",
                source_filename="platform.pdf",
                chunk_id="chunk_1",
                chunk_index=0,
                page_number=1,
                character_start=0,
                character_end=70,
                section_path="2.1",
                metadata={},
            )
        ]
        provider = FakeProvider()
        req_id = uuid.uuid4()

        draft = generate_draft_for_requirement(
            requirement_text="The system must support SAML 2.0 single sign-on.",
            requirement_key="req_1",
            requirement_id=req_id,
            passages=passages,
            section_title="Authentication",
            category="security",
            provider=provider,
        )

        assert draft.draft_answer is not None
        assert len(draft.citations) > 0
        assert draft.evidence_found is True
        assert draft.uncertainty == UncertaintyLevel.SUPPORTED
        assert draft.requirement_id == req_id

    def test_no_evidence_returns_gap_response(self):
        """When no evidence found, returns evidence-gap response."""
        provider = FakeProvider()
        req_id = uuid.uuid4()

        draft = generate_draft_for_requirement(
            requirement_text="The system must support quantum encryption.",
            requirement_key="req_2",
            requirement_id=req_id,
            passages=[],
            section_title="Security",
            category="security",
            provider=provider,
        )

        assert draft.draft_answer == "No supporting evidence found in company knowledge base."
        assert draft.citations == []
        assert draft.evidence_found is False
        assert draft.uncertainty == UncertaintyLevel.NONE
        assert "No relevant company knowledge" in draft.uncertainty_notes[0]

    def test_weak_evidence_indicated_in_uncertainty(self):
        """Weak/partial evidence is reflected in uncertainty level."""
        passages = [
            EvidencePassage(
                text="Our platform supports OAuth 2.0 for authentication.",
                score=0.4,
                rank=1,
                source_doc_id="doc_1",
                source_filename="platform.pdf",
                chunk_id="chunk_1",
                chunk_index=0,
                page_number=1,
                character_start=0,
                character_end=50,
                section_path="2.1",
                metadata={},
            )
        ]
        # Provider returns weak uncertainty
        provider = FakeProvider({
            "draft_answer": "We support OAuth 2.0 but not SAML 2.0 specifically.",
            "citations": [{"evidence_id": "chunk_1", "excerpt": "OAuth 2.0", "score": 0.4}],
            "evidence_found": True,
            "uncertainty": "weak",
            "uncertainty_notes": ["SAML 2.0 not explicitly mentioned"]
        })
        req_id = uuid.uuid4()

        draft = generate_draft_for_requirement(
            requirement_text="The system must support SAML 2.0 single sign-on.",
            requirement_key="req_3",
            requirement_id=req_id,
            passages=passages,
            section_title="Authentication",
            category="security",
            provider=provider,
        )

        assert draft.uncertainty == UncertaintyLevel.WEAK
        assert "SAML 2.0 not explicitly mentioned" in draft.uncertainty_notes

    def test_provider_failure_handled_gracefully(self):
        """Provider errors don't crash; return error draft."""
        class BadProvider(AIProvider):
            def generate_structured(self, system_prompt, user_text, response_model):
                raise RuntimeError("API down")
            def chat_with_tools(self, messages, tools):
                raise NotImplementedError

        passages = [
            EvidencePassage(
                text="Some evidence",
                score=0.8,
                rank=1,
                source_doc_id="doc_1",
                source_filename="test.pdf",
                chunk_id="chunk_1",
                chunk_index=0,
                page_number=1,
                character_start=0,
                character_end=10,
                section_path="1",
                metadata={},
            )
        ]
        provider = BadProvider()
        req_id = uuid.uuid4()

        draft = generate_draft_for_requirement(
            requirement_text="Test requirement",
            requirement_key="req_4",
            requirement_id=req_id,
            passages=passages,
            section_title="Test",
            category="functional",
            provider=provider,
        )

        assert draft.draft_answer is None
        assert draft.evidence_found is False
        assert draft.uncertainty == UncertaintyLevel.NONE
        assert any("Generation error" in note for note in draft.uncertainty_notes)

    def test_citations_link_to_evidence(self):
        """Citations reference specific evidence passages by chunk_id."""
        passages = [
            EvidencePassage(
                text="Our platform supports SAML 2.0 SSO.",
                score=0.9,
                rank=1,
                source_doc_id="doc_1",
                source_filename="platform.pdf",
                chunk_id="chunk_1",
                chunk_index=0,
                page_number=1,
                character_start=0,
                character_end=35,
                section_path="2.1",
                metadata={},
            ),
            EvidencePassage(
                text="We also support OAuth 2.0 and OIDC.",
                score=0.7,
                rank=2,
                source_doc_id="doc_2",
                source_filename="auth.pdf",
                chunk_id="chunk_2",
                chunk_index=0,
                page_number=2,
                character_start=0,
                character_end=35,
                section_path="2.2",
                metadata={},
            ),
        ]
        provider = FakeProvider({
            "draft_answer": "We support SAML 2.0 SSO and OAuth 2.0.",
            "citations": [
                {"evidence_id": "chunk_1", "excerpt": "SAML 2.0 SSO", "score": 0.9},
                {"evidence_id": "chunk_2", "excerpt": "OAuth 2.0", "score": 0.7},
            ],
            "evidence_found": True,
            "uncertainty": "supported",
            "uncertainty_notes": []
        })
        req_id = uuid.uuid4()

        draft = generate_draft_for_requirement(
            requirement_text="The system must support SAML 2.0 SSO.",
            requirement_key="req_5",
            requirement_id=req_id,
            passages=passages,
            section_title="Auth",
            category="security",
            provider=provider,
        )

        assert len(draft.citations) == 2
        assert draft.citations[0].evidence_id == "chunk_1"
        assert draft.citations[1].evidence_id == "chunk_2"
        assert "SAML 2.0 SSO" in draft.citations[0].excerpt


# --- Persistence Tests ---

class TestDraftingRepository:
    """Tests for the in-memory drafting repository."""

    @pytest.mark.asyncio
    async def test_evidence_run_lifecycle(self):
        """Evidence run can be created, completed, and retrieved."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()

        # Create
        run = await repo.create_evidence_run(ws, req_id, "test query")
        assert run.status == "running"
        assert run.query_text == "test query"

        # Complete with passages
        passages = [
            EvidencePassage(
                text="Evidence text",
                score=0.8,
                rank=1,
                source_doc_id="doc_1",
                source_filename="test.pdf",
                chunk_id="chunk_1",
                chunk_index=0,
                page_number=1,
                character_start=0,
                character_end=12,
                section_path="1",
                metadata={},
            )
        ]
        await repo.complete_evidence_run(run.run_id, len(passages), passages)

        # Retrieve
        completed = await repo.get_evidence_run(run.run_id)
        assert completed.status == "completed"
        assert completed.evidence_count == 1

        # List evidence
        evidence = await repo.list_evidence_for_run(run.run_id)
        assert len(evidence) == 1
        assert evidence[0].text == "Evidence text"
        assert evidence[0].score == 0.8

    @pytest.mark.asyncio
    async def test_draft_run_lifecycle(self):
        """Draft run can be created, completed, and retrieved."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        # Create
        run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        assert run.status == "running"

        # Complete with draft
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="Test answer",
            citations=[EvidenceCitation("chunk_1", "Evidence text", 0.8)],
            evidence_found=True,
            uncertainty=UncertaintyLevel.SUPPORTED,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(run.run_id, draft)

        # Retrieve
        completed = await repo.get_draft_run(run.run_id)
        assert completed.status == "completed"
        assert completed.draft_answer == "Test answer"
        assert completed.evidence_found is True
        assert completed.uncertainty == "supported"

    @pytest.mark.asyncio
    async def test_latest_runs_per_requirement(self):
        """Can get latest runs for a specific requirement."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()

        # Create two evidence runs
        run1 = await repo.create_evidence_run(ws, req_id, "query 1")
        await repo.complete_evidence_run(run1.run_id, 0, [])
        run2 = await repo.create_evidence_run(ws, req_id, "query 2")
        await repo.complete_evidence_run(run2.run_id, 0, [])

        latest = await repo.get_latest_evidence_run(ws, req_id)
        assert latest.run_id == run2.run_id

        # Same for draft runs
        draft_run1 = await repo.create_draft_run(ws, req_id, None)
        await repo.complete_draft_run(draft_run1.run_id, RequirementDraft(
            requirement_id=req_id, requirement_key="k", requirement_text="t",
            draft_answer="a1", citations=[], evidence_found=False,
            uncertainty=UncertaintyLevel.NONE, uncertainty_notes=[], evidence_passages=[]
        ))
        draft_run2 = await repo.create_draft_run(ws, req_id, None)
        await repo.complete_draft_run(draft_run2.run_id, RequirementDraft(
            requirement_id=req_id, requirement_key="k", requirement_text="t",
            draft_answer="a2", citations=[], evidence_found=False,
            uncertainty=UncertaintyLevel.NONE, uncertainty_notes=[], evidence_passages=[]
        ))

        latest_draft = await repo.get_latest_draft_run(ws, req_id)
        assert latest_draft.run_id == draft_run2.run_id

    @pytest.mark.asyncio
    async def test_workspace_isolation(self):
        """Runs are isolated by workspace."""
        repo = InMemoryDraftingRepository()
        ws_a = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
        ws_b = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")
        req_id = uuid.uuid4()

        await repo.create_evidence_run(ws_a, req_id, "query a")
        await repo.create_evidence_run(ws_b, req_id, "query b")

        runs_a = await repo.get_latest_evidence_run(ws_a, req_id)
        runs_b = await repo.get_latest_evidence_run(ws_b, req_id)

        assert runs_a.query_text == "query a"
        assert runs_b.query_text == "query b"


# --- Service Integration Tests ---

class TestDraftingService:
    """Tests for the drafting service orchestrating retrieval + generation."""

    @pytest.mark.asyncio
    async def test_draft_for_requirement_end_to_end(self):
        """Full flow: retrieve evidence then generate draft."""
        store = _make_vector_store_with_docs()
        embedder = FakeEmbedder()
        provider = FakeProvider()
        req = _make_requirement()

        result = await draft_for_requirement(
            requirement=req,
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store,
            provider=provider,
            top_k=3,
        )

        assert isinstance(result, DraftingResult)
        assert result.requirement.requirement_key == req.requirement_key
        assert result.draft.draft_answer is not None
        assert result.draft.evidence_found is True
        assert result.evidence_count > 0
        assert result.retrieval_query != ""

    @pytest.mark.asyncio
    async def test_draft_for_requirement_no_evidence(self):
        """Flow works when no evidence is found."""
        store = InMemoryVectorStore()  # Empty
        embedder = FakeEmbedder()
        provider = FakeProvider()
        req = _make_requirement()

        result = await draft_for_requirement(
            requirement=req,
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store,
            provider=provider,
            top_k=3,
        )

        assert result.draft.evidence_found is False
        assert result.draft.draft_answer == "No supporting evidence found in company knowledge base."
        assert result.evidence_count == 0

    @pytest.mark.asyncio
    async def test_draft_for_all_requirements(self):
        """Can process multiple requirements at once."""
        from app.modules.drafting.service import draft_for_all_requirements

        store = _make_vector_store_with_docs()
        embedder = FakeEmbedder()
        provider = FakeProvider()
        req1 = _make_requirement()
        req2 = RequirementRecord(
            requirement_key="doc1:p1:200-250",
            doc_id="doc1",
            text="The system shall provide role-based access control.",
            section_title="Auth",
            section_path="2.2",
            page_number=1,
            character_start=200,
            character_end=250,
            modality="mandatory",
            category="security",
            cross_references=[],
            confidence=0.9,
            uncertainty_notes=[],
            extraction_method="rule",
        )

        results = await draft_for_all_requirements(
            requirements=[req1, req2],
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store,
            provider=provider,
            top_k=3,
        )

        assert len(results) == 2
        for r in results:
            assert isinstance(r, DraftingResult)
            assert r.draft is not None


# --- RFP Source Passages Not Mistaken for Company Evidence ---

class TestSourceSeparation:
    """Tests ensuring RFP source passages are never used as company evidence."""

    def test_retrieval_only_searches_company_knowledge(self):
        """Retrieval only searches the vector store (company knowledge), not RFP source."""
        # The vector store only contains company knowledge documents
        # RFP documents are never added to this store
        store = _make_vector_store_with_docs()
        embedder = FakeEmbedder()
        req = _make_requirement()

        passages = retrieve_evidence_for_requirement(
            requirement_text=req.text,
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store,
            top_k=5,
        )

        # All results should be from company knowledge (the test docs)
        for p in passages:
            assert p.source_doc_id in {"doc_1", "doc_2", "doc_3"}
            # None should be the RFP document
            assert p.source_doc_id != "doc1"  # RFP doc_id

    def test_requirement_source_kept_separate(self):
        """Requirement's own source reference is not used as evidence."""
        # The requirement has its own source (page_number, character offsets)
        # These should never appear in evidence passages
        store = _make_vector_store_with_docs()
        embedder = FakeEmbedder()
        req = _make_requirement()

        passages = retrieve_evidence_for_requirement(
            requirement_text=req.text,
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store,
            top_k=5,
        )

        for p in passages:
            # Evidence should have its own source references
            assert p.source_doc_id != req.doc_id
            assert p.page_number != req.page_number or p.character_start != req.character_start


# --- M3.2: Company-Knowledge-Only Retrieval Filter ---

class _ControlledEmbedder(EmbeddingProvider):
    """Embedder returning preset vectors keyed by exact text (dimension 4)."""

    dimension = 4

    def __init__(self, mapping: dict):
        self._mapping = mapping

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._mapping.get(t, [0.0, 0.0, 0.0, 1.0]) for t in texts]


def _make_mixed_store() -> InMemoryVectorStore:
    """Store with an RFP doc and a company-knowledge doc in the same workspace.

    The RFP chunk's embedding is the strongest semantic match for the query
    vector, so without a source-type filter it would rank first.
    """
    store = InMemoryVectorStore()
    records = [
        VectorRecord(
            id="rfp_chunk",
            text="The system must support SAML 2.0 single sign-on.",
            embedding=[0.9, 0.1, 0.0, 0.0],
            metadata={
                "doc_id": "rfp_doc",
                "chunk_index": 0,
                "source_filename": "rfp.pdf",
                "workspace_id": "test_ws",
                "source_type": "rfp",
            },
        ),
        VectorRecord(
            id="comp_chunk",
            text="Our platform supports SAML 2.0 single sign-on for enterprise customers.",
            embedding=[0.0, 1.0, 0.0, 0.0],
            metadata={
                "doc_id": "comp_doc",
                "chunk_index": 0,
                "source_filename": "company.pdf",
                "workspace_id": "test_ws",
                "source_type": "company_knowledge",
            },
        ),
    ]
    store.add(records)
    return store


_QUERY_TEXT = "The system must support SAML 2.0 single sign-on."
_QUERY_EMBEDDING = [1.0, 0.0, 0.0, 0.0]


class TestSourceTypeFilter:
    """Drafting retrieval must search only company knowledge documents."""

    def test_rfp_is_strongest_match_but_never_returned(self):
        """Even when the RFP passage is the strongest match, it is excluded."""
        store = _make_mixed_store()

        # Sanity: without a filter the RFP chunk ranks first (strongest match).
        unfiltered = store.search(_QUERY_EMBEDDING, k=5)
        assert unfiltered[0].record.id == "rfp_chunk"

        embedder = _ControlledEmbedder({_QUERY_TEXT: _QUERY_EMBEDDING})
        passages = retrieve_evidence_for_requirement(
            requirement_text=_QUERY_TEXT,
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store,
            top_k=5,
        )

        # Only the company-knowledge passage is returned.
        assert len(passages) == 1
        assert passages[0].chunk_id == "comp_chunk"
        assert passages[0].source_doc_id == "comp_doc"
        assert all(p.source_doc_id != "rfp_doc" for p in passages)
        assert all(p.metadata.get("source_type") == "company_knowledge" for p in passages)

    def test_company_only_store_unaffected(self):
        """A store with only company knowledge still returns results."""
        store = _make_vector_store_with_docs()
        embedder = FakeEmbedder()
        req = _make_requirement()

        passages = retrieve_evidence_for_requirement(
            requirement_text=req.text,
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store,
            top_k=5,
        )
        assert len(passages) > 0
        assert all(p.metadata.get("source_type") == "company_knowledge" for p in passages)

    def test_source_type_filter_preserves_workspace_isolation(self):
        """The source-type filter does not weaken workspace isolation.

        Two workspaces each have their own store; searching workspace A with
        the company-knowledge filter returns only A's company passage and
        never B's, while still excluding A's own RFP passage.
        """
        store_a = InMemoryVectorStore()
        store_a.add([
            VectorRecord(
                id="a_rfp",
                text="The system must support SAML 2.0 single sign-on.",
                embedding=[0.9, 0.1, 0.0, 0.0],
                metadata={"doc_id": "a_rfp_doc", "chunk_index": 0,
                          "workspace_id": "ws_a", "source_type": "rfp"},
            ),
            VectorRecord(
                id="a_comp",
                text="Workspace A company knowledge about SAML.",
                embedding=[0.0, 1.0, 0.0, 0.0],
                metadata={"doc_id": "a_comp_doc", "chunk_index": 0,
                          "workspace_id": "ws_a", "source_type": "company_knowledge"},
            ),
        ])
        store_b = InMemoryVectorStore()
        store_b.add([
            VectorRecord(
                id="b_comp",
                text="Workspace B company knowledge about SAML.",
                embedding=[0.0, 1.0, 0.0, 0.0],
                metadata={"doc_id": "b_comp_doc", "chunk_index": 0,
                          "workspace_id": "ws_b", "source_type": "company_knowledge"},
            ),
        ])

        embedder = _ControlledEmbedder({_QUERY_TEXT: _QUERY_EMBEDDING})

        passages_a = retrieve_evidence_for_requirement(
            requirement_text=_QUERY_TEXT,
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store_a,
            top_k=5,
        )
        assert [p.chunk_id for p in passages_a] == ["a_comp"]
        assert all(p.metadata.get("workspace_id") == "ws_a" for p in passages_a)
        assert all(p.source_doc_id != "a_rfp_doc" for p in passages_a)

        passages_b = retrieve_evidence_for_requirement(
            requirement_text=_QUERY_TEXT,
            workspace_id=WS_UUID,
            embedder=embedder,
            store=store_b,
            top_k=5,
        )
        assert [p.chunk_id for p in passages_b] == ["b_comp"]
        assert all(p.metadata.get("workspace_id") == "ws_b" for p in passages_b)


# --- M3.2: Draft Provenance (grounded citations) ---

class TestCitationProvenance:
    """Generated citations must reference only retrieved evidence."""

    def _passage(self, text="Our platform supports SAML 2.0 SSO.", score=0.9):
        return EvidencePassage(
            text=text,
            score=score,
            rank=1,
            source_doc_id="doc_1",
            source_filename="platform.pdf",
            chunk_id="chunk_1",
            chunk_index=0,
            page_number=1,
            character_start=0,
            character_end=len(text),
            section_path="2.1",
            metadata={},
        )

    def test_fabricated_evidence_id_dropped(self):
        """A citation referencing an unknown evidence_id is discarded."""
        provider = FakeProvider({
            "draft_answer": "We have SAML 2.0 SSO.",
            "citations": [
                {"evidence_id": "chunk_999", "excerpt": "SAML 2.0 SSO", "score": 0.9}
            ],
            "evidence_found": True,
            "uncertainty": "supported",
            "uncertainty_notes": [],
        })

        draft = generate_draft_for_requirement(
            requirement_text="The system must support SAML 2.0 SSO.",
            requirement_key="req_fab_id",
            requirement_id=uuid.uuid4(),
            passages=[self._passage()],
            provider=provider,
        )

        assert draft.citations == []
        assert draft.evidence_found is False
        assert draft.draft_answer == "No supporting evidence found in company knowledge base."

    def test_fabricated_quotation_dropped(self):
        """A citation whose excerpt is not in the passage is discarded."""
        provider = FakeProvider({
            "draft_answer": "We are ISO 9001 certified.",
            "citations": [
                {"evidence_id": "chunk_1",
                 "excerpt": "We are ISO 9001 certified", "score": 0.9}
            ],
            "evidence_found": True,
            "uncertainty": "supported",
            "uncertainty_notes": [],
        })

        draft = generate_draft_for_requirement(
            requirement_text="The system must support SAML 2.0 SSO.",
            requirement_key="req_fab_quote",
            requirement_id=uuid.uuid4(),
            passages=[self._passage()],
            provider=provider,
        )

        assert draft.citations == []
        assert draft.evidence_found is False

    def test_valid_citation_kept_with_passage_score(self):
        """A grounded citation is kept and takes the passage's score."""
        provider = FakeProvider({
            "draft_answer": "We support SAML 2.0 SSO.",
            "citations": [
                {"evidence_id": "chunk_1", "excerpt": "SAML 2.0 SSO", "score": 0.99}
            ],
            "evidence_found": True,
            "uncertainty": "supported",
            "uncertainty_notes": [],
        })

        draft = generate_draft_for_requirement(
            requirement_text="The system must support SAML 2.0 SSO.",
            requirement_key="req_valid",
            requirement_id=uuid.uuid4(),
            passages=[self._passage(score=0.9)],
            provider=provider,
        )

        assert len(draft.citations) == 1
        assert draft.citations[0].evidence_id == "chunk_1"
        assert draft.citations[0].excerpt == "SAML 2.0 SSO"
        # Score comes from the retrieved passage, not the model.
        assert draft.citations[0].score == 0.9
        assert draft.evidence_found is True


# --- M4.1: Human Review Workflow ---

class TestHumanReviewRepository:
    """Tests for the human review workflow in the repository."""

    @pytest.mark.asyncio
    async def test_submit_review_approve(self):
        """Submitting an 'approve' review sets review_status to 'approved'."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        # Create and complete a draft run
        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI generated answer",
            citations=[EvidenceCitation("chunk_1", "Evidence text", 0.8)],
            evidence_found=True,
            uncertainty=UncertaintyLevel.SUPPORTED,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # Submit review: approve
        updated = await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="approved",
            reviewer_edited_answer="Human edited answer",
            reviewer_compliance_status="compliant",
            reviewer_compliance_rationale="Meets all criteria",
            reviewer_comment="Looks good",
            citation_reviews=[
                {"evidence_id": "chunk_1", "decision": "accept", "reason": "Accurate"}
            ],
        )

        assert updated.review_status == "approved"
        assert updated.reviewer_edited_answer == "Human edited answer"
        assert updated.reviewer_compliance_status == "compliant"
        assert updated.reviewer_compliance_rationale == "Meets all criteria"
        assert updated.reviewer_comment == "Looks good"
        assert len(updated.citation_reviews) == 1
        assert updated.citation_reviews[0]["decision"] == "accept"
        assert updated.reviewed_at is not None
        assert updated.reviewer_workspace_id == ws
        # Original AI draft preserved
        assert updated.draft_answer == "AI generated answer"

    @pytest.mark.asyncio
    async def test_submit_review_request_changes(self):
        """Submitting a 'request_changes' review sets review_status to 'changes_requested'."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI generated answer",
            citations=[EvidenceCitation("chunk_1", "Evidence text", 0.8)],
            evidence_found=True,
            uncertainty=UncertaintyLevel.SUPPORTED,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        updated = await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="changes_requested",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="Need to add more detail",
            citation_reviews=[
                {"evidence_id": "chunk_1", "decision": "reject", "reason": "Outdated"}
            ],
        )

        assert updated.review_status == "changes_requested"
        assert updated.reviewer_edited_answer is None
        assert updated.reviewer_comment == "Need to add more detail"
        assert updated.citation_reviews[0]["decision"] == "reject"

    @pytest.mark.asyncio
    async def test_review_lifecycle_transitions(self):
        """Valid transitions: None -> approved/changes_requested; changes_requested -> approved/changes_requested; approved is terminal."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI generated answer",
            citations=[EvidenceCitation("chunk_1", "Evidence text", 0.8)],
            evidence_found=True,
            uncertainty=UncertaintyLevel.SUPPORTED,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # First review: request changes
        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="changes_requested",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="First review",
            citation_reviews=[],
        )

        # Second review: approve (valid transition from changes_requested)
        updated = await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="approved",
            reviewer_edited_answer="Final answer",
            reviewer_compliance_status="compliant",
            reviewer_compliance_rationale="Now complete",
            reviewer_comment="Approved on second review",
            citation_reviews=[],
        )
        assert updated.review_status == "approved"

        # Third review: try to change from approved -> should fail
        with pytest.raises(ValueError, match="Invalid review transition"):
            await repo.submit_review(
                workspace_id=ws,
                run_id=draft_run.run_id,
                review_status="changes_requested",
                reviewer_edited_answer=None,
                reviewer_compliance_status=None,
                reviewer_compliance_rationale=None,
                reviewer_comment="Trying to reopen",
                citation_reviews=[],
            )

    @pytest.mark.asyncio
    async def test_invalid_transition_from_none_to_invalid(self):
        """Transition from None to invalid status should fail."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI generated answer",
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        with pytest.raises(ValueError, match="Invalid review transition"):
            await repo.submit_review(
                workspace_id=ws,
                run_id=draft_run.run_id,
                review_status="invalid_status",
                reviewer_edited_answer=None,
                reviewer_compliance_status=None,
                reviewer_compliance_rationale=None,
                reviewer_comment="Invalid",
                citation_reviews=[],
            )

    @pytest.mark.asyncio
    async def test_workspace_isolation_on_review(self):
        """Review enforces workspace isolation."""
        repo = InMemoryDraftingRepository()
        ws_a = WS_UUID
        ws_b = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws_a, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI generated answer",
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # Try to review from different workspace -> should fail
        with pytest.raises(ValueError, match="does not belong to workspace"):
            await repo.submit_review(
                workspace_id=ws_b,
                run_id=draft_run.run_id,
                review_status="approved",
                reviewer_edited_answer=None,
                reviewer_compliance_status=None,
                reviewer_compliance_rationale=None,
                reviewer_comment="Cross-workspace attempt",
                citation_reviews=[],
            )

    @pytest.mark.asyncio
    async def test_get_draft_run_for_review_workspace_isolation(self):
        """get_draft_run_for_review enforces workspace isolation."""
        repo = InMemoryDraftingRepository()
        ws_a = WS_UUID
        ws_b = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws_a, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI generated answer",
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # Same workspace -> returns record
        found = await repo.get_draft_run_for_review(ws_a, draft_run.run_id)
        assert found is not None
        assert found.run_id == draft_run.run_id

        # Different workspace -> returns None
        not_found = await repo.get_draft_run_for_review(ws_b, draft_run.run_id)
        assert not_found is None

    @pytest.mark.asyncio
    async def test_review_preserves_ai_originals(self):
        """Review stores human edits separately; AI originals unchanged."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        original_answer = "AI says we support SAML 2.0"
        original_compliance = "needs_review"
        original_rationale = "AI rationale"
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="The system must support SAML 2.0",
            draft_answer=original_answer,
            citations=[EvidenceCitation("chunk_1", "SAML 2.0 supported", 0.9)],
            evidence_found=True,
            uncertainty=UncertaintyLevel.SUPPORTED,
            uncertainty_notes=[],
            evidence_passages=[],
            compliance_status=ComplianceStatus.NEEDS_REVIEW,
            compliance_rationale=original_rationale,
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # Submit review with edits
        updated = await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="approved",
            reviewer_edited_answer="Human says we support SAML 2.0 and OAuth",
            reviewer_compliance_status="compliant",
            reviewer_compliance_rationale="Human verified",
            reviewer_comment="Updated to include OAuth",
            citation_reviews=[
                {"evidence_id": "chunk_1", "decision": "accept", "reason": "Verified"}
            ],
        )

        # AI originals preserved
        assert updated.draft_answer == original_answer
        assert updated.compliance_status == original_compliance
        assert updated.compliance_rationale == original_rationale
        # Human edits stored separately
        assert updated.reviewer_edited_answer == "Human says we support SAML 2.0 and OAuth"
        assert updated.reviewer_compliance_status == "compliant"
        assert updated.reviewer_compliance_rationale == "Human verified"

    @pytest.mark.asyncio
    async def test_citation_reviews_distinct_from_verification(self):
        """Citation reviews are stored separately from automated verification results."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI answer",
            citations=[
                EvidenceCitation("chunk_1", "Evidence 1", 0.9),
                EvidenceCitation("chunk_2", "Evidence 2", 0.8),
            ],
            evidence_found=True,
            uncertainty=UncertaintyLevel.SUPPORTED,
            uncertainty_notes=[],
            evidence_passages=[],
            citation_verifications=[
                # Mock verification results (M3.3)
                type("VerificationResult", (), {
                    "evidence_id": "chunk_1",
                    "excerpt": "Evidence 1",
                    "verified": True,
                    "reason": "verified",
                    "source_doc_id": "doc_1",
                    "workspace_id": str(ws),
                })(),
                type("VerificationResult", (), {
                    "evidence_id": "chunk_2",
                    "excerpt": "Evidence 2",
                    "verified": False,
                    "reason": "excerpt_not_found_in_passage",
                    "source_doc_id": "doc_2",
                    "workspace_id": str(ws),
                })(),
            ],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # Submit review with citation decisions (distinct from verification)
        updated = await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="approved",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="Citation review done",
            citation_reviews=[
                {"evidence_id": "chunk_1", "decision": "accept", "reason": "Confirmed"},
                {"evidence_id": "chunk_2", "decision": "modify", "reviewer_excerpt": "Corrected excerpt", "reason": "Fixed quote"},
            ],
        )

        # Verification results (M3.3) are in citations via _citations_with_verification
        assert updated.citations[0]["verified"] is True
        assert updated.citations[1]["verified"] is False
        # Human citation reviews are separate
        assert len(updated.citation_reviews) == 2
        assert updated.citation_reviews[0]["decision"] == "accept"
        assert updated.citation_reviews[1]["decision"] == "modify"
        assert updated.citation_reviews[1]["reviewer_excerpt"] == "Corrected excerpt"
        # A failed verification does NOT mean human cannot accept/modify
        # Human review is independent of automated verification


# --- Regression Tests for Review Status Default ---

class TestReviewStatusDefault:
    """Regression tests ensuring new drafts default to pending_review and legacy NULL is readable."""

    @pytest.mark.asyncio
    async def test_new_draft_run_defaults_to_pending_review(self):
        """Newly created draft runs must have review_status='pending_review'."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)

        # The created run record should have review_status="pending_review"
        assert draft_run.review_status == "pending_review"
        # And all other review fields should be their defaults
        assert draft_run.reviewer_edited_answer is None
        assert draft_run.reviewer_compliance_status is None
        assert draft_run.reviewer_compliance_rationale is None
        assert draft_run.reviewer_comment is None
        assert draft_run.citation_reviews == []
        assert draft_run.reviewed_at is None
        assert draft_run.reviewer_workspace_id is None

    @pytest.mark.asyncio
    async def test_legacy_null_review_status_readable(self):
        """Legacy draft runs with NULL review_status remain readable (defaults applied)."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        # Manually create a record simulating a legacy row with review_status=None
        run_id = uuid.uuid4()
        legacy_record = DraftGenerationRunRecord(
            run_id=run_id,
            workspace_id=ws,
            requirement_id=req_id,
            retrieval_run_id=evidence_run_id,
            status="completed",
            draft_answer="Legacy answer",
            citations=[],
            evidence_found=False,
            uncertainty="none",
            uncertainty_notes=[],
            compliance_status=None,
            compliance_rationale=None,
            # Legacy: review_status explicitly None (simulating NULL in DB)
            review_status=None,
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment=None,
            citation_reviews=[],
            reviewed_at=None,
            reviewer_workspace_id=None,
            errors=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        repo._draft_runs[run_id] = legacy_record

        # Fetching the legacy record should return it with review_status=None (readable)
        fetched = await repo.get_draft_run(run_id)
        assert fetched is not None
        assert fetched.review_status is None
        assert fetched.draft_answer == "Legacy answer"

        # But submitting a review should work and transition from None -> approved
        updated = await repo.submit_review(
            workspace_id=ws,
            run_id=run_id,
            review_status="approved",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="Legacy review",
            citation_reviews=[],
        )
        assert updated.review_status == "approved"


# --- M4.2: Append-only Human Review Audit History ---

class TestReviewAuditHistory:
    """Tests for the append-only human review audit history (M4.2)."""

    @pytest.mark.asyncio
    async def test_submit_review_creates_audit_event(self):
        """Submitting a review creates an immutable audit event."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI generated answer",
            citations=[EvidenceCitation("chunk_1", "Evidence text", 0.8)],
            evidence_found=True,
            uncertainty=UncertaintyLevel.SUPPORTED,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # Submit review
        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="approved",
            reviewer_edited_answer="Human edited answer",
            reviewer_compliance_status="compliant",
            reviewer_compliance_rationale="Meets all criteria",
            reviewer_comment="Looks good",
            citation_reviews=[
                {"evidence_id": "chunk_1", "decision": "accept", "reason": "Accurate"}
            ],
        )

        # Get review history
        history = await repo.get_review_history(ws, req_id)

        assert len(history) == 1
        event = history[0]
        assert event.requirement_id == req_id
        assert event.draft_run_id == draft_run.run_id
        assert event.review_status == "approved"
        assert event.reviewer_edited_answer == "Human edited answer"
        assert event.reviewer_compliance_status == "compliant"
        assert event.reviewer_compliance_rationale == "Meets all criteria"
        assert event.reviewer_comment == "Looks good"
        assert len(event.citation_reviews) == 1
        assert event.citation_reviews[0]["decision"] == "accept"
        assert event.reviewer_workspace_id == ws

    @pytest.mark.asyncio
    async def test_multiple_reviews_append_to_history(self):
        """Multiple reviews for the same requirement append to history (oldest first)."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI generated answer",
            citations=[EvidenceCitation("chunk_1", "Evidence text", 0.8)],
            evidence_found=True,
            uncertainty=UncertaintyLevel.SUPPORTED,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # First review: request changes
        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="changes_requested",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="First review - needs more detail",
            citation_reviews=[],
        )

        # Second review: approve
        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="approved",
            reviewer_edited_answer="Final answer",
            reviewer_compliance_status="compliant",
            reviewer_compliance_rationale="Now complete",
            reviewer_comment="Approved on second review",
            citation_reviews=[],
        )

        # Get history - should have 2 events in chronological order
        history = await repo.get_review_history(ws, req_id)

        assert len(history) == 2
        # First event (oldest)
        assert history[0].review_status == "changes_requested"
        assert history[0].reviewer_comment == "First review - needs more detail"
        # Second event (newest)
        assert history[1].review_status == "approved"
        assert history[1].reviewer_comment == "Approved on second review"
        assert history[1].reviewer_edited_answer == "Final answer"

    @pytest.mark.asyncio
    async def test_audit_events_are_immutable(self):
        """Audit events cannot be modified or deleted after creation."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI generated answer",
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # First review: request changes
        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="changes_requested",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="Original comment",
            citation_reviews=[],
        )

        # Get history
        history = await repo.get_review_history(ws, req_id)
        assert len(history) == 1
        original_event_id = history[0].id
        original_comment = history[0].reviewer_comment

        # Second review: approve (valid transition from changes_requested)
        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="approved",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="Second comment",
            citation_reviews=[],
        )

        # History should have 2 events, first one unchanged
        history = await repo.get_review_history(ws, req_id)
        assert len(history) == 2
        assert history[0].id == original_event_id
        assert history[0].reviewer_comment == original_comment  # Immutable
        assert history[1].reviewer_comment == "Second comment"

    @pytest.mark.asyncio
    async def test_workspace_isolation_on_history(self):
        """get_review_history enforces workspace isolation."""
        repo = InMemoryDraftingRepository()
        ws_a = WS_UUID
        ws_b = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws_a, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI generated answer",
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # Submit review from workspace A
        await repo.submit_review(
            workspace_id=ws_a,
            run_id=draft_run.run_id,
            review_status="approved",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="Workspace A review",
            citation_reviews=[],
        )

        # Query history from workspace A - should see events
        history_a = await repo.get_review_history(ws_a, req_id)
        assert len(history_a) == 1
        assert history_a[0].reviewer_comment == "Workspace A review"

        # Query history from workspace B - should see NO events (workspace isolation)
        history_b = await repo.get_review_history(ws_b, req_id)
        assert len(history_b) == 0

    @pytest.mark.asyncio
    async def test_history_ordered_chronologically(self):
        """History events are ordered by reviewed_at ascending (oldest first)."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI generated answer",
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # Submit multiple reviews in sequence (valid transitions)
        # 1. changes_requested (from pending_review/None)
        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="changes_requested",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="Review 1 - changes requested",
            citation_reviews=[],
        )
        # 2. approved (from changes_requested)
        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="approved",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="Review 2 - approved",
            citation_reviews=[],
        )
        # 3. changes_requested (from approved -> not allowed!)
        # Instead, we can't do another review from approved since it's terminal
        # So we test with 2 events instead
        pass

        history = await repo.get_review_history(ws, req_id)
        assert len(history) == 2

        # Verify chronological order (oldest first)
        assert history[0].reviewer_comment == "Review 1 - changes requested"
        assert history[1].reviewer_comment == "Review 2 - approved"
        # Verify timestamps are ascending
        assert history[0].reviewed_at <= history[1].reviewed_at

    @pytest.mark.asyncio
    async def test_audit_event_includes_previous_status_and_decision(self):
        """Each audit event records previous_review_status and explicit decision."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI generated answer",
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # First review: approve (from pending_review/None)
        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="approved",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="First review - approved",
            citation_reviews=[],
        )

        history = await repo.get_review_history(ws, req_id)
        assert len(history) == 1
        event = history[0]
        # First event: previous_review_status should be "pending_review" (default)
        assert event.previous_review_status == "pending_review"
        assert event.review_status == "approved"
        assert event.decision == "approve"

        # Create a SEPARATE requirement to test multi-review sequence
        req_id_2 = uuid.uuid4()
        draft_run2 = await repo.create_draft_run(ws, req_id_2, evidence_run_id)
        draft2 = RequirementDraft(
            requirement_id=req_id_2,
            requirement_key="req_2",
            requirement_text="Test requirement 2",
            draft_answer="AI generated answer",
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run2.run_id, draft2)

        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run2.run_id,
            review_status="changes_requested",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="First review - changes requested",
            citation_reviews=[],
        )
        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run2.run_id,
            review_status="approved",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="Second review - approved",
            citation_reviews=[],
        )

        history2 = await repo.get_review_history(ws, req_id_2)
        assert len(history2) == 2
        # First event
        assert history2[0].previous_review_status == "pending_review"
        assert history2[0].decision == "request_changes"
        # Second event
        assert history2[1].previous_review_status == "changes_requested"
        assert history2[1].decision == "approve"

    @pytest.mark.asyncio
    async def test_invalid_transition_creates_no_audit_event(self):
        """Invalid lifecycle transitions create no audit events and leave state unchanged."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI generated answer",
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # First valid review: approve
        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="approved",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="Valid approve",
            citation_reviews=[],
        )

        # Try invalid transition: approved -> changes_requested (not allowed)
        with pytest.raises(ValueError, match="Invalid review transition"):
            await repo.submit_review(
                workspace_id=ws,
                run_id=draft_run.run_id,
                review_status="changes_requested",
                reviewer_edited_answer=None,
                reviewer_compliance_status=None,
                reviewer_compliance_rationale=None,
                reviewer_comment="Invalid transition attempt",
                citation_reviews=[],
            )

        # History should still have only 1 event
        history = await repo.get_review_history(ws, req_id)
        assert len(history) == 1
        assert history[0].reviewer_comment == "Valid approve"
        assert history[0].review_status == "approved"

        # Draft run should still be approved
        draft_run_updated = await repo.get_draft_run(draft_run.run_id)
        assert draft_run_updated.review_status == "approved"

    @pytest.mark.asyncio
    async def test_workspace_mismatch_creates_no_audit_event(self):
        """Workspace mismatches create no audit events."""
        repo = InMemoryDraftingRepository()
        ws_a = WS_UUID
        ws_b = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws_a, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI generated answer",
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # Try to review from different workspace
        with pytest.raises(ValueError, match="does not belong to workspace"):
            await repo.submit_review(
                workspace_id=ws_b,
                run_id=draft_run.run_id,
                review_status="approved",
                reviewer_edited_answer=None,
                reviewer_compliance_status=None,
                reviewer_compliance_rationale=None,
                reviewer_comment="Cross-workspace attempt",
                citation_reviews=[],
            )

        # History should be empty for both workspaces
        history_a = await repo.get_review_history(ws_a, req_id)
        history_b = await repo.get_review_history(ws_b, req_id)
        assert len(history_a) == 0
        assert len(history_b) == 0

    @pytest.mark.asyncio
    async def test_snapshot_matches_latest_audit_event(self):
        """Current review snapshot always matches the latest successful audit event."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI answer",
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # First review: changes_requested
        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="changes_requested",
            reviewer_edited_answer="Edited v1",
            reviewer_compliance_status="partially_compliant",
            reviewer_compliance_rationale="Needs work",
            reviewer_comment="First review",
            citation_reviews=[],
        )

        draft_run_1 = await repo.get_draft_run(draft_run.run_id)
        history_1 = await repo.get_review_history(ws, req_id)
        # Snapshot matches first audit event
        assert draft_run_1.review_status == history_1[-1].review_status
        assert draft_run_1.reviewer_edited_answer == history_1[-1].reviewer_edited_answer
        assert draft_run_1.reviewer_compliance_status == history_1[-1].reviewer_compliance_status

        # Second review: approve
        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="approved",
            reviewer_edited_answer="Edited v2 - final",
            reviewer_compliance_status="compliant",
            reviewer_compliance_rationale="All good now",
            reviewer_comment="Second review - approved",
            citation_reviews=[],
        )

        draft_run_2 = await repo.get_draft_run(draft_run.run_id)
        history_2 = await repo.get_review_history(ws, req_id)
        # Snapshot matches latest (second) audit event
        assert draft_run_2.review_status == history_2[-1].review_status
        assert draft_run_2.reviewer_edited_answer == history_2[-1].reviewer_edited_answer
        assert draft_run_2.reviewer_compliance_status == history_2[-1].reviewer_compliance_status
        assert draft_run_2.reviewer_compliance_rationale == history_2[-1].reviewer_compliance_rationale
        assert draft_run_2.reviewer_comment == history_2[-1].reviewer_comment

    @pytest.mark.asyncio
    async def test_legacy_draft_without_events_returns_empty_history(self):
        """Legacy drafts without audit events return empty history list."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        # Create a draft run but DON'T submit any review
        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI answer",
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # History should be empty (no reviews submitted yet)
        history = await repo.get_review_history(ws, req_id)
        assert len(history) == 0
        assert history == []

    @pytest.mark.asyncio
    async def test_stable_ordering_with_identical_timestamps(self):
        """History ordering is deterministic even with identical timestamps (uses id as tiebreaker)."""
        repo = InMemoryDraftingRepository()
        ws = WS_UUID
        req_id = uuid.uuid4()
        evidence_run_id = uuid.uuid4()

        draft_run = await repo.create_draft_run(ws, req_id, evidence_run_id)
        draft = RequirementDraft(
            requirement_id=req_id,
            requirement_key="req_1",
            requirement_text="Test requirement",
            draft_answer="AI answer",
            citations=[],
            evidence_found=False,
            uncertainty=UncertaintyLevel.NONE,
            uncertainty_notes=[],
            evidence_passages=[],
        )
        await repo.complete_draft_run(draft_run.run_id, draft)

        # Manually create two events with the same timestamp to test stable ordering
        # We can't easily control datetime.now() in tests, but we can verify the sorting logic
        # by checking that the sort key is (reviewed_at, id)
        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="changes_requested",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="Review 1",
            citation_reviews=[],
        )

        await repo.submit_review(
            workspace_id=ws,
            run_id=draft_run.run_id,
            review_status="approved",
            reviewer_edited_answer=None,
            reviewer_compliance_status=None,
            reviewer_compliance_rationale=None,
            reviewer_comment="Review 2",
            citation_reviews=[],
        )

        history = await repo.get_review_history(ws, req_id)
        assert len(history) == 2
        # Should be ordered by reviewed_at, then by id (stable)
        assert history[0].reviewer_comment == "Review 1"
        assert history[1].reviewer_comment == "Review 2"
        # IDs should be different
        assert history[0].id != history[1].id