"""Unit tests for the drafting module (M3).

Tests evidence retrieval, draft generation, and persistence with mocked providers.
No external AI APIs or database required.
"""

from __future__ import annotations

import uuid
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
from app.modules.drafting.retrieval import (
    EvidencePassage,
    retrieve_evidence_for_requirement,
)
from app.modules.drafting.repository import InMemoryDraftingRepository
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