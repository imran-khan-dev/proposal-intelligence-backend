"""PostgreSQL integration tests for the document ingestion milestone.

These tests are marked ``integration`` and require a running PostgreSQL server
at the DATABASE_URL configured in ``conftest.py`` (the
``proposal_intelligence_test`` database). They are NOT run by the default
``pytest`` invocation; run them explicitly with:

    pytest tests/ -m integration

They verify real database persistence, workspace isolation, and source-reference
resolution through ``PostgreSQLVectorStore``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid

import pytest
from sqlalchemy import text

from app.core.database import (
    get_async_session_factory,
    initialize_database,
    shutdown_database,
)
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.extraction_run import ExtractionRun
from app.models.requirement import Requirement
from app.modules.drafting.assessment import ComplianceStatus
from app.modules.drafting.generation import (
    EvidenceCitation,
    RequirementDraft,
    UncertaintyLevel,
)
from app.modules.drafting.repository import PostgresDraftingRepository
from app.modules.drafting.retrieval import _search_result_to_passage
from app.modules.drafting.verification import (
    REASON_EVIDENCE_ID_NOT_FOUND,
    REASON_VERIFIED,
    REASON_WRONG_WORKSPACE,
    CitationVerificationResult,
    verify_citations,
)
from app.rag.store import VectorRecord
from app.rag.stores.postgres_vector_store import PostgreSQLVectorStore

pytestmark = pytest.mark.integration

TEST_DATABASE_URL = os.environ.get(
    "DATABASE_TEST_URL",
    "postgresql+asyncpg://postgres:45.86.86.86@localhost:5432/proposal_intelligence_test",
)

WS_A = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
WS_B = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")


@pytest.fixture(scope="module")
def event_loop():
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def db_engine():
    """Initialize the test database and run migrations via alembic."""
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    initialize_database()

    # Apply migrations head using the project's alembic config.
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"alembic upgrade head failed:\n{result.stderr}\n{result.stdout}"
        )
    yield
    await shutdown_database()


@pytest.fixture
async def clean_tables(db_engine):
    """Truncate document tables before each test for isolation."""
    factory = get_async_session_factory()
    async with factory() as session:
        await session.execute(text("DELETE FROM document_chunks"))
        await session.execute(text("DELETE FROM documents"))
        await session.execute(text("DELETE FROM tenants"))
        await session.commit()
    yield
    async with factory() as session:
        await session.execute(text("DELETE FROM document_chunks"))
        await session.execute(text("DELETE FROM documents"))
        await session.execute(text("DELETE FROM tenants"))
        await session.commit()


def _make_records(doc_id, workspace_id, pages):
    records = []
    ws_str = str(workspace_id)
    for i, (page_number, text) in enumerate(pages):
        records.append(
            VectorRecord(
                id=f"{doc_id}_chunk_{i}",
                text=text,
                embedding=[0.1] * 384,
                metadata={
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "page_number": page_number,
                    "character_start": 0,
                    "character_end": len(text),
                    "source_filename": f"{doc_id}.txt",
                    "workspace_id": ws_str,
                },
            )
        )
    return records


class TestPostgresWorkspaceIsolation:
    async def test_queries_cannot_leak_across_workspaces(
        self, db_engine, clean_tables
    ):
        store_ws_a = PostgreSQLVectorStore(workspace_id=WS_A)
        store_ws_b = PostgreSQLVectorStore(workspace_id=WS_B)

        await store_ws_a.async_add(
            _make_records("doc-a", WS_A, [(1, "workspace a content")])
        )
        await store_ws_b.async_add(
            _make_records("doc-b", WS_B, [(1, "workspace b content")])
        )

        # Search in workspace-a must not return workspace-b's chunk.
        results = await store_ws_a.async_search([0.1] * 384, k=5)
        assert all(r.record.metadata["workspace_id"] == str(WS_A) for r in results)

        # Count is scoped.
        assert await store_ws_a.async_count() == 1
        assert await store_ws_b.async_count() == 1

    async def test_delete_by_doc_id_scoped_to_workspace(
        self, db_engine, clean_tables
    ):
        store = PostgreSQLVectorStore(workspace_id=WS_A)
        await store.async_add(
            _make_records("doc-a", WS_A, [(1, "content")])
        )
        assert await store.async_count() == 1

        # Deleting a doc that belongs to another workspace must be a no-op.
        other = PostgreSQLVectorStore(workspace_id=WS_B)
        deleted = await other.async_delete_by_metadata({"doc_id": "doc-a"})
        assert deleted == 0
        assert await store.async_count() == 1

    async def test_document_and_chunk_persist_with_workspace_id(
        self, db_engine, clean_tables
    ):
        store = PostgreSQLVectorStore(workspace_id=WS_A)
        await store.async_add(
            _make_records("doc-a", WS_A, [(1, "content")])
        )

        factory = get_async_session_factory()
        async with factory() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(Document).where(Document.workspace_id == store._workspace_id)
            )
            documents = result.scalars().all()
            assert len(documents) == 1
            assert documents[0].doc_id == "doc-a"

            result = await session.execute(
                select(DocumentChunk).where(
                    DocumentChunk.workspace_id == store._workspace_id
                )
            )
            chunks = result.scalars().all()
            assert len(chunks) == 1
            assert chunks[0].chunk_metadata["page_number"] == 1

    async def test_list_get_get_chunks_source_references(
        self, db_engine, clean_tables
    ):
        store = PostgreSQLVectorStore(workspace_id=WS_A)
        pages = [
            (1, "Requirement one text here."),
            (2, "Requirement two text here."),
        ]
        await store.async_add(_make_records("doc-x", WS_A, pages))

        docs = await store.async_list_documents(workspace_id=WS_A)
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "doc-x"
        assert docs[0]["chunk_count"] == 2
        assert docs[0]["pages"] == [1, 2]

        doc = await store.async_get_document("doc-x", workspace_id=WS_A)
        assert doc is not None
        assert doc["doc_id"] == "doc-x"

        chunks = await store.async_get_document_chunks(
            "doc-x", workspace_id=WS_A
        )
        assert len(chunks) == 2
        assert chunks[0]["page_number"] == 1
        assert chunks[1]["page_number"] == 2
        assert chunks[0]["character_start"] == 0
        assert chunks[1]["source_filename"] == "doc-x.txt"

        # Another workspace sees nothing.
        other = PostgreSQLVectorStore(workspace_id=WS_B)
        assert await other.async_list_documents() == []
        assert await other.async_get_document("doc-x") is None
        assert await other.async_get_document_chunks("doc-x") == []


class TestPostgresReIngestion:
    async def test_re_ingestion_replaces_chunks_without_duplicates(
        self, db_engine, clean_tables
    ):
        store = PostgreSQLVectorStore(workspace_id=WS_A)
        await store.async_add(
            _make_records("doc-a", WS_A, [(1, "first version")])
        )
        assert await store.async_count() == 1

        await store.async_add(
            _make_records("doc-a", WS_A, [(1, "second version"), (2, "more")])
        )
        assert await store.async_count() == 2

        chunks = await store.async_get_document_chunks("doc-a")
        assert [c["text"] for c in chunks] == ["second version", "more"]


def _make_typed_records(doc_id, workspace_id, source_type, text):
    """Build a single-chunk record carrying an explicit ``source_type``."""
    return [
        VectorRecord(
            id=f"{doc_id}_chunk_0",
            text=text,
            embedding=[0.1] * 384,
            metadata={
                "doc_id": doc_id,
                "chunk_index": 0,
                "page_number": 1,
                "character_start": 0,
                "character_end": len(text),
                "source_filename": f"{doc_id}.txt",
                "workspace_id": str(workspace_id),
                "source_type": source_type,
            },
        )
    ]


class TestPostgresSourceTypeFilter:
    """The source-type filter must not weaken workspace isolation."""

    async def test_company_only_search_is_workspace_scoped(
        self, db_engine, clean_tables
    ):
        store_a = PostgreSQLVectorStore(workspace_id=WS_A)
        store_b = PostgreSQLVectorStore(workspace_id=WS_B)

        # Workspace A: one RFP doc and one company-knowledge doc.
        await store_a.async_add(
            _make_typed_records("a-rfp", WS_A, "rfp", "rfp requirement text")
        )
        await store_a.async_add(
            _make_typed_records(
                "a-comp", WS_A, "company_knowledge", "company capability text"
            )
        )
        # Workspace B: one company-knowledge doc.
        await store_b.async_add(
            _make_typed_records(
                "b-comp", WS_B, "company_knowledge", "other workspace capability"
            )
        )

        query = [0.1] * 384

        # With the company-knowledge filter, only A's company doc is returned.
        company = await store_a.async_search(
            query, k=5, source_type="company_knowledge"
        )
        assert [r.record.metadata["doc_id"] for r in company] == ["a-comp"]

        # Without the filter, A's RFP doc is also present (proves the filter is
        # what excludes it, not workspace scoping alone).
        all_a = await store_a.async_search(query, k=5)
        assert {r.record.metadata["doc_id"] for r in all_a} == {"a-rfp", "a-comp"}

        # Workspace B's company doc never leaks into A, even with the filter.
        company_b = await store_b.async_search(
            query, k=5, source_type="company_knowledge"
        )
        assert [r.record.metadata["doc_id"] for r in company_b] == ["b-comp"]
        assert all(
            r.record.metadata["workspace_id"] == str(WS_B) for r in company_b
        )


class _FixedEmbedder:
    """Embedder returning a constant vector (dimension 384)."""

    dimension = 384

    def embed(self, texts):
        return [[0.1] * 384 for _ in texts]


class TestCitationVerificationIntegration:
    """End-to-end citation verification against real PostgreSQL evidence."""

    async def test_company_citation_verified_rfp_and_cross_ws_fail(
        self, db_engine, clean_tables
    ):
        store_a = PostgreSQLVectorStore(workspace_id=WS_A)
        store_b = PostgreSQLVectorStore(workspace_id=WS_B)

        await store_a.async_add(
            _make_typed_records(
                "a-comp", WS_A, "company_knowledge",
                "company capability text about SAML single sign-on",
            )
        )
        await store_a.async_add(
            _make_typed_records(
                "a-rfp", WS_A, "rfp", "rfp requirement text about SAML"
            )
        )
        await store_b.async_add(
            _make_typed_records(
                "b-comp", WS_B, "company_knowledge",
                "workspace B company capability text",
            )
        )

        embedder = _FixedEmbedder()

        async def _company_passages(store):
            q = embedder.embed(["company capability SAML single sign-on"])[0]
            results = await store.async_search(
                q, k=5, source_type="company_knowledge"
            )
            return [
                _search_result_to_passage(r, i + 1) for i, r in enumerate(results)
            ]

        # Retrieval is company-knowledge only, scoped to workspace A.
        passages_a = await _company_passages(store_a)
        assert [p.source_doc_id for p in passages_a] == ["a-comp"]
        company_chunk_id = passages_a[0].chunk_id
        assert passages_a[0].metadata.get("workspace_id") == str(WS_A)

        # A valid company-knowledge citation verifies.
        ok = verify_citations(
            [EvidenceCitation(company_chunk_id, "SAML single sign-on", 0.1)],
            passages_a,
            WS_A,
        )
        assert len(ok) == 1 and ok[0].verified is True
        assert ok[0].reason == REASON_VERIFIED

        # A citation to the RFP chunk is not among retrieved evidence -> fails.
        rfp = verify_citations(
            [EvidenceCitation("a-rfp_chunk_0", "rfp requirement text", 0.1)],
            passages_a,
            WS_A,
        )
        assert rfp[0].verified is False
        assert rfp[0].reason == REASON_EVIDENCE_ID_NOT_FOUND

        # Cross-workspace: a WS_B passage verified against WS_A fails.
        passages_b = await _company_passages(store_b)
        assert [p.source_doc_id for p in passages_b] == ["b-comp"]
        cross = verify_citations(
            [EvidenceCitation(passages_b[0].chunk_id, "company capability", 0.1)],
            passages_b,
            WS_A,
        )
        assert cross[0].verified is False
        assert cross[0].reason == REASON_WRONG_WORKSPACE


class TestCompliancePersistenceIntegration:
    """Compliance assessment persists via the existing repository pattern."""

    async def _make_requirement(self) -> uuid.UUID:
        factory = get_async_session_factory()
        async with factory() as session:
            run = ExtractionRun(
                workspace_id=WS_A,
                doc_id="doc1",
                status="completed",
                requirement_count=1,
                errors=[],
            )
            session.add(run)
            await session.flush()
            req = Requirement(
                workspace_id=WS_A,
                run_id=run.id,
                requirement_key="doc1:p1:1-10",
                doc_id="doc1",
                text="The system must support SAML 2.0 single sign-on.",
                page_number=1,
                character_start=1,
                character_end=10,
                modality="mandatory",
                category="security",
                confidence=0.9,
                extraction_method="rule",
            )
            session.add(req)
            await session.commit()
            return req.id

    async def test_compliance_persisted_and_older_records_readable(
        self, db_engine, clean_tables
    ):
        requirement_id = await self._make_requirement()
        repo = PostgresDraftingRepository()

        # Unassessed record remains readable (NULL compliance).
        run = await repo.create_draft_run(WS_A, requirement_id, None)
        fetched = await repo.get_draft_run(run.run_id)
        assert fetched is not None
        assert fetched.compliance_status is None

        # Complete with an assessment.
        draft = RequirementDraft(
            requirement_id=requirement_id,
            requirement_key="doc1:p1:1-10",
            requirement_text="The system must support SAML 2.0 single sign-on.",
            draft_answer="We support SAML 2.0 single sign-on.",
            citations=[EvidenceCitation("chunk_1", "SAML 2.0 single sign-on", 0.9)],
            evidence_found=True,
            uncertainty=UncertaintyLevel.SUPPORTED,
            uncertainty_notes=[],
            evidence_passages=[],
            citation_verifications=[
                CitationVerificationResult(
                    evidence_id="chunk_1",
                    excerpt="SAML 2.0 single sign-on",
                    verified=True,
                    reason=REASON_VERIFIED,
                    source_doc_id="comp_doc",
                    workspace_id=str(WS_A),
                )
            ],
            compliance_status=ComplianceStatus.COMPLIANT,
            compliance_rationale="Verified evidence meets the requirement.",
        )
        await repo.complete_draft_run(run.run_id, draft)

        stored = await repo.get_draft_run(run.run_id)
        assert stored is not None
        assert stored.compliance_status == "compliant"
        assert stored.compliance_rationale == "Verified evidence meets the requirement."
        # Citations (with verification) and answer preserved.
        assert stored.draft_answer == "We support SAML 2.0 single sign-on."
        assert stored.citations[0]["verified"] is True

        # Workspace isolation: another workspace cannot read this run.
        other = await repo.get_latest_draft_run(WS_B, requirement_id)
        assert other is None
