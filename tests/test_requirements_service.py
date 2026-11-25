"""Unit tests for the requirement extraction service.

Uses the in-memory repository and deterministic providers so no database,
network, or API key is required.  Covers persistence, status reporting,
dedup, empty documents, and provider-failure handling.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from fixtures import rfp_fixtures as fx
from app.modules.requirements.repository import InMemoryRequirementRepository
from app.modules.requirements.service import (
    extract_from_chunks,
    extract_from_pages,
    get_requirement,
    get_status,
    list_requirements,
)

WS = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")


def _chunks_from_pages(pages):
    chunks = []
    for p in pages:
        chunks.append(
            {
                "doc_id": "doc1",
                "chunk_index": p["page_number"] - 1,
                "page_number": p["page_number"],
                "character_start": p["character_start"],
                "character_end": p["character_end"],
                "text": p["text"],
                "source_filename": "rfp.pdf",
            }
        )
    return chunks


class TestExtractFromPages:
    @pytest.mark.asyncio
    async def test_persists_requirements_and_status(self):
        repo = InMemoryRequirementRepository()
        run, reqs, warnings = await extract_from_pages(
            "doc1", "rfp.pdf", fx.numbered_fixture(), WS, repo
        )
        assert run.status == "completed"
        assert run.requirement_count == len(reqs)
        assert reqs

        stored = await list_requirements("doc1", WS, repo)
        assert len(stored) == len(reqs)
        status = await get_status("doc1", WS, repo)
        assert status.status == "completed"

    @pytest.mark.asyncio
    async def test_get_single_requirement(self):
        repo = InMemoryRequirementRepository()
        _, reqs, _ = await extract_from_pages(
            "doc1", "rfp.pdf", fx.numbered_fixture(), WS, repo
        )
        key = reqs[0].requirement_key
        record = await get_requirement("doc1", key, WS, repo)
        assert record is not None
        assert record.requirement_key == key

    @pytest.mark.asyncio
    async def test_empty_document_no_requirements(self):
        repo = InMemoryRequirementRepository()
        run, reqs, warnings = await extract_from_pages(
            "doc1", "rfp.pdf", fx.make_pages([""]), WS, repo
        )
        assert reqs == []
        assert run.status == "completed"
        assert any("no extractable text" in w for w in warnings)


class TestExtractFromChunks:
    @pytest.mark.asyncio
    async def test_reconstructs_and_extracts(self):
        repo = InMemoryRequirementRepository()
        chunks = _chunks_from_pages(fx.table_fixture())
        run, reqs, _ = await extract_from_chunks(
            "doc1", chunks, WS, repo, filename="rfp.pdf"
        )
        assert run.status == "completed"
        assert any("REQ-001" in r.text for r in reqs)

    @pytest.mark.asyncio
    async def test_no_chunks_raises_value_error(self):
        repo = InMemoryRequirementRepository()
        with pytest.raises(ValueError, match="No usable source references"):
            await extract_from_chunks("doc1", [], WS, repo)


class TestProviderFailure:
    @pytest.mark.asyncio
    async def test_provider_failure_does_not_crash(self):
        class _BadProvider:
            def generate_structured(self, *a, **k):
                raise RuntimeError("provider down")

            def chat_with_tools(self, *a, **k):
                raise NotImplementedError

        repo = InMemoryRequirementRepository()
        run, reqs, _ = await extract_from_pages(
            "doc1", "rfp.pdf", fx.numbered_fixture(), WS, repo,
            provider=_BadProvider(),
        )
        # Rule-based results are preserved; extraction still completes.
        assert run.status == "completed"
        assert reqs


class TestWorkspaceIsolation:
    @pytest.mark.asyncio
    async def test_requirements_scoped_by_workspace(self):
        repo = InMemoryRequirementRepository()
        ws_a = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
        ws_b = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")
        await extract_from_pages("shared", "a.pdf", fx.numbered_fixture(), ws_a, repo)
        # Workspace B has no run for "shared".
        assert await list_requirements("shared", ws_b, repo) == []
        assert await get_status("shared", ws_b, repo) is None