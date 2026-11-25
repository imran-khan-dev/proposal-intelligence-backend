"""Unit tests for the requirements controller.

The repository, store, and chunk lookups are patched so no database, network,
or API key is required.  Covers extraction success, 404/422 handling, provider
failure tolerance, listing, single-retrieval, status, and server-side workspace
resolution.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
from fastapi import HTTPException

from fixtures import rfp_fixtures as fx
from app.modules.requirements.controller import (
    extract_requirements_controller,
    get_requirement_controller,
    get_status_controller,
    list_requirements_controller,
)
from app.modules.requirements.repository import InMemoryRequirementRepository

WS = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")


def _chunks_from_pages(pages):
    return [
        {
            "doc_id": "doc1",
            "chunk_index": p["page_number"] - 1,
            "page_number": p["page_number"],
            "character_start": p["character_start"],
            "character_end": p["character_end"],
            "text": p["text"],
            "source_filename": "rfp.pdf",
        }
        for p in pages
    ]


def _patch_common(repo, chunks):
    """Patch controller dependencies with an in-memory repo and fixed chunks."""
    return (
        patch(
            "app.modules.requirements.controller.PostgresRequirementRepository",
            return_value=repo,
        ),
        patch(
            "app.modules.requirements.controller.async_get_document_chunks",
            new=AsyncMock(return_value=chunks),
        ),
        patch(
            "app.modules.requirements.controller._scoped_store",
            return_value=MagicMock(),
        ),
        patch(
            "app.modules.requirements.controller._get_provider_optional",
            return_value=None,
        ),
    )


class TestExtractController:
    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        repo = InMemoryRequirementRepository()
        p1, p2, p3, p4 = _patch_common(repo, _chunks_from_pages(fx.numbered_fixture()))
        with p1, p2, p3, p4:
            resp = await extract_requirements_controller("doc1")
        assert resp.doc_id == "doc1"
        assert resp.status == "completed"
        assert resp.requirement_count > 0
        assert resp.run_id

    @pytest.mark.asyncio
    async def test_document_not_found_404(self):
        repo = InMemoryRequirementRepository()
        p1, p2, p3, p4 = _patch_common(repo, [])
        with p1, p2, p3, p4:
            with pytest.raises(HTTPException) as exc:
                await extract_requirements_controller("missing")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_no_usable_refs_422(self):
        repo = InMemoryRequirementRepository()
        bad_chunks = [{"text": "no offsets", "page_number": 1}]
        p1, p2, p3, p4 = _patch_common(repo, bad_chunks)
        with p1, p2, p3, p4:
            with pytest.raises(HTTPException) as exc:
                await extract_requirements_controller("doc1")
        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_provider_failure_tolerated(self):
        class _BadProvider:
            def generate_structured(self, *a, **k):
                raise RuntimeError("down")

            def chat_with_tools(self, *a, **k):
                raise NotImplementedError

        repo = InMemoryRequirementRepository()
        p1, p2, p3, p4 = _patch_common(repo, _chunks_from_pages(fx.numbered_fixture()))
        with p1, p2, p3, patch(
            "app.modules.requirements.controller._get_provider_optional",
            return_value=_BadProvider(),
        ):
            resp = await extract_requirements_controller("doc1")
        assert resp.status == "completed"


class TestListAndGetController:
    @pytest.mark.asyncio
    async def test_list_after_extract(self):
        repo = InMemoryRequirementRepository()
        chunks = _chunks_from_pages(fx.numbered_fixture())
        p1, p2, p3, p4 = _patch_common(repo, chunks)
        with p1, p2, p3, p4:
            await extract_requirements_controller("doc1")
            listing = await list_requirements_controller("doc1")
        assert listing.doc_id == "doc1"
        assert len(listing.requirements) > 0

    @pytest.mark.asyncio
    async def test_get_requirement_found_and_missing(self):
        repo = InMemoryRequirementRepository()
        chunks = _chunks_from_pages(fx.numbered_fixture())
        p1, p2, p3, p4 = _patch_common(repo, chunks)
        with p1, p2, p3, p4:
            await extract_requirements_controller("doc1")
            listing = await list_requirements_controller("doc1")
            key = listing.requirements[0].requirement_key
            detail = await get_requirement_controller("doc1", key)
            assert detail.requirement.requirement_key == key
            assert detail.source.page_number == detail.requirement.page_number

            with pytest.raises(HTTPException) as exc:
                await get_requirement_controller("doc1", "does-not-exist")
        assert exc.value.status_code == 404


class TestStatusController:
    @pytest.mark.asyncio
    async def test_status_after_extract(self):
        repo = InMemoryRequirementRepository()
        p1, p2, p3, p4 = _patch_common(repo, _chunks_from_pages(fx.numbered_fixture()))
        with p1, p2, p3, p4:
            await extract_requirements_controller("doc1")
            status = await get_status_controller("doc1")
        assert status.status == "completed"
        assert status.requirement_count > 0

    @pytest.mark.asyncio
    async def test_status_not_found(self):
        repo = InMemoryRequirementRepository()
        p1, p2, p3, p4 = _patch_common(repo, [])
        with p1, p2, p3, p4:
            with pytest.raises(HTTPException) as exc:
                await get_status_controller("missing")
        assert exc.value.status_code == 404


class TestWorkspaceResolution:
    @pytest.mark.asyncio
    async def test_client_workspace_ignored(self):
        # Controller resolves workspace server-side; a client-supplied value
        # that is not a valid UUID falls back to the resolved default.
        repo = InMemoryRequirementRepository()
        p1, p2, p3, p4 = _patch_common(repo, _chunks_from_pages(fx.numbered_fixture()))
        with p1, p2, p3, p4:
            resp = await extract_requirements_controller(
                "doc1", workspace_id="not-a-uuid"
            )
        assert resp.status == "completed"