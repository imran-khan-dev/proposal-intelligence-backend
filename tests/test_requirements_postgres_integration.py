"""PostgreSQL integration tests for structured requirement extraction.

Marked ``integration``; requires a running PostgreSQL server at the
``DATABASE_URL`` configured in ``conftest.py``.  Run explicitly with:

    pytest tests/ -m integration

Verifies real persistence of extraction runs and requirements, workspace
isolation, source-reference integrity, and re-extraction (idempotent replace).
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
from app.models.extraction_run import ExtractionRun
from app.models.requirement import Requirement
from app.modules.requirements.repository import PostgresRequirementRepository
from app.modules.requirements.service import extract_from_pages

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
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    initialize_database()

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
    factory = get_async_session_factory()
    async with factory() as session:
        await session.execute(text("DELETE FROM requirements"))
        await session.execute(text("DELETE FROM extraction_runs"))
        await session.execute(text("DELETE FROM document_chunks"))
        await session.execute(text("DELETE FROM documents"))
        await session.execute(text("DELETE FROM tenants"))
        await session.commit()
    yield
    async with factory() as session:
        await session.execute(text("DELETE FROM requirements"))
        await session.execute(text("DELETE FROM extraction_runs"))
        await session.execute(text("DELETE FROM document_chunks"))
        await session.execute(text("DELETE FROM documents"))
        await session.execute(text("DELETE FROM tenants"))
        await session.commit()


def _make_pages():
    text = (
        "1. Scope\n"
        "The contractor shall provide a cloud-based portal.\n"
        "2. Security\n"
        "The system must support SAML 2.0 single sign-on.\n"
    )
    return [
        {
            "page_number": 1,
            "text": text,
            "character_start": 0,
            "character_end": len(text),
        }
    ]


class TestPostgresPersistence:
    async def test_requirements_and_run_persist(self, db_engine, clean_tables):
        repo = PostgresRequirementRepository()
        run, reqs, warnings = await extract_from_pages(
            "doc-a", "rfp.pdf", _make_pages(), WS_A, repo
        )
        assert run.status == "completed"
        assert reqs

        factory = get_async_session_factory()
        async with factory() as session:
            from sqlalchemy import select

            runs = (
                await session.execute(select(ExtractionRun))
            ).scalars().all()
            assert len(runs) == 1
            assert runs[0].doc_id == "doc-a"
            assert runs[0].status == "completed"

            stored = (
                await session.execute(select(Requirement))
            ).scalars().all()
            assert len(stored) == len(reqs)
            # Source references preserved.
            for r in stored:
                assert r.page_number >= 1
                assert r.character_end >= r.character_start
                assert r.modality in {
                    "mandatory", "optional", "informational", "uncertain"
                }

    async def test_workspace_isolation(self, db_engine, clean_tables):
        repo = PostgresRequirementRepository()
        await extract_from_pages("shared", "a.pdf", _make_pages(), WS_A, repo)

        # Workspace B sees nothing for the same doc_id.
        assert await repo.list_requirements(WS_B, "shared") == []
        assert await repo.get_latest_run(WS_B, "shared") is None

        # Workspace A sees its requirements.
        reqs_a = await repo.list_requirements(WS_A, "shared")
        assert reqs_a

    async def test_re_extraction_replaces_without_duplicates(
        self, db_engine, clean_tables
    ):
        repo = PostgresRequirementRepository()
        await extract_from_pages("doc-a", "rfp.pdf", _make_pages(), WS_A, repo)
        first = await repo.list_requirements(WS_A, "doc-a")
        assert first

        await extract_from_pages("doc-a", "rfp.pdf", _make_pages(), WS_A, repo)
        second = await repo.list_requirements(WS_A, "doc-a")
        # Same count, no duplicates from the re-run.
        assert len(second) == len(first)

        factory = get_async_session_factory()
        async with factory() as session:
            from sqlalchemy import select

            runs = (
                await session.execute(select(ExtractionRun))
            ).scalars().all()
            # Two runs recorded, but requirements belong to the latest only.
            assert len(runs) == 2

    async def test_get_single_requirement(self, db_engine, clean_tables):
        repo = PostgresRequirementRepository()
        _, reqs, _ = await extract_from_pages(
            "doc-a", "rfp.pdf", _make_pages(), WS_A, repo
        )
        key = reqs[0].requirement_key
        record = await repo.get_requirement(WS_A, "doc-a", key)
        assert record is not None
        assert record.requirement_key == key
        # Another workspace cannot fetch it.
        assert await repo.get_requirement(WS_B, "doc-a", key) is None