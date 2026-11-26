"""Persistence for extracted requirements and extraction runs.

Two implementations:

- :class:`PostgresRequirementRepository` — production, uses the shared async
  engine/session factory and the SQLAlchemy models.  Workspace-scoped.
- :class:`InMemoryRequirementRepository` — deterministic, dependency-free
  store used by unit tests.

Both return Pydantic records from :mod:`app.modules.requirements.schema` so
the service layer is persistence-agnostic.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from sqlalchemy import select

from app.core.database import get_async_session_factory
from app.models.document import Document
from app.models.extraction_run import ExtractionRun
from app.models.requirement import Requirement
from app.modules.requirements.extraction import ExtractedRequirement
from app.modules.requirements.schema import (
    ExtractionRunRecord,
    RequirementRecord,
)


class RequirementRepository(ABC):
    """Abstract persistence interface for requirements."""

    @abstractmethod
    async def create_run(
        self, workspace_id: uuid.UUID, doc_id: str
    ) -> ExtractionRunRecord:
        """Create a pending run and return its record."""
        ...

    @abstractmethod
    async def save_requirements(
        self,
        run_id: str,
        workspace_id: uuid.UUID,
        doc_id: str,
        requirements: List[ExtractedRequirement],
        document_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Persist requirements for a run (replacing any prior rows)."""
        ...

    @abstractmethod
    async def complete_run(self, run_id: str, requirement_count: int) -> None:
        """Mark a run completed with the given requirement count."""
        ...

    @abstractmethod
    async def fail_run(self, run_id: str, errors: List[str]) -> None:
        """Mark a run failed with error details."""
        ...

    @abstractmethod
    async def get_latest_run(
        self, workspace_id: uuid.UUID, doc_id: str
    ) -> Optional[ExtractionRunRecord]:
        """Return the most recent run for a document, if any."""
        ...

    @abstractmethod
    async def list_requirements(
        self, workspace_id: uuid.UUID, doc_id: str
    ) -> List[RequirementRecord]:
        """List requirements for the latest completed run of a document."""
        ...

    @abstractmethod
    async def get_requirement(
        self, workspace_id: uuid.UUID, doc_id: str, requirement_key: str
    ) -> Optional[RequirementRecord]:
        """Return a single requirement by key within a workspace."""
        ...

    @abstractmethod
    async def get_requirement_id(
        self, workspace_id: uuid.UUID, doc_id: str, requirement_key: str
    ) -> Optional[uuid.UUID]:
        """Return the primary key UUID of a requirement by key."""
        ...


def _run_to_record(run: ExtractionRun) -> ExtractionRunRecord:
    return ExtractionRunRecord(
        run_id=str(run.id),
        doc_id=run.doc_id,
        status=run.status,
        requirement_count=run.requirement_count,
        errors=list(run.errors or []),
    )


def _req_to_record(req: Requirement) -> RequirementRecord:
    return RequirementRecord(
        requirement_key=req.requirement_key,
        doc_id=req.doc_id,
        text=req.text,
        section_title=req.section_title,
        section_path=req.section_path,
        page_number=req.page_number,
        character_start=req.character_start,
        character_end=req.character_end,
        modality=req.modality,
        category=req.category,
        cross_references=list(req.cross_references or []),
        confidence=req.confidence,
        uncertainty_notes=list(req.uncertainty_notes or []),
        extraction_method=req.extraction_method,
    )


class PostgresRequirementRepository(RequirementRepository):
    """PostgreSQL-backed repository using the shared async engine."""

    def __init__(self) -> None:
        self._session_factory = get_async_session_factory()

    async def _resolve_document_id(
        self, session, workspace_id: uuid.UUID, doc_id: str
    ) -> Optional[uuid.UUID]:
        stmt = select(Document).where(
            Document.workspace_id == workspace_id,
            Document.doc_id == doc_id,
        )
        result = await session.execute(stmt)
        document = result.scalar_one_or_none()
        return document.id if document is not None else None

    async def create_run(
        self, workspace_id: uuid.UUID, doc_id: str
    ) -> ExtractionRunRecord:
        async with self._session_factory() as session:
            document_id = await self._resolve_document_id(
                session, workspace_id, doc_id
            )
            run = ExtractionRun(
                workspace_id=workspace_id,
                document_id=document_id,
                doc_id=doc_id,
                status="running",
                requirement_count=0,
                errors=[],
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return _run_to_record(run)

    async def save_requirements(
        self,
        run_id: str,
        workspace_id: uuid.UUID,
        doc_id: str,
        requirements: List[ExtractedRequirement],
        document_id: Optional[uuid.UUID] = None,
    ) -> None:
        run_uuid = uuid.UUID(run_id)
        async with self._session_factory() as session:
            if document_id is None:
                document_id = await self._resolve_document_id(
                    session, workspace_id, doc_id
                )
            # Replace any prior rows for this run (idempotent re-run).
            await session.execute(
                Requirement.__table__.delete().where(
                    Requirement.run_id == run_uuid
                )
            )
            for req in requirements:
                session.add(
                    Requirement(
                        workspace_id=workspace_id,
                        document_id=document_id,
                        run_id=run_uuid,
                        requirement_key=req.requirement_key,
                        doc_id=doc_id,
                        text=req.text,
                        section_title=req.section_title,
                        section_path=req.section_path,
                        page_number=req.page_number,
                        character_start=req.character_start,
                        character_end=req.character_end,
                        modality=req.modality.value,
                        category=req.category.value,
                        cross_references=list(req.cross_references),
                        confidence=req.confidence,
                        uncertainty_notes=list(req.uncertainty_notes),
                        extraction_method=req.extraction_method,
                    )
                )
            await session.commit()

    async def complete_run(self, run_id: str, requirement_count: int) -> None:
        async with self._session_factory() as session:
            run = await session.get(ExtractionRun, uuid.UUID(run_id))
            if run is not None:
                run.status = "completed"
                run.requirement_count = requirement_count
                await session.commit()

    async def fail_run(self, run_id: str, errors: List[str]) -> None:
        async with self._session_factory() as session:
            run = await session.get(ExtractionRun, uuid.UUID(run_id))
            if run is not None:
                run.status = "failed"
                run.errors = errors
                await session.commit()

    async def get_latest_run(
        self, workspace_id: uuid.UUID, doc_id: str
    ) -> Optional[ExtractionRunRecord]:
        async with self._session_factory() as session:
            stmt = (
                select(ExtractionRun)
                .where(
                    ExtractionRun.workspace_id == workspace_id,
                    ExtractionRun.doc_id == doc_id,
                )
                .order_by(ExtractionRun.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            run = result.scalar_one_or_none()
            return _run_to_record(run) if run is not None else None

    async def _latest_run_model(
        self, session, workspace_id: uuid.UUID, doc_id: str
    ) -> Optional[ExtractionRun]:
        stmt = (
            select(ExtractionRun)
            .where(
                ExtractionRun.workspace_id == workspace_id,
                ExtractionRun.doc_id == doc_id,
            )
            .order_by(ExtractionRun.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_requirements(
        self, workspace_id: uuid.UUID, doc_id: str
    ) -> List[RequirementRecord]:
        async with self._session_factory() as session:
            run = await self._latest_run_model(session, workspace_id, doc_id)
            if run is None:
                return []
            stmt = (
                select(Requirement)
                .where(Requirement.run_id == run.id)
                .order_by(
                    Requirement.page_number,
                    Requirement.character_start,
                )
            )
            result = await session.execute(stmt)
            return [_req_to_record(r) for r in result.scalars().all()]

    async def get_requirement(
        self, workspace_id: uuid.UUID, doc_id: str, requirement_key: str
    ) -> Optional[RequirementRecord]:
        async with self._session_factory() as session:
            stmt = (
                select(Requirement)
                .join(ExtractionRun, Requirement.run_id == ExtractionRun.id)
                .where(
                    Requirement.workspace_id == workspace_id,
                    Requirement.doc_id == doc_id,
                    Requirement.requirement_key == requirement_key,
                )
                .order_by(ExtractionRun.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            req = result.scalar_one_or_none()
            return _req_to_record(req) if req is not None else None

    async def get_requirement_id(
        self, workspace_id: uuid.UUID, doc_id: str, requirement_key: str
    ) -> Optional[uuid.UUID]:
        """Return the primary key UUID of a requirement by key."""
        async with self._session_factory() as session:
            stmt = (
                select(Requirement.id)
                .join(ExtractionRun, Requirement.run_id == ExtractionRun.id)
                .where(
                    Requirement.workspace_id == workspace_id,
                    Requirement.doc_id == doc_id,
                    Requirement.requirement_key == requirement_key,
                )
                .order_by(ExtractionRun.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()


class InMemoryRequirementRepository(RequirementRepository):
    """Deterministic in-memory repository for tests."""

    def __init__(self) -> None:
        self._runs: dict[str, ExtractionRunRecord] = {}
        self._run_order: List[str] = []
        self._run_workspaces: dict[str, uuid.UUID] = {}
        self._requirements: dict[str, List[RequirementRecord]] = {}

    async def create_run(
        self, workspace_id: uuid.UUID, doc_id: str
    ) -> ExtractionRunRecord:
        run_id = str(uuid.uuid4())
        record = ExtractionRunRecord(
            run_id=run_id,
            doc_id=doc_id,
            status="running",
            requirement_count=0,
            errors=[],
        )
        self._runs[run_id] = record
        self._run_workspaces[run_id] = workspace_id
        self._run_order.append(run_id)
        return record

    async def save_requirements(
        self,
        run_id: str,
        workspace_id: uuid.UUID,
        doc_id: str,
        requirements: List[ExtractedRequirement],
        document_id: Optional[uuid.UUID] = None,
    ) -> None:
        self._requirements[run_id] = [
            RequirementRecord(
                requirement_key=r.requirement_key,
                doc_id=doc_id,
                text=r.text,
                section_title=r.section_title,
                section_path=r.section_path,
                page_number=r.page_number,
                character_start=r.character_start,
                character_end=r.character_end,
                modality=r.modality,
                category=r.category,
                cross_references=list(r.cross_references),
                confidence=r.confidence,
                uncertainty_notes=list(r.uncertainty_notes),
                extraction_method=r.extraction_method,
            )
            for r in requirements
        ]

    async def complete_run(self, run_id: str, requirement_count: int) -> None:
        record = self._runs.get(run_id)
        if record is not None:
            record.status = "completed"
            record.requirement_count = requirement_count

    async def fail_run(self, run_id: str, errors: List[str]) -> None:
        record = self._runs.get(run_id)
        if record is not None:
            record.status = "failed"
            record.errors = errors

    def _latest_run_id(self, workspace_id: uuid.UUID, doc_id: str) -> Optional[str]:
        for run_id in reversed(self._run_order):
            record = self._runs[run_id]
            if (
                record.doc_id == doc_id
                and self._run_workspaces.get(run_id) == workspace_id
            ):
                return run_id
        return None

    async def get_latest_run(
        self, workspace_id: uuid.UUID, doc_id: str
    ) -> Optional[ExtractionRunRecord]:
        run_id = self._latest_run_id(workspace_id, doc_id)
        return self._runs.get(run_id) if run_id else None

    async def list_requirements(
        self, workspace_id: uuid.UUID, doc_id: str
    ) -> List[RequirementRecord]:
        run_id = self._latest_run_id(workspace_id, doc_id)
        if run_id is None:
            return []
        return list(self._requirements.get(run_id, []))

    async def get_requirement(
        self, workspace_id: uuid.UUID, doc_id: str, requirement_key: str
    ) -> Optional[RequirementRecord]:
        for req in await self.list_requirements(workspace_id, doc_id):
            if req.requirement_key == requirement_key:
                return req
        return None

    async def get_requirement_id(
        self, workspace_id: uuid.UUID, doc_id: str, requirement_key: str
    ) -> Optional[uuid.UUID]:
        # In-memory doesn't have persistent UUIDs; generate deterministic one
        # based on the requirement key for consistency within a test run
        for req in await self.list_requirements(workspace_id, doc_id):
            if req.requirement_key == requirement_key:
                # Generate a deterministic UUID from the requirement key
                return uuid.uuid5(uuid.NAMESPACE_DNS, requirement_key)
        return None