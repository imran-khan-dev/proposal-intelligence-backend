"""Persistence for evidence retrieval runs, evidence records, and draft generation runs.

Two implementations:
- PostgresDraftingRepository — production, uses the shared async engine/session.
- InMemoryDraftingRepository — deterministic, dependency-free for unit tests.

Both return Pydantic records so the service layer is persistence-agnostic.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from sqlalchemy import select

from app.core.database import get_async_session_factory
from app.models.drafting import (
    DraftGenerationRun,
    EvidenceRecord,
    EvidenceRetrievalRun,
)
from app.modules.drafting.generation import EvidenceCitation, RequirementDraft, UncertaintyLevel
from app.modules.drafting.retrieval import EvidencePassage
from app.modules.drafting.service import DraftingResult
from app.modules.requirements.schema import RequirementRecord


class DraftingRepository(ABC):
    """Abstract persistence interface for drafting."""

    # --- Evidence Retrieval Runs ---

    @abstractmethod
    async def create_evidence_run(
        self,
        workspace_id: uuid.UUID,
        requirement_id: uuid.UUID,
        query_text: str,
    ) -> "EvidenceRetrievalRunRecord":
        """Create a pending evidence retrieval run."""
        ...

    @abstractmethod
    async def complete_evidence_run(
        self,
        run_id: uuid.UUID,
        evidence_count: int,
        evidence_passages: List[EvidencePassage],
    ) -> None:
        """Mark an evidence run completed with its passages."""
        ...

    @abstractmethod
    async def fail_evidence_run(self, run_id: uuid.UUID, errors: List[str]) -> None:
        """Mark an evidence run failed."""
        ...

    @abstractmethod
    async def get_evidence_run(
        self, run_id: uuid.UUID
    ) -> Optional["EvidenceRetrievalRunRecord"]:
        """Get an evidence retrieval run by ID."""
        ...

    @abstractmethod
    async def get_latest_evidence_run(
        self, workspace_id: uuid.UUID, requirement_id: uuid.UUID
    ) -> Optional["EvidenceRetrievalRunRecord"]:
        """Get the latest evidence run for a requirement."""
        ...

    @abstractmethod
    async def list_evidence_for_run(
        self, run_id: uuid.UUID
    ) -> List["EvidenceRecordRecord"]:
        """List evidence records for a run."""
        ...

    # --- Draft Generation Runs ---

    @abstractmethod
    async def create_draft_run(
        self,
        workspace_id: uuid.UUID,
        requirement_id: uuid.UUID,
        retrieval_run_id: Optional[uuid.UUID],
    ) -> "DraftGenerationRunRecord":
        """Create a pending draft generation run."""
        ...

    @abstractmethod
    async def complete_draft_run(
        self,
        run_id: uuid.UUID,
        draft: RequirementDraft,
    ) -> None:
        """Mark a draft run completed with the generated draft."""
        ...

    @abstractmethod
    async def fail_draft_run(self, run_id: uuid.UUID, errors: List[str]) -> None:
        """Mark a draft run failed."""
        ...

    @abstractmethod
    async def get_draft_run(
        self, run_id: uuid.UUID
    ) -> Optional["DraftGenerationRunRecord"]:
        """Get a draft generation run by ID."""
        ...

    @abstractmethod
    async def get_latest_draft_run(
        self, workspace_id: uuid.UUID, requirement_id: uuid.UUID
    ) -> Optional["DraftGenerationRunRecord"]:
        """Get the latest draft run for a requirement."""
        ...

    @abstractmethod
    async def list_draft_runs_for_requirement(
        self, workspace_id: uuid.UUID, requirement_id: uuid.UUID
    ) -> List["DraftGenerationRunRecord"]:
        """List all draft runs for a requirement (for history)."""
        ...


# --- Pydantic Records for API/Service Layer ---

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class EvidenceRetrievalRunRecord(BaseModel):
    run_id: uuid.UUID
    workspace_id: uuid.UUID
    requirement_id: uuid.UUID
    query_text: str
    status: str
    evidence_count: int
    errors: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class EvidenceRecordRecord(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    workspace_id: uuid.UUID
    source_doc_id: str
    source_filename: Optional[str]
    chunk_id: Optional[str]
    chunk_index: Optional[int]
    page_number: Optional[int]
    character_start: Optional[int]
    character_end: Optional[int]
    section_path: Optional[str]
    text: str
    score: float
    rank: int
    created_at: datetime


class DraftGenerationRunRecord(BaseModel):
    run_id: uuid.UUID
    workspace_id: uuid.UUID
    requirement_id: uuid.UUID
    retrieval_run_id: Optional[uuid.UUID]
    draft_answer: Optional[str]
    citations: List[dict] = Field(default_factory=list)
    evidence_found: bool
    uncertainty: str
    uncertainty_notes: List[str] = Field(default_factory=list)
    status: str
    errors: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


# --- Conversion Helpers ---

def _evidence_run_to_record(run: EvidenceRetrievalRun) -> EvidenceRetrievalRunRecord:
    return EvidenceRetrievalRunRecord(
        run_id=run.id,
        workspace_id=run.workspace_id,
        requirement_id=run.requirement_id,
        query_text=run.query_text,
        status=run.status,
        evidence_count=run.evidence_count,
        errors=list(run.errors or []),
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def _evidence_to_record(ev: EvidenceRecord) -> EvidenceRecordRecord:
    return EvidenceRecordRecord(
        id=ev.id,
        run_id=ev.run_id,
        workspace_id=ev.workspace_id,
        source_doc_id=ev.source_doc_id,
        source_filename=ev.source_filename,
        chunk_id=ev.chunk_id,
        chunk_index=ev.chunk_index,
        page_number=ev.page_number,
        character_start=ev.character_start,
        character_end=ev.character_end,
        section_path=ev.section_path,
        text=ev.text,
        score=ev.score,
        rank=ev.rank,
        created_at=ev.created_at,
    )


def _draft_run_to_record(run: DraftGenerationRun) -> DraftGenerationRunRecord:
    return DraftGenerationRunRecord(
        run_id=run.id,
        workspace_id=run.workspace_id,
        requirement_id=run.requirement_id,
        retrieval_run_id=run.retrieval_run_id,
        draft_answer=run.draft_answer,
        citations=list(run.citations or []),
        evidence_found=run.evidence_found,
        uncertainty=run.uncertainty,
        uncertainty_notes=list(run.uncertainty_notes or []),
        status=run.status,
        errors=list(run.errors or []),
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


# --- PostgreSQL Implementation ---

class PostgresDraftingRepository(DraftingRepository):
    """PostgreSQL-backed repository using the shared async engine."""

    def __init__(self) -> None:
        self._session_factory = get_async_session_factory()

    # Evidence Retrieval Runs

    async def create_evidence_run(
        self,
        workspace_id: uuid.UUID,
        requirement_id: uuid.UUID,
        query_text: str,
    ) -> EvidenceRetrievalRunRecord:
        async with self._session_factory() as session:
            run = EvidenceRetrievalRun(
                workspace_id=workspace_id,
                requirement_id=requirement_id,
                query_text=query_text,
                status="running",
                evidence_count=0,
                errors=[],
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return _evidence_run_to_record(run)

    async def complete_evidence_run(
        self,
        run_id: uuid.UUID,
        evidence_count: int,
        evidence_passages: List[EvidencePassage],
    ) -> None:
        async with self._session_factory() as session:
            run = await session.get(EvidenceRetrievalRun, run_id)
            if run is not None:
                run.status = "completed"
                run.evidence_count = evidence_count
                # Add evidence records
                for passage in evidence_passages:
                    session.add(
                        EvidenceRecord(
                            run_id=run_id,
                            workspace_id=run.workspace_id,
                            source_doc_id=passage.source_doc_id,
                            source_filename=passage.source_filename,
                            chunk_id=passage.chunk_id,
                            chunk_index=passage.chunk_index,
                            page_number=passage.page_number,
                            character_start=passage.character_start,
                            character_end=passage.character_end,
                            section_path=passage.section_path,
                            text=passage.text,
                            score=passage.score,
                            rank=passage.rank,
                        )
                    )
                await session.commit()

    async def fail_evidence_run(self, run_id: uuid.UUID, errors: List[str]) -> None:
        async with self._session_factory() as session:
            run = await session.get(EvidenceRetrievalRun, run_id)
            if run is not None:
                run.status = "failed"
                run.errors = errors
                await session.commit()

    async def get_evidence_run(
        self, run_id: uuid.UUID
    ) -> Optional[EvidenceRetrievalRunRecord]:
        async with self._session_factory() as session:
            run = await session.get(EvidenceRetrievalRun, run_id)
            return _evidence_run_to_record(run) if run else None

    async def get_latest_evidence_run(
        self, workspace_id: uuid.UUID, requirement_id: uuid.UUID
    ) -> Optional[EvidenceRetrievalRunRecord]:
        async with self._session_factory() as session:
            stmt = (
                select(EvidenceRetrievalRun)
                .where(
                    EvidenceRetrievalRun.workspace_id == workspace_id,
                    EvidenceRetrievalRun.requirement_id == requirement_id,
                )
                .order_by(EvidenceRetrievalRun.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            run = result.scalar_one_or_none()
            return _evidence_run_to_record(run) if run else None

    async def list_evidence_for_run(
        self, run_id: uuid.UUID
    ) -> List[EvidenceRecordRecord]:
        async with self._session_factory() as session:
            stmt = (
                select(EvidenceRecord)
                .where(EvidenceRecord.run_id == run_id)
                .order_by(EvidenceRecord.rank)
            )
            result = await session.execute(stmt)
            return [_evidence_to_record(e) for e in result.scalars().all()]

    # Draft Generation Runs

    async def create_draft_run(
        self,
        workspace_id: uuid.UUID,
        requirement_id: uuid.UUID,
        retrieval_run_id: Optional[uuid.UUID],
    ) -> DraftGenerationRunRecord:
        async with self._session_factory() as session:
            run = DraftGenerationRun(
                workspace_id=workspace_id,
                requirement_id=requirement_id,
                retrieval_run_id=retrieval_run_id,
                status="running",
                draft_answer=None,
                citations=[],
                evidence_found=False,
                uncertainty="none",
                uncertainty_notes=[],
                errors=[],
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return _draft_run_to_record(run)

    async def complete_draft_run(
        self,
        run_id: uuid.UUID,
        draft: RequirementDraft,
    ) -> None:
        async with self._session_factory() as session:
            run = await session.get(DraftGenerationRun, run_id)
            if run is not None:
                run.status = "completed"
                run.draft_answer = draft.draft_answer
                run.citations = [
                    {
                        "evidence_id": c.evidence_id,
                        "excerpt": c.excerpt,
                        "score": c.score,
                    }
                    for c in draft.citations
                ]
                run.evidence_found = draft.evidence_found
                run.uncertainty = draft.uncertainty.value
                run.uncertainty_notes = list(draft.uncertainty_notes)
                await session.commit()

    async def fail_draft_run(self, run_id: uuid.UUID, errors: List[str]) -> None:
        async with self._session_factory() as session:
            run = await session.get(DraftGenerationRun, run_id)
            if run is not None:
                run.status = "failed"
                run.errors = errors
                await session.commit()

    async def get_draft_run(
        self, run_id: uuid.UUID
    ) -> Optional[DraftGenerationRunRecord]:
        async with self._session_factory() as session:
            run = await session.get(DraftGenerationRun, run_id)
            return _draft_run_to_record(run) if run else None

    async def get_latest_draft_run(
        self, workspace_id: uuid.UUID, requirement_id: uuid.UUID
    ) -> Optional[DraftGenerationRunRecord]:
        async with self._session_factory() as session:
            stmt = (
                select(DraftGenerationRun)
                .where(
                    DraftGenerationRun.workspace_id == workspace_id,
                    DraftGenerationRun.requirement_id == requirement_id,
                )
                .order_by(DraftGenerationRun.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            run = result.scalar_one_or_none()
            return _draft_run_to_record(run) if run else None

    async def list_draft_runs_for_requirement(
        self, workspace_id: uuid.UUID, requirement_id: uuid.UUID
    ) -> List[DraftGenerationRunRecord]:
        async with self._session_factory() as session:
            stmt = (
                select(DraftGenerationRun)
                .where(
                    DraftGenerationRun.workspace_id == workspace_id,
                    DraftGenerationRun.requirement_id == requirement_id,
                )
                .order_by(DraftGenerationRun.created_at.desc())
            )
            result = await session.execute(stmt)
            return [_draft_run_to_record(r) for r in result.scalars().all()]


# --- In-Memory Implementation for Tests ---

class InMemoryDraftingRepository(DraftingRepository):
    """Deterministic in-memory repository for tests."""

    def __init__(self) -> None:
        self._evidence_runs: dict[uuid.UUID, EvidenceRetrievalRunRecord] = {}
        self._evidence_records: dict[uuid.UUID, List[EvidenceRecordRecord]] = {}
        self._draft_runs: dict[uuid.UUID, DraftGenerationRunRecord] = {}

    # Evidence Retrieval Runs

    async def create_evidence_run(
        self,
        workspace_id: uuid.UUID,
        requirement_id: uuid.UUID,
        query_text: str,
    ) -> EvidenceRetrievalRunRecord:
        run_id = uuid.uuid4()
        record = EvidenceRetrievalRunRecord(
            run_id=run_id,
            workspace_id=workspace_id,
            requirement_id=requirement_id,
            query_text=query_text,
            status="running",
            evidence_count=0,
            errors=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self._evidence_runs[run_id] = record
        self._evidence_records[run_id] = []
        return record

    async def complete_evidence_run(
        self,
        run_id: uuid.UUID,
        evidence_count: int,
        evidence_passages: List[EvidencePassage],
    ) -> None:
        record = self._evidence_runs.get(run_id)
        if record:
            record.status = "completed"
            record.evidence_count = evidence_count
            record.updated_at = datetime.now()
            self._evidence_records[run_id] = [
                EvidenceRecordRecord(
                    id=uuid.uuid4(),
                    run_id=run_id,
                    workspace_id=record.workspace_id,
                    source_doc_id=p.source_doc_id,
                    source_filename=p.source_filename,
                    chunk_id=p.chunk_id,
                    chunk_index=p.chunk_index,
                    page_number=p.page_number,
                    character_start=p.character_start,
                    character_end=p.character_end,
                    section_path=p.section_path,
                    text=p.text,
                    score=p.score,
                    rank=p.rank,
                    created_at=datetime.now(),
                )
                for p in evidence_passages
            ]

    async def fail_evidence_run(self, run_id: uuid.UUID, errors: List[str]) -> None:
        record = self._evidence_runs.get(run_id)
        if record:
            record.status = "failed"
            record.errors = errors
            record.updated_at = datetime.now()

    async def get_evidence_run(
        self, run_id: uuid.UUID
    ) -> Optional[EvidenceRetrievalRunRecord]:
        return self._evidence_runs.get(run_id)

    async def get_latest_evidence_run(
        self, workspace_id: uuid.UUID, requirement_id: uuid.UUID
    ) -> Optional[EvidenceRetrievalRunRecord]:
        for run in reversed(list(self._evidence_runs.values())):
            if run.workspace_id == workspace_id and run.requirement_id == requirement_id:
                return run
        return None

    async def list_evidence_for_run(
        self, run_id: uuid.UUID
    ) -> List[EvidenceRecordRecord]:
        return self._evidence_records.get(run_id, [])

    # Draft Generation Runs

    async def create_draft_run(
        self,
        workspace_id: uuid.UUID,
        requirement_id: uuid.UUID,
        retrieval_run_id: Optional[uuid.UUID],
    ) -> DraftGenerationRunRecord:
        run_id = uuid.uuid4()
        record = DraftGenerationRunRecord(
            run_id=run_id,
            workspace_id=workspace_id,
            requirement_id=requirement_id,
            retrieval_run_id=retrieval_run_id,
            status="running",
            draft_answer=None,
            citations=[],
            evidence_found=False,
            uncertainty="none",
            uncertainty_notes=[],
            errors=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self._draft_runs[run_id] = record
        return record

    async def complete_draft_run(
        self,
        run_id: uuid.UUID,
        draft: RequirementDraft,
    ) -> None:
        record = self._draft_runs.get(run_id)
        if record:
            record.status = "completed"
            record.draft_answer = draft.draft_answer
            record.citations = [
                {"evidence_id": c.evidence_id, "excerpt": c.excerpt, "score": c.score}
                for c in draft.citations
            ]
            record.evidence_found = draft.evidence_found
            record.uncertainty = draft.uncertainty.value
            record.uncertainty_notes = list(draft.uncertainty_notes)
            record.updated_at = datetime.now()

    async def fail_draft_run(self, run_id: uuid.UUID, errors: List[str]) -> None:
        record = self._draft_runs.get(run_id)
        if record:
            record.status = "failed"
            record.errors = errors
            record.updated_at = datetime.now()

    async def get_draft_run(
        self, run_id: uuid.UUID
    ) -> Optional[DraftGenerationRunRecord]:
        return self._draft_runs.get(run_id)

    async def get_latest_draft_run(
        self, workspace_id: uuid.UUID, requirement_id: uuid.UUID
    ) -> Optional[DraftGenerationRunRecord]:
        for run in reversed(list(self._draft_runs.values())):
            if run.workspace_id == workspace_id and run.requirement_id == requirement_id:
                return run
        return None

    async def list_draft_runs_for_requirement(
        self, workspace_id: uuid.UUID, requirement_id: uuid.UUID
    ) -> List[DraftGenerationRunRecord]:
        return [
            r for r in self._draft_runs.values()
            if r.workspace_id == workspace_id and r.requirement_id == requirement_id
        ]