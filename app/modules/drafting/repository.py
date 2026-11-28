"""Persistence for evidence retrieval runs, evidence records, and draft generation runs.

Two implementations:
- PostgresDraftingRepository — production, uses the shared async engine/session.
- InMemoryDraftingRepository — deterministic, dependency-free for unit tests.

Both return Pydantic records so the service layer is persistence-agnostic.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select

from app.core.database import get_async_session_factory
from app.models.drafting import (
    DraftGenerationRun,
    EvidenceRecord,
    EvidenceRetrievalRun,
    ReviewAuditEvent,
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

    # --- Human Review (M4.1) ---

    @abstractmethod
    async def submit_review(
        self,
        workspace_id: uuid.UUID,
        run_id: uuid.UUID,
        review_status: str,  # "approved" | "changes_requested"
        reviewer_edited_answer: Optional[str],
        reviewer_compliance_status: Optional[str],
        reviewer_compliance_rationale: Optional[str],
        reviewer_comment: str,
        citation_reviews: List[dict],
    ) -> "DraftGenerationRunRecord":
        """Submit a human review for a draft run.

        Returns the updated draft run record with review fields populated.
        Raises ValueError if the run doesn't exist or doesn't belong to the workspace.
        """
        ...

    @abstractmethod
    async def get_draft_run_for_review(
        self, workspace_id: uuid.UUID, run_id: uuid.UUID
    ) -> Optional["DraftGenerationRunRecord"]:
        """Get a draft run for review, enforcing workspace isolation."""
        ...

    # --- Human Review Audit History (M4.2) ---

    @abstractmethod
    async def create_review_audit_event(
        self,
        workspace_id: uuid.UUID,
        requirement_id: uuid.UUID,
        draft_run_id: uuid.UUID,
        review_status: str,
        reviewer_edited_answer: Optional[str],
        reviewer_compliance_status: Optional[str],
        reviewer_compliance_rationale: Optional[str],
        reviewer_comment: str,
        citation_reviews: List[dict],
        reviewer_workspace_id: uuid.UUID,
        previous_review_status: Optional[str] = None,
    ) -> "ReviewAuditEventRecord":
        """Create an append-only audit event for a human review submission.

        Records are immutable and never modified or deleted.
        """
        ...

    @abstractmethod
    async def get_review_history(
        self, workspace_id: uuid.UUID, requirement_id: uuid.UUID
    ) -> List["ReviewAuditEventRecord"]:
        """Get the complete review audit history for a requirement, ordered chronologically.

        Returns events oldest-first (append-only order).
        """
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
    compliance_status: Optional[str] = None
    compliance_rationale: Optional[str] = None
    # Human review fields (M4.1)
    review_status: Optional[str] = None
    reviewer_edited_answer: Optional[str] = None
    reviewer_compliance_status: Optional[str] = None
    reviewer_compliance_rationale: Optional[str] = None
    reviewer_comment: Optional[str] = None
    citation_reviews: List[dict] = Field(default_factory=list)
    reviewed_at: Optional[datetime] = None
    reviewer_workspace_id: Optional[uuid.UUID] = None
    status: str
    errors: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ReviewAuditEventRecord(BaseModel):
    """Append-only audit event for human review actions (M4.2)."""

    id: uuid.UUID
    workspace_id: uuid.UUID
    requirement_id: uuid.UUID
    draft_run_id: uuid.UUID
    previous_review_status: Optional[str] = None
    review_status: str
    decision: str
    reviewer_edited_answer: Optional[str] = None
    reviewer_compliance_status: Optional[str] = None
    reviewer_compliance_rationale: Optional[str] = None
    reviewer_comment: str
    citation_reviews: List[dict] = Field(default_factory=list)
    reviewed_at: datetime
    reviewer_workspace_id: uuid.UUID

    class Config:
        from_attributes = True


# --- Conversion Helpers ---

def _citations_with_verification(draft: RequirementDraft) -> List[dict]:
    """Serialize citations enriched with their verification outcome (M3.3).

    Each stored citation dict carries ``evidence_id``, ``excerpt``, ``score``
    plus the independent verification ``verified``/``reason``/``source_doc_id``
    so failed citations are persisted and reported rather than discarded.
    """
    verification_by_id = {
        v.evidence_id: v for v in (draft.citation_verifications or [])
    }
    serialized: List[dict] = []
    for c in draft.citations:
        verification = verification_by_id.get(c.evidence_id)
        serialized.append(
            {
                "evidence_id": c.evidence_id,
                "excerpt": c.excerpt,
                "score": c.score,
                "verified": verification.verified if verification else None,
                "reason": verification.reason if verification else None,
                "source_doc_id": verification.source_doc_id if verification else None,
            }
        )
    return serialized

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
        compliance_status=run.compliance_status,
        compliance_rationale=run.compliance_rationale,
        # Human review fields (M4.1)
        review_status=run.review_status,
        reviewer_edited_answer=run.reviewer_edited_answer,
        reviewer_compliance_status=run.reviewer_compliance_status,
        reviewer_compliance_rationale=run.reviewer_compliance_rationale,
        reviewer_comment=run.reviewer_comment,
        citation_reviews=list(run.citation_reviews or []),
        reviewed_at=run.reviewed_at,
        reviewer_workspace_id=run.reviewer_workspace_id,
        status=run.status,
        errors=list(run.errors or []),
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def _audit_event_to_record(event: ReviewAuditEvent) -> ReviewAuditEventRecord:
    return ReviewAuditEventRecord(
        id=event.id,
        workspace_id=event.workspace_id,
        requirement_id=event.requirement_id,
        draft_run_id=event.draft_run_id,
        previous_review_status=event.previous_review_status,
        review_status=event.review_status,
        decision=event.decision,
        reviewer_edited_answer=event.reviewer_edited_answer,
        reviewer_compliance_status=event.reviewer_compliance_status,
        reviewer_compliance_rationale=event.reviewer_compliance_rationale,
        reviewer_comment=event.reviewer_comment,
        citation_reviews=list(event.citation_reviews or []),
        reviewed_at=event.reviewed_at,
        reviewer_workspace_id=event.reviewer_workspace_id,
    )


# --- PostgreSQL Implementation ---

class PostgresDraftingRepository(DraftingRepository):
    """PostgreSQL-backed repository using the shared async engine."""

    def __init__(self, session_factory=None) -> None:
        self._session_factory = session_factory or get_async_session_factory()

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
                compliance_status=None,
                compliance_rationale=None,
                review_status="pending_review",
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
                run.citations = _citations_with_verification(draft)
                run.evidence_found = draft.evidence_found
                run.uncertainty = draft.uncertainty.value
                run.uncertainty_notes = list(draft.uncertainty_notes)
                run.compliance_status = (
                    draft.compliance_status.value
                    if draft.compliance_status is not None
                    else None
                )
                run.compliance_rationale = draft.compliance_rationale
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

    # --- Human Review (M4.1) ---

    async def submit_review(
        self,
        workspace_id: uuid.UUID,
        run_id: uuid.UUID,
        review_status: str,
        reviewer_edited_answer: Optional[str],
        reviewer_compliance_status: Optional[str],
        reviewer_compliance_rationale: Optional[str],
        reviewer_comment: str,
        citation_reviews: List[dict],
    ) -> DraftGenerationRunRecord:
        """Submit a human review for a draft run.

        Updates the draft run and creates an audit event atomically in a single transaction.
        """
        # Map review_status to explicit decision
        decision = "approve" if review_status == "approved" else "request_changes"

        async with self._session_factory() as session:
            run = await session.get(DraftGenerationRun, run_id)
            if run is None:
                raise ValueError(f"Draft run {run_id} not found")
            if run.workspace_id != workspace_id:
                raise ValueError(f"Draft run {run_id} does not belong to workspace {workspace_id}")

            # Validate review lifecycle transition
            current_status = run.review_status
            allowed_transitions = {
                None: {"approved", "changes_requested"},
                "pending_review": {"approved", "changes_requested"},
                "approved": set(),  # Terminal state
                "changes_requested": {"approved", "changes_requested"},  # Can re-review
            }
            if current_status not in allowed_transitions:
                # Treat unknown as pending_review
                current_status = "pending_review"
            if review_status not in allowed_transitions.get(current_status, set()):
                raise ValueError(
                    f"Invalid review transition: {current_status} -> {review_status}"
                )

            # Update draft run
            run.review_status = review_status
            if reviewer_edited_answer is not None:
                run.reviewer_edited_answer = reviewer_edited_answer
            if reviewer_compliance_status is not None:
                run.reviewer_compliance_status = reviewer_compliance_status
            if reviewer_compliance_rationale is not None:
                run.reviewer_compliance_rationale = reviewer_compliance_rationale
            run.reviewer_comment = reviewer_comment
            run.citation_reviews = citation_reviews
            run.reviewed_at = datetime.now(timezone.utc)
            run.reviewer_workspace_id = workspace_id

            # Create append-only audit event (M4.2) in same transaction
            audit_event = ReviewAuditEvent(
                workspace_id=workspace_id,
                requirement_id=run.requirement_id,
                draft_run_id=run.id,
                previous_review_status=current_status,
                review_status=review_status,
                decision=decision,
                reviewer_edited_answer=reviewer_edited_answer,
                reviewer_compliance_status=reviewer_compliance_status,
                reviewer_compliance_rationale=reviewer_compliance_rationale,
                reviewer_comment=reviewer_comment,
                citation_reviews=citation_reviews,
                reviewer_workspace_id=workspace_id,
            )
            session.add(audit_event)

            # Single commit for both changes
            await session.commit()
            await session.refresh(run)

            return _draft_run_to_record(run)

    async def get_draft_run_for_review(
        self, workspace_id: uuid.UUID, run_id: uuid.UUID
    ) -> Optional[DraftGenerationRunRecord]:
        """Get a draft run for review, enforcing workspace isolation."""
        async with self._session_factory() as session:
            run = await session.get(DraftGenerationRun, run_id)
            if run is None or run.workspace_id != workspace_id:
                return None
            return _draft_run_to_record(run)

    async def create_review_audit_event(
        self,
        workspace_id: uuid.UUID,
        requirement_id: uuid.UUID,
        draft_run_id: uuid.UUID,
        review_status: str,
        reviewer_edited_answer: Optional[str],
        reviewer_compliance_status: Optional[str],
        reviewer_compliance_rationale: Optional[str],
        reviewer_comment: str,
        citation_reviews: List[dict],
        reviewer_workspace_id: uuid.UUID,
        previous_review_status: Optional[str] = None,
    ) -> ReviewAuditEventRecord:
        """Create an append-only audit event for a human review submission.

        Records are immutable and never modified or deleted.
        """
        decision = "approve" if review_status == "approved" else "request_changes"
        async with self._session_factory() as session:
            event = ReviewAuditEvent(
                workspace_id=workspace_id,
                requirement_id=requirement_id,
                draft_run_id=draft_run_id,
                previous_review_status=previous_review_status,
                review_status=review_status,
                decision=decision,
                reviewer_edited_answer=reviewer_edited_answer,
                reviewer_compliance_status=reviewer_compliance_status,
                reviewer_compliance_rationale=reviewer_compliance_rationale,
                reviewer_comment=reviewer_comment,
                citation_reviews=citation_reviews,
                reviewer_workspace_id=reviewer_workspace_id,
            )
            session.add(event)
            await session.commit()
            await session.refresh(event)
            return _audit_event_to_record(event)

    async def get_review_history(
        self, workspace_id: uuid.UUID, requirement_id: uuid.UUID
    ) -> List[ReviewAuditEventRecord]:
        """Get the complete review audit history for a requirement, ordered chronologically.

        Returns events oldest-first (append-only order). Uses reviewed_at + id for stable ordering.
        """
        async with self._session_factory() as session:
            stmt = (
                select(ReviewAuditEvent)
                .where(
                    ReviewAuditEvent.workspace_id == workspace_id,
                    ReviewAuditEvent.requirement_id == requirement_id,
                )
                .order_by(ReviewAuditEvent.reviewed_at.asc(), ReviewAuditEvent.id.asc())
            )
            result = await session.execute(stmt)
            return [_audit_event_to_record(e) for e in result.scalars().all()]


# --- In-Memory Implementation for Tests ---

class InMemoryDraftingRepository(DraftingRepository):
    """Deterministic in-memory repository for tests."""

    def __init__(self) -> None:
        self._evidence_runs: dict[uuid.UUID, EvidenceRetrievalRunRecord] = {}
        self._evidence_records: dict[uuid.UUID, List[EvidenceRecordRecord]] = {}
        self._draft_runs: dict[uuid.UUID, DraftGenerationRunRecord] = {}
        self._review_audit_events: List[ReviewAuditEventRecord] = []

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
            compliance_status=None,
            compliance_rationale=None,
            # Human review fields (M4.1)
            review_status="pending_review",
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
            record.citations = _citations_with_verification(draft)
            record.evidence_found = draft.evidence_found
            record.uncertainty = draft.uncertainty.value
            record.uncertainty_notes = list(draft.uncertainty_notes)
            record.compliance_status = (
                draft.compliance_status.value
                if draft.compliance_status is not None
                else None
            )
            record.compliance_rationale = draft.compliance_rationale
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

    # --- Human Review (M4.1) ---

    async def submit_review(
        self,
        workspace_id: uuid.UUID,
        run_id: uuid.UUID,
        review_status: str,
        reviewer_edited_answer: Optional[str],
        reviewer_compliance_status: Optional[str],
        reviewer_compliance_rationale: Optional[str],
        reviewer_comment: str,
        citation_reviews: List[dict],
    ) -> DraftGenerationRunRecord:
        """Submit a human review for a draft run (in-memory)."""
        record = self._draft_runs.get(run_id)
        if record is None:
            raise ValueError(f"Draft run {run_id} not found")
        if record.workspace_id != workspace_id:
            raise ValueError(f"Draft run {run_id} does not belong to workspace {workspace_id}")

        # Validate review lifecycle transition
        current_status = record.review_status
        allowed_transitions = {
            None: {"approved", "changes_requested"},
            "pending_review": {"approved", "changes_requested"},
            "approved": set(),
            "changes_requested": {"approved", "changes_requested"},
        }
        if current_status not in allowed_transitions:
            current_status = "pending_review"
        if review_status not in allowed_transitions.get(current_status, set()):
            raise ValueError(
                f"Invalid review transition: {current_status} -> {review_status}"
            )

        record.review_status = review_status
        if reviewer_edited_answer is not None:
            record.reviewer_edited_answer = reviewer_edited_answer
        if reviewer_compliance_status is not None:
            record.reviewer_compliance_status = reviewer_compliance_status
        if reviewer_compliance_rationale is not None:
            record.reviewer_compliance_rationale = reviewer_compliance_rationale
        record.reviewer_comment = reviewer_comment
        record.citation_reviews = citation_reviews
        record.reviewed_at = datetime.now(timezone.utc)
        record.reviewer_workspace_id = workspace_id
        record.updated_at = datetime.now(timezone.utc)

        # Map review_status to explicit decision
        decision = "approve" if review_status == "approved" else "request_changes"

        # Create append-only audit event (M4.2)
        audit_event = ReviewAuditEventRecord(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            requirement_id=record.requirement_id,
            draft_run_id=record.run_id,
            previous_review_status=current_status,
            review_status=review_status,
            decision=decision,
            reviewer_edited_answer=reviewer_edited_answer,
            reviewer_compliance_status=reviewer_compliance_status,
            reviewer_compliance_rationale=reviewer_compliance_rationale,
            reviewer_comment=reviewer_comment,
            citation_reviews=citation_reviews,
            reviewed_at=datetime.now(timezone.utc),
            reviewer_workspace_id=workspace_id,
        )
        self._review_audit_events.append(audit_event)

        return record

    async def create_review_audit_event(
        self,
        workspace_id: uuid.UUID,
        requirement_id: uuid.UUID,
        draft_run_id: uuid.UUID,
        review_status: str,
        reviewer_edited_answer: Optional[str],
        reviewer_compliance_status: Optional[str],
        reviewer_compliance_rationale: Optional[str],
        reviewer_comment: str,
        citation_reviews: List[dict],
        reviewer_workspace_id: uuid.UUID,
        previous_review_status: Optional[str] = None,
    ) -> ReviewAuditEventRecord:
        """Create an append-only audit event for a human review submission (in-memory).

        Records are immutable and never modified or deleted.
        """
        decision = "approve" if review_status == "approved" else "request_changes"
        event = ReviewAuditEventRecord(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            requirement_id=requirement_id,
            draft_run_id=draft_run_id,
            previous_review_status=previous_review_status,
            review_status=review_status,
            decision=decision,
            reviewer_edited_answer=reviewer_edited_answer,
            reviewer_compliance_status=reviewer_compliance_status,
            reviewer_compliance_rationale=reviewer_compliance_rationale,
            reviewer_comment=reviewer_comment,
            citation_reviews=citation_reviews,
            reviewed_at=datetime.now(timezone.utc),
            reviewer_workspace_id=reviewer_workspace_id,
        )
        self._review_audit_events.append(event)
        return event

    async def get_review_history(
        self, workspace_id: uuid.UUID, requirement_id: uuid.UUID
    ) -> List[ReviewAuditEventRecord]:
        """Get the complete review audit history for a requirement, ordered chronologically (in-memory).

        Returns events oldest-first (append-only order). Uses reviewed_at + id for stable ordering.
        """
        events = [
            e for e in self._review_audit_events
            if e.workspace_id == workspace_id and e.requirement_id == requirement_id
        ]
        # Sort by reviewed_at ascending, then id for stable ordering
        events.sort(key=lambda e: (e.reviewed_at, e.id))
        return events

    async def get_draft_run_for_review(
        self, workspace_id: uuid.UUID, run_id: uuid.UUID
    ) -> Optional[DraftGenerationRunRecord]:
        """Get a draft run for review, enforcing workspace isolation (in-memory)."""
        record = self._draft_runs.get(run_id)
        if record is None or record.workspace_id != workspace_id:
            return None
        return record