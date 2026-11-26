"""Drafting and evidence retrieval SQLAlchemy models.

Models for requirement-level evidence retrieval and draft response generation.
Each requirement can have a retrieval run that fetches evidence from the company
knowledge base, and a draft generation run that produces a response with citations.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.tenant import Base

if TYPE_CHECKING:
    from app.models.requirement import Requirement


class EvidenceRetrievalRun(Base):
    """A single evidence retrieval pass for a requirement.

    Records status, query used, and results so callers can report progress
    and failures without a background-job system.
    """

    __tablename__ = "evidence_retrieval_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    requirement_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("requirements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # The search query used for retrieval
    query_text: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    # pending | running | completed | failed
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
    )
    evidence_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    errors: Mapped[list] = mapped_column(
        "errors",
        JSONB,
        nullable=False,
        default=list,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    requirement: Mapped["Requirement"] = relationship(
        "Requirement",
        back_populates="evidence_runs",
    )
    evidence: Mapped[list["EvidenceRecord"]] = relationship(
        "EvidenceRecord",
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    draft_runs: Mapped[list["DraftGenerationRun"]] = relationship(
        "DraftGenerationRun",
        back_populates="retrieval_run",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<EvidenceRetrievalRun(id={self.id}, requirement_id={self.requirement_id}, "
            f"status={self.status})>"
        )


class EvidenceRecord(Base):
    """A single piece of evidence retrieved for a requirement.

    Preserves all source references (doc_id, filename, page, character offsets,
    chunk_id) and the similarity score from retrieval.
    """

    __tablename__ = "evidence_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evidence_retrieval_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    # Source document in company knowledge base
    source_doc_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    source_filename: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    chunk_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    chunk_index: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    page_number: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    character_start: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    character_end: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    section_path: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )
    # The retrieved text passage
    text: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    # Similarity score (0.0 to 1.0, higher is more similar)
    score: Mapped[float] = mapped_column(
        nullable=False,
    )
    # Rank in retrieval results (1-based)
    rank: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    run: Mapped["EvidenceRetrievalRun"] = relationship(
        "EvidenceRetrievalRun",
        back_populates="evidence",
    )

    def __repr__(self) -> str:
        return (
            f"<EvidenceRecord(id={self.id}, run_id={self.run_id}, "
            f"source_doc={self.source_doc_id}, score={self.score:.3f})>"
        )


class DraftGenerationRun(Base):
    """A single draft generation pass for a requirement.

    Records status, the draft answer, evidence citations, and uncertainty indicators.
    """

    __tablename__ = "draft_generation_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    requirement_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("requirements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    retrieval_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evidence_retrieval_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # The generated draft answer text
    draft_answer: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )
    # Evidence citations: list of {"evidence_id": str, "excerpt": str}
    citations: Mapped[list] = mapped_column(
        "citations",
        JSONB,
        nullable=False,
        default=list,
    )
    # Whether supporting evidence was found
    evidence_found: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
    )
    # Uncertainty indicator: "supported" | "weak" | "none" | "conflicting"
    uncertainty: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="none",
    )
    # Notes about uncertainty or gaps
    uncertainty_notes: Mapped[list] = mapped_column(
        "uncertainty_notes",
        JSONB,
        nullable=False,
        default=list,
    )
    # Requirement-level compliance assessment (M3.4). Nullable so older draft
    # records without an assessment remain readable (treated as needs_review).
    compliance_status: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
    )
    compliance_rationale: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    # Human review lifecycle (M4.1): pending_review | approved | changes_requested
    review_status: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
    )
    # Human reviewer's edited answer (separate from AI-generated draft_answer)
    reviewer_edited_answer: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    # Human reviewer's compliance status override (separate from AI assessment)
    reviewer_compliance_status: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
    )
    # Human reviewer's compliance rationale
    reviewer_compliance_rationale: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    # Reviewer comment/reason for the decision
    reviewer_comment: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    # Per-citation review decisions (distinct from automated M3.3 verification).
    # JSONB array of: {"evidence_id": str, "decision": "accept|reject|modify", "reviewer_excerpt": str, "reason": str}
    citation_reviews: Mapped[list] = mapped_column(
        "citation_reviews",
        JSONB,
        nullable=False,
        default=list,
    )
    # Timestamp when review was completed
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    # Identity/audit: server-resolved workspace of the reviewer (feasible with current architecture)
    reviewer_workspace_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    # pending | running | completed | failed
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
    )
    errors: Mapped[list] = mapped_column(
        "errors",
        JSONB,
        nullable=False,
        default=list,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    requirement: Mapped["Requirement"] = relationship(
        "Requirement",
        back_populates="draft_runs",
    )
    retrieval_run: Mapped[Optional["EvidenceRetrievalRun"]] = relationship(
        "EvidenceRetrievalRun",
        back_populates="draft_runs",
    )

    def __repr__(self) -> str:
        return (
            f"<DraftGenerationRun(id={self.id}, requirement_id={self.requirement_id}, "
            f"status={self.status}, uncertainty={self.uncertainty})>"
        )


# Add back-populates to Requirement model via relationship
# This is done in the Requirement model file to avoid circular imports
# The relationships are defined here for reference:
# Requirement.evidence_runs = relationship("EvidenceRetrievalRun", back_populates="requirement")
# Requirement.draft_runs = relationship("DraftGenerationRun", back_populates="requirement")
# EvidenceRetrievalRun.draft_runs = relationship("DraftGenerationRun", back_populates="retrieval_run")


class ReviewAuditEvent(Base):
    """Append-only audit event for human review actions (M4.2).

    Each successful human review submission creates an immutable audit record
    capturing the reviewer's decisions, comments, and timestamp. Records are
    never modified or deleted — they form a complete chronological history
    of all review actions for a requirement.
    """

    __tablename__ = "review_audit_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    requirement_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("requirements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    draft_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("draft_generation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Previous review status before this event (None for first review)
    previous_review_status: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
    )
    # Review decision: approved | changes_requested
    review_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    # Explicit decision: approve | request_changes
    decision: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    # Reviewer's edited answer (may be None if not changed)
    reviewer_edited_answer: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    # Reviewer's compliance status override
    reviewer_compliance_status: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
    )
    # Reviewer's compliance rationale
    reviewer_compliance_rationale: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    # Reviewer's comment/reason for the decision
    reviewer_comment: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    # Per-citation review decisions
    citation_reviews: Mapped[list] = mapped_column(
        "citation_reviews",
        JSONB,
        nullable=False,
        default=list,
    )
    # Timestamp when review was completed
    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    # Identity/audit: server-resolved workspace of the reviewer
    reviewer_workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<ReviewAuditEvent(id={self.id}, requirement_id={self.requirement_id}, "
            f"previous_review_status={self.previous_review_status}, review_status={self.review_status}, "
            f"decision={self.decision}, reviewed_at={self.reviewed_at})>"
        )