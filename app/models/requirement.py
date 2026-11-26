"""Extracted requirement SQLAlchemy model.

Each row is one individually identifiable, traceable requirement extracted
from an ingested RFP document.  Source references (page number + absolute
character offsets) are preserved so every requirement can be traced back to
its exact location in the normalized source text.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.tenant import Base

if TYPE_CHECKING:
    from app.models.extraction_run import ExtractionRun
    from app.models.drafting import DraftGenerationRun, EvidenceRetrievalRun


class Requirement(Base):
    """A single extracted requirement with its source reference."""

    __tablename__ = "requirements"

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
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("extraction_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Stable, deterministic identifier derived from doc + source span.
    requirement_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    doc_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    text: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    section_title: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    section_path: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )
    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    character_start: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    character_end: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    # mandatory | optional | informational | uncertain
    modality: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    # functional | technical | security | ... | uncertain
    category: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    cross_references: Mapped[list] = mapped_column(
        "cross_references",
        JSONB,
        nullable=False,
        default=list,
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    uncertainty_notes: Mapped[list] = mapped_column(
        "uncertainty_notes",
        JSONB,
        nullable=False,
        default=list,
    )
    # rule | llm | hybrid
    extraction_method: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    run: Mapped["ExtractionRun"] = relationship(
        "ExtractionRun",
        back_populates="requirements",
    )
    evidence_runs: Mapped[list["EvidenceRetrievalRun"]] = relationship(
        "EvidenceRetrievalRun",
        back_populates="requirement",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    draft_runs: Mapped[list["DraftGenerationRun"]] = relationship(
        "DraftGenerationRun",
        back_populates="requirement",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # Stable per (run, requirement_key): re-extraction replaces cleanly.
    __table_args__ = (
        UniqueConstraint(
            "run_id", "requirement_key", name="uq_requirements_run_key"
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Requirement(id={self.id}, key={self.requirement_key}, "
            f"modality={self.modality})>"
        )