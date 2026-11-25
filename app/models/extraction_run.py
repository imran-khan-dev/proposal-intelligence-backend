"""Extraction run SQLAlchemy model.

An extraction run records a single requirement-extraction pass over an
ingested RFP document.  It captures status and errors so callers can report
extraction progress and failures without a background-job system.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.tenant import Base

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.requirement import Requirement


class ExtractionRun(Base):
    """A single requirement-extraction pass over an ingested document."""

    __tablename__ = "extraction_runs"

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
    doc_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    # pending | running | completed | failed
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
    )
    requirement_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    # Structured error / warning details (JSON-serializable list).
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
    document: Mapped[Optional["Document"]] = relationship(
        "Document",
        foreign_keys=[document_id],
    )
    requirements: Mapped[list["Requirement"]] = relationship(
        "Requirement",
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<ExtractionRun(id={self.id}, doc_id={self.doc_id}, "
            f"status={self.status})>"
        )