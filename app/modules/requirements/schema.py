"""Pydantic schemas for the requirements module (API + persistence records)."""

from typing import List, Optional

from pydantic import BaseModel, Field

from app.modules.requirements.classification import (
    RequirementCategory,
    RequirementModality,
)


class ExtractRequest(BaseModel):
    """Request body for POST /api/requirements/extract."""

    doc_id: str = Field(
        ..., min_length=1, description="Ingested document ID to extract from"
    )


class RequirementRecord(BaseModel):
    """A persisted requirement with its source reference."""

    requirement_key: str = Field(..., description="Stable requirement identifier")
    doc_id: str = Field(..., description="Source document ID")
    text: str = Field(..., description="Original requirement text")
    section_title: Optional[str] = Field(None, description="Section title")
    section_path: Optional[str] = Field(None, description="Dotted section path")
    page_number: int = Field(..., description="1-based page number")
    character_start: int = Field(..., description="Absolute start offset")
    character_end: int = Field(..., description="Absolute end offset")
    modality: RequirementModality = Field(..., description="Bindingness")
    category: RequirementCategory = Field(..., description="Category")
    cross_references: List[str] = Field(
        default_factory=list, description="Referenced sections/ids"
    )
    confidence: float = Field(..., description="Confidence in [0, 1]")
    uncertainty_notes: List[str] = Field(
        default_factory=list, description="Uncertainty notes"
    )
    extraction_method: str = Field(..., description="rule | llm | hybrid")


class SourceReference(BaseModel):
    """Where a requirement lives in the source document."""

    doc_id: str = Field(..., description="Source document ID")
    source_filename: Optional[str] = Field(
        None, description="Original filename of the source document"
    )
    page_number: int = Field(..., description="1-based page number")
    character_start: int = Field(..., description="Absolute start offset")
    character_end: int = Field(..., description="Absolute end offset")


class RequirementDetailResponse(BaseModel):
    """A requirement together with its source reference."""

    requirement: RequirementRecord = Field(...)
    source: SourceReference = Field(...)


class RequirementListResponse(BaseModel):
    """Response body for GET /api/requirements/{doc_id}."""

    doc_id: str = Field(..., description="Document ID")
    requirements: List[RequirementRecord] = Field(
        default_factory=list, description="Extracted requirements"
    )


class ExtractionStatusResponse(BaseModel):
    """Response body for GET /api/requirements/{doc_id}/status."""

    doc_id: str = Field(..., description="Document ID")
    run_id: Optional[str] = Field(
        None, description="Most recent extraction run identifier"
    )
    status: str = Field(..., description="pending | running | completed | failed")
    requirement_count: int = Field(
        ..., description="Number of requirements in the run"
    )
    errors: List[str] = Field(
        default_factory=list, description="Errors / warnings from the run"
    )


class ExtractResponse(BaseModel):
    """Response body for POST /api/requirements/extract."""

    doc_id: str = Field(..., description="Document ID")
    run_id: str = Field(..., description="Extraction run identifier")
    status: str = Field(..., description="Run status")
    requirement_count: int = Field(
        ..., description="Number of requirements extracted"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Parser / extraction warnings (e.g. empty pages)",
    )


class ExtractionRunRecord(BaseModel):
    """Internal record of an extraction run."""

    run_id: str
    doc_id: str
    status: str
    requirement_count: int
    errors: List[str] = Field(default_factory=list)