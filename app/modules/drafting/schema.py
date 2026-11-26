"""Pydantic schemas for the drafting module (API + persistence records)."""

from typing import List, Optional
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, Field


class EvidencePassageResponse(BaseModel):
    """A retrieved evidence passage with score and source references."""

    text: str = Field(..., description="The evidence passage text")
    score: float = Field(..., description="Similarity score (0.0 to 1.0)")
    rank: int = Field(..., description="Rank in retrieval results (1-based)")
    source_doc_id: str = Field(..., description="Source document ID in company knowledge base")
    source_filename: Optional[str] = Field(None, description="Original filename")
    chunk_id: Optional[str] = Field(None, description="Chunk ID in vector store")
    chunk_index: Optional[int] = Field(None, description="Chunk index within document")
    page_number: Optional[int] = Field(None, description="Page number (1-based)")
    character_start: Optional[int] = Field(None, description="Absolute character start offset")
    character_end: Optional[int] = Field(None, description="Absolute character end offset")
    section_path: Optional[str] = Field(None, description="Section path (e.g., '3.2.1')")


class EvidenceCitationResponse(BaseModel):
    """A citation linking draft text to an evidence passage."""

    evidence_id: str = Field(..., description="Evidence passage ID (chunk_id)")
    excerpt: str = Field(..., description="Exact excerpt from evidence used")
    score: float = Field(..., description="Similarity score of the evidence")


class RequirementDraftResponse(BaseModel):
    """A draft response for a single requirement with evidence citations."""

    requirement_id: UUID = Field(..., description="Database UUID of the requirement")
    requirement_key: str = Field(..., description="Stable requirement identifier")
    requirement_text: str = Field(..., description="Original requirement text")
    draft_answer: Optional[str] = Field(None, description="Generated draft answer (null if no evidence)")
    citations: List[EvidenceCitationResponse] = Field(default_factory=list, description="Evidence citations")
    evidence_found: bool = Field(..., description="Whether supporting evidence was found")
    uncertainty: str = Field(..., description="Uncertainty level: supported|weak|none|conflicting")
    uncertainty_notes: List[str] = Field(default_factory=list, description="Notes about gaps or conflicts")


class DraftGenerationRunRecord(BaseModel):
    """Record of a draft generation run."""

    run_id: UUID = Field(..., description="Draft generation run ID")
    workspace_id: UUID = Field(..., description="Workspace UUID")
    requirement_id: UUID = Field(..., description="Requirement UUID")
    retrieval_run_id: Optional[UUID] = Field(None, description="Linked evidence retrieval run ID")
    draft_answer: Optional[str] = Field(None, description="Generated draft answer")
    citations: List[dict] = Field(default_factory=list, description="Evidence citations")
    evidence_found: bool = Field(..., description="Whether evidence was found")
    uncertainty: str = Field(..., description="Uncertainty level")
    uncertainty_notes: List[str] = Field(default_factory=list, description="Uncertainty notes")
    status: str = Field(..., description="pending|running|completed|failed")
    errors: List[str] = Field(default_factory=list, description="Errors from the run")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class EvidenceRetrievalRunRecord(BaseModel):
    """Record of an evidence retrieval run."""

    run_id: UUID = Field(..., description="Evidence retrieval run ID")
    workspace_id: UUID = Field(..., description="Workspace UUID")
    requirement_id: UUID = Field(..., description="Requirement UUID")
    query_text: str = Field(..., description="The search query used")
    status: str = Field(..., description="pending|running|completed|failed")
    evidence_count: int = Field(..., description="Number of evidence passages retrieved")
    errors: List[str] = Field(default_factory=list, description="Errors from the run")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class EvidenceRecordResponse(BaseModel):
    """A persisted evidence record."""

    id: UUID = Field(..., description="Evidence record ID")
    run_id: UUID = Field(..., description="Retrieval run ID")
    workspace_id: UUID = Field(..., description="Workspace UUID")
    source_doc_id: str = Field(..., description="Source document ID")
    source_filename: Optional[str] = Field(None, description="Original filename")
    chunk_id: Optional[str] = Field(None, description="Chunk ID")
    chunk_index: Optional[int] = Field(None, description="Chunk index")
    page_number: Optional[int] = Field(None, description="Page number")
    character_start: Optional[int] = Field(None, description="Character start offset")
    character_end: Optional[int] = Field(None, description="Character end offset")
    section_path: Optional[str] = Field(None, description="Section path")
    text: str = Field(..., description="Evidence passage text")
    score: float = Field(..., description="Similarity score")
    rank: int = Field(..., description="Rank in retrieval results")
    created_at: datetime = Field(..., description="Creation timestamp")


# Request/Response models for API endpoints

class StartDraftingRequest(BaseModel):
    """Request to start drafting for an extracted RFP's requirements."""

    doc_id: str = Field(..., min_length=1, description="Document ID that was extracted")
    top_k: int = Field(5, ge=1, le=20, description="Max evidence passages per requirement")


class StartDraftingResponse(BaseModel):
    """Response after starting drafting for all requirements of a document."""

    doc_id: str = Field(..., description="Document ID")
    requirements_processed: int = Field(..., description="Number of requirements drafted")
    evidence_runs_created: int = Field(..., description="Number of evidence retrieval runs created")
    draft_runs_created: int = Field(..., description="Number of draft generation runs created")
    status: str = Field(..., description="Overall status: completed|partial_failure")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


class DraftListResponse(BaseModel):
    """List of drafts for a document's requirements."""

    doc_id: str = Field(..., description="Document ID")
    drafts: List[RequirementDraftResponse] = Field(default_factory=list, description="Draft responses")


class SingleDraftResponse(BaseModel):
    """Single requirement draft with full details."""

    requirement: RequirementDraftResponse = Field(...)
    retrieval_query: str = Field(..., description="The query used for evidence retrieval")
    evidence_passages: List[EvidencePassageResponse] = Field(default_factory=list, description="Retrieved evidence")
    retrieval_run: Optional[EvidenceRetrievalRunRecord] = Field(None, description="Retrieval run record")
    draft_run: Optional[DraftGenerationRunRecord] = Field(None, description="Draft generation run record")


class DraftingStatusResponse(BaseModel):
    """Processing status for a document's drafting."""

    doc_id: str = Field(..., description="Document ID")
    requirements_total: int = Field(..., description="Total requirements for this document")
    evidence_runs_completed: int = Field(..., description="Completed evidence retrieval runs")
    draft_runs_completed: int = Field(..., description="Completed draft generation runs")
    evidence_runs_failed: int = Field(..., description="Failed evidence retrieval runs")
    draft_runs_failed: int = Field(..., description="Failed draft generation runs")
    overall_status: str = Field(..., description="pending|running|completed|failed|partial")