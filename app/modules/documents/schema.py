"""Pydantic schemas for the documents module."""

from pydantic import BaseModel, Field


class DocumentIngestResponse(BaseModel):
    """Response body for POST /api/documents/ingest."""

    doc_id: str = Field(
        ..., description="Document ID used for ingestion"
    )
    chunk_count: int = Field(
        ..., description="Number of chunks created and stored"
    )
    message: str = Field(
        ..., description="Human-readable status message"
    )
