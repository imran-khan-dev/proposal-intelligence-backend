"""Pydantic schemas for the documents module."""

from typing import List, Optional

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
    workspace_id: Optional[str] = Field(
        None, description="Workspace the document was ingested into"
    )


class DocumentChunkResponse(BaseModel):
    """A stored chunk together with its source reference."""

    chunk_id: str = Field(..., description="Stored chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    doc_id: str = Field(..., description="Document ID used at ingestion")
    chunk_index: int = Field(..., description="Index of the chunk within the document")
    text: str = Field(..., description="Chunk text")
    page_number: Optional[int] = Field(
        None, description="1-based page number the chunk originates from"
    )
    character_start: Optional[int] = Field(
        None, description="Absolute character offset (inclusive) in the source document"
    )
    character_end: Optional[int] = Field(
        None, description="Absolute character offset (exclusive) in the source document"
    )
    section_path: Optional[str] = Field(
        None, description="Dotted section path, when available"
    )
    source_filename: Optional[str] = Field(
        None, description="Original filename of the source document"
    )
    workspace_id: Optional[str] = Field(
        None, description="Workspace the document belongs to"
    )


class DocumentResponse(BaseModel):
    """Metadata for an ingested document."""

    doc_id: str = Field(..., description="Document ID used at ingestion")
    source_filename: Optional[str] = Field(
        None, description="Original filename of the source document"
    )
    title: Optional[str] = Field(None, description="Document title")
    chunk_count: int = Field(..., description="Number of chunks stored")
    workspace_id: Optional[str] = Field(
        None, description="Workspace the document belongs to"
    )
    pages: List[int] = Field(
        default_factory=list,
        description="1-based page numbers present in the document",
    )


class DocumentListResponse(BaseModel):
    """Response body for GET /api/documents."""

    documents: List[DocumentResponse] = Field(
        default_factory=list, description="Ingested documents in scope"
    )


class ChunkListResponse(BaseModel):
    """Response body for GET /api/documents/{doc_id}/chunks."""

    doc_id: str = Field(..., description="Document ID")
    chunks: List[DocumentChunkResponse] = Field(
        default_factory=list, description="Stored chunks with source references"
    )
