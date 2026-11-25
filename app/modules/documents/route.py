"""HTTP routes for document ingestion and queries."""

from typing import Optional

from fastapi import APIRouter, File, UploadFile, Query

from app.modules.documents.controller import (
    get_document_controller,
    get_document_chunks_controller,
    ingest_document_controller,
    list_documents_controller,
)
from app.modules.documents.schema import DocumentIngestResponse

router = APIRouter(tags=["documents"])


@router.post("/ingest", response_model=DocumentIngestResponse)
async def ingest_document_route(
    file: UploadFile = File(...),
    workspace_id: Optional[str] = Query(
        None, description="Workspace to ingest the document into."
    ),
) -> DocumentIngestResponse:
    """Upload and ingest a document (.pdf, .txt, or .md) into the vector store."""
    return await ingest_document_controller(file, workspace_id=workspace_id)


@router.get("", response_model=dict)
async def list_documents_route(
    workspace_id: Optional[str] = Query(
        None, description="Workspace to scope the listing to."
    ),
):
    """List ingested documents, optionally scoped to a workspace."""
    return await list_documents_controller(workspace_id=workspace_id)


@router.get("/{doc_id}", response_model=dict)
async def get_document_route(
    doc_id: str,
    workspace_id: Optional[str] = Query(
        None, description="Workspace to scope the lookup to."
    ),
):
    """Get metadata for a single ingested document."""
    return await get_document_controller(doc_id, workspace_id=workspace_id)


@router.get("/{doc_id}/chunks", response_model=dict)
async def get_document_chunks_route(
    doc_id: str,
    workspace_id: Optional[str] = Query(
        None, description="Workspace to scope the lookup to."
    ),
):
    """Get all chunks for a document with their source references."""
    return await get_document_chunks_controller(
        doc_id, workspace_id=workspace_id
    )
