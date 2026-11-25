"""HTTP routes for document ingestion and queries."""

from typing import Optional

from fastapi import APIRouter, File, UploadFile

from app.core.workspace import resolve_workspace_id
from app.modules.documents.controller import (
    get_document_chunks_controller,
    get_document_controller,
    ingest_document_controller,
    list_documents_controller,
)
from app.modules.documents.schema import (
    ChunkListResponse,
    DocumentIngestResponse,
    DocumentListResponse,
)

router = APIRouter(tags=["documents"])


@router.post("/ingest", response_model=DocumentIngestResponse)
async def ingest_document_route(
    file: UploadFile = File(...),
) -> DocumentIngestResponse:
    """Upload and ingest a document (.pdf, .txt, or .md) into the vector store.

    The workspace is resolved server-side; clients cannot select a workspace.
    """
    workspace_id = resolve_workspace_id()
    return await ingest_document_controller(file, workspace_id=workspace_id)


@router.get("", response_model=DocumentListResponse)
async def list_documents_route() -> DocumentListResponse:
    """List ingested documents in the server-resolved workspace."""
    workspace_id = resolve_workspace_id()
    return await list_documents_controller(workspace_id=workspace_id)


@router.get("/{doc_id}", response_model=dict)
async def get_document_route(doc_id: str):
    """Get metadata for a single ingested document in the resolved workspace."""
    workspace_id = resolve_workspace_id()
    return await get_document_controller(doc_id, workspace_id=workspace_id)


@router.get("/{doc_id}/chunks", response_model=ChunkListResponse)
async def get_document_chunks_route(doc_id: str) -> ChunkListResponse:
    """Get all chunks for a document with their source references."""
    workspace_id = resolve_workspace_id()
    return await get_document_chunks_controller(
        doc_id, workspace_id=workspace_id
    )
