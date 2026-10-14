"""HTTP route for document ingestion."""

from fastapi import APIRouter, File, UploadFile

from app.modules.documents.controller import ingest_document_controller
from app.modules.documents.schema import DocumentIngestResponse

router = APIRouter(tags=["documents"])


@router.post("/ingest", response_model=DocumentIngestResponse)
async def ingest_document_route(
    file: UploadFile = File(...),
) -> DocumentIngestResponse:
    """Upload and ingest a text document (.txt or .md) into the vector store."""
    return await ingest_document_controller(file)
