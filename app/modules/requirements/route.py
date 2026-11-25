"""HTTP routes for structured requirement extraction."""

from fastapi import APIRouter

from app.modules.requirements.controller import (
    extract_requirements_controller,
    get_requirement_controller,
    get_status_controller,
    list_requirements_controller,
)
from app.modules.requirements.schema import (
    ExtractRequest,
    ExtractResponse,
    ExtractionStatusResponse,
    RequirementDetailResponse,
    RequirementListResponse,
)

router = APIRouter(tags=["requirements"])


@router.post("/extract", response_model=ExtractResponse)
async def extract_requirements_route(
    request: ExtractRequest,
) -> ExtractResponse:
    """Start requirement extraction for an ingested RFP.

    The workspace is resolved server-side; clients cannot select a workspace.
    """
    return await extract_requirements_controller(request.doc_id)


@router.get("/{doc_id}", response_model=RequirementListResponse)
async def list_requirements_route(doc_id: str) -> RequirementListResponse:
    """List extracted requirements for a document in the resolved workspace."""
    return await list_requirements_controller(doc_id)


@router.get("/{doc_id}/status", response_model=ExtractionStatusResponse)
async def get_status_route(doc_id: str) -> ExtractionStatusResponse:
    """Report extraction status and errors for a document."""
    return await get_status_controller(doc_id)


@router.get("/{doc_id}/requirements/{requirement_key}", response_model=RequirementDetailResponse)
async def get_requirement_route(
    doc_id: str, requirement_key: str
) -> RequirementDetailResponse:
    """Retrieve a single requirement with its source reference."""
    return await get_requirement_controller(doc_id, requirement_key)