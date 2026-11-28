"""HTTP routes for requirement-level evidence retrieval and drafting."""

from fastapi import APIRouter

from app.modules.drafting.controller import (
    get_draft_controller,
    get_drafting_status_controller,
    get_review_history_controller,
    list_drafts_controller,
    review_draft_controller,
    start_drafting_controller,
)
from app.modules.drafting.schema import (
    DraftListResponse,
    DraftingStatusResponse,
    ReviewDraftRequest,
    ReviewDraftResponse,
    ReviewHistoryResponse,
    SingleDraftResponse,
    StartDraftingRequest,
    StartDraftingResponse,
)

router = APIRouter(tags=["drafting"])


@router.post("/start", response_model=StartDraftingResponse)
async def start_drafting_route(
    request: StartDraftingRequest,
) -> StartDraftingResponse:
    """Start evidence retrieval and draft generation for all requirements of an extracted RFP.

    The workspace is resolved server-side; clients cannot select a workspace.
    """
    return await start_drafting_controller(request)


@router.get("/{doc_id}", response_model=DraftListResponse)
async def list_drafts_route(doc_id: str) -> DraftListResponse:
    """List all draft responses for a document's requirements."""
    return await list_drafts_controller(doc_id)


@router.get("/{doc_id}/status", response_model=DraftingStatusResponse)
async def get_drafting_status_route(doc_id: str) -> DraftingStatusResponse:
    """Report drafting status and errors for a document."""
    return await get_drafting_status_controller(doc_id)


@router.get("/{doc_id}/requirements/{requirement_key}", response_model=SingleDraftResponse)
async def get_draft_route(
    doc_id: str, requirement_key: str
) -> SingleDraftResponse:
    """Retrieve a single requirement's draft with full evidence and run details."""
    return await get_draft_controller(doc_id, requirement_key)


@router.post("/{doc_id}/requirements/{requirement_key}/review", response_model=ReviewDraftResponse)
async def review_draft_route(
    doc_id: str, requirement_key: str, request: ReviewDraftRequest
) -> ReviewDraftResponse:
    """Submit a human review for a requirement draft.

    The reviewer can:
    - Provide an edited answer (replaces AI-generated draft)
    - Override the compliance status and rationale
    - Record per-citation review decisions (accept/reject/modify)
    - Approve the draft or request changes
    - Provide a concise comment/reason

    All original AI-generated content is preserved. Review decisions are stored separately.
    """
    return await review_draft_controller(doc_id, requirement_key, request)


@router.get("/{doc_id}/requirements/{requirement_key}/history", response_model=ReviewHistoryResponse)
async def get_review_history_route(
    doc_id: str, requirement_key: str
) -> ReviewHistoryResponse:
    """Retrieve the complete append-only review audit history for a requirement.

    Returns events ordered chronologically (oldest first).
    """
    return await get_review_history_controller(doc_id, requirement_key)