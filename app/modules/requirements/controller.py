"""HTTP controller for structured requirement extraction."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import HTTPException

from app.core.workspace import resolve_workspace_id, workspace_id_str
from app.modules.documents.service import async_get_document_chunks
from app.modules.requirements.repository import PostgresRequirementRepository
from app.modules.requirements.schema import (
    ExtractResponse,
    ExtractionStatusResponse,
    RequirementDetailResponse,
    RequirementListResponse,
    SourceReference,
)
from app.modules.requirements.service import (
    extract_from_chunks,
    get_requirement,
    get_status,
    list_requirements,
)
from app.rag.store_factory import get_vector_store
from app.rag.stores.postgres_vector_store import PostgreSQLVectorStore

logger = logging.getLogger(__name__)


def _resolve_workspace_id(workspace_id: Optional[str] = None) -> uuid.UUID:
    """Return the server-resolved workspace; never trust client input."""
    if workspace_id:
        if isinstance(workspace_id, uuid.UUID):
            return workspace_id
        try:
            return uuid.UUID(workspace_id)
        except (ValueError, AttributeError):
            return resolve_workspace_id()
    return resolve_workspace_id()


def _scoped_store(workspace_id: uuid.UUID):
    """Return a workspace-scoped vector store for reading source chunks."""
    store = get_vector_store()
    if isinstance(store, PostgreSQLVectorStore):
        store = PostgreSQLVectorStore(workspace_id=workspace_id)
    return store


def _get_provider_optional():
    """Return the RFP AI provider, or ``None`` if it cannot be created.

    Extraction still works rule-based when no provider is configured; the
    ``None`` case is logged rather than failing the request.
    """
    try:
        from app.ai.factory import get_rfp_provider

        return get_rfp_provider()
    except Exception as exc:  # noqa: BLE001
        logger.info("AI provider unavailable; using rule-based only: %s", exc)
        return None


async def extract_requirements_controller(
    doc_id: str,
    workspace_id: Optional[str] = None,
) -> ExtractResponse:
    """Start requirement extraction for an ingested RFP (synchronous)."""
    ws_uuid = _resolve_workspace_id(workspace_id)
    ws_str = workspace_id_str(ws_uuid)
    store = _scoped_store(ws_uuid)

    try:
        chunks = await async_get_document_chunks(
            store, doc_id, workspace_id=ws_str
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=501,
            detail="Configured store does not support chunk lookup.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to read chunks for %s", doc_id)
        raise HTTPException(
            status_code=500, detail=f"Failed to read document: {exc}"
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Document '{doc_id}' not found in the resolved workspace. "
                "Ingest it first via POST /api/documents/ingest."
            ),
        )

    repo = PostgresRequirementRepository()
    provider = _get_provider_optional()

    try:
        run, requirements, warnings = await extract_from_chunks(
            doc_id=doc_id,
            chunks=chunks,
            workspace_id=ws_uuid,
            repo=repo,
            provider=provider,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        from app.ai.error_handling import handle_ai_exception

        logger.exception("Requirement extraction failed for %s", doc_id)
        raise handle_ai_exception(exc)

    return ExtractResponse(
        doc_id=doc_id,
        run_id=run.run_id,
        status=run.status,
        requirement_count=len(requirements),
        warnings=warnings,
    )


async def list_requirements_controller(
    doc_id: str,
    workspace_id: Optional[str] = None,
) -> RequirementListResponse:
    """List extracted requirements for a document in the resolved workspace."""
    ws_uuid = _resolve_workspace_id(workspace_id)
    repo = PostgresRequirementRepository()
    try:
        requirements = await list_requirements(doc_id, ws_uuid, repo)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to list requirements for %s", doc_id)
        raise HTTPException(
            status_code=500, detail=f"Failed to list requirements: {exc}"
        ) from exc
    return RequirementListResponse(doc_id=doc_id, requirements=requirements)


async def get_requirement_controller(
    doc_id: str,
    requirement_key: str,
    workspace_id: Optional[str] = None,
) -> RequirementDetailResponse:
    """Return a single requirement with its source reference."""
    ws_uuid = _resolve_workspace_id(workspace_id)
    repo = PostgresRequirementRepository()
    try:
        record = await get_requirement(doc_id, requirement_key, ws_uuid, repo)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Failed to get requirement %s for %s", requirement_key, doc_id
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get requirement: {exc}"
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Requirement '{requirement_key}' not found for document "
                f"'{doc_id}'"
            ),
        )

    source = SourceReference(
        doc_id=doc_id,
        source_filename=None,
        page_number=record.page_number,
        character_start=record.character_start,
        character_end=record.character_end,
    )
    return RequirementDetailResponse(requirement=record, source=source)


async def get_status_controller(
    doc_id: str,
    workspace_id: Optional[str] = None,
) -> ExtractionStatusResponse:
    """Report extraction status and errors for a document."""
    ws_uuid = _resolve_workspace_id(workspace_id)
    repo = PostgresRequirementRepository()
    try:
        run = await get_status(doc_id, ws_uuid, repo)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to get status for %s", doc_id)
        raise HTTPException(
            status_code=500, detail=f"Failed to get status: {exc}"
        ) from exc

    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"No extraction run found for document '{doc_id}'",
        )

    return ExtractionStatusResponse(
        doc_id=doc_id,
        run_id=run.run_id,
        status=run.status,
        requirement_count=run.requirement_count,
        errors=run.errors,
    )