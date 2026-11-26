"""HTTP controller for requirement-level evidence retrieval and drafting."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import HTTPException

from app.ai.factory import get_rfp_provider
from app.ai.embedding_factory import get_embedding_provider
from app.ai.error_handling import handle_ai_exception
from app.core.workspace import resolve_workspace_id, workspace_id_str
from app.modules.drafting.repository import PostgresDraftingRepository
from app.modules.drafting.schema import (
    CitationVerificationResponse,
    DraftListResponse,
    DraftingStatusResponse,
    EvidenceCitationResponse,
    EvidencePassageResponse,
    RequirementDraftResponse,
    SingleDraftResponse,
    StartDraftingRequest,
    StartDraftingResponse,
)
from app.modules.drafting.generation import generate_draft_for_requirement
from app.modules.drafting.retrieval import (
    EvidencePassage,
    _build_retrieval_query,
    retrieve_evidence_for_requirement,
)
from app.modules.drafting.verification import verify_citations
from app.modules.requirements.repository import PostgresRequirementRepository
from app.modules.requirements.service import list_requirements
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


def _citation_verifications_from_stored(citations: list) -> List[CitationVerificationResponse]:
    """Build verification responses from persisted citation dicts (M3.3).

    Citations stored before verification ran (no ``verified`` key) yield an
    empty list so legacy drafts remain readable.
    """
    results: List[CitationVerificationResponse] = []
    for c in citations or []:
        if not isinstance(c, dict) or c.get("verified") is None:
            continue
        results.append(
            CitationVerificationResponse(
                evidence_id=c.get("evidence_id", ""),
                excerpt=c.get("excerpt", ""),
                verified=bool(c.get("verified")),
                reason=c.get("reason", ""),
                source_doc_id=c.get("source_doc_id"),
                workspace_id=c.get("workspace_id"),
            )
        )
    return results


async def start_drafting_controller(
    request: StartDraftingRequest,
    workspace_id: Optional[str] = None,
) -> StartDraftingResponse:
    """Start evidence retrieval and draft generation for all requirements of an extracted RFP.

    This is the main M3 entry point. It:
    1. Lists all extracted requirements for the document
    2. For each requirement, retrieves evidence from company knowledge base
    3. Generates an evidence-backed draft response
    4. Persists all runs and results

    Execution is synchronous and bounded (no background jobs).
    """
    ws_uuid = _resolve_workspace_id(workspace_id)
    ws_str = workspace_id_str(ws_uuid)

    # Get the extracted requirements for this document
    req_repo = PostgresRequirementRepository()
    try:
        requirements = await list_requirements(request.doc_id, ws_uuid, req_repo)
    except Exception as exc:
        logger.exception("Failed to list requirements for %s", request.doc_id)
        raise HTTPException(
            status_code=500, detail=f"Failed to list requirements: {exc}"
        ) from exc

    if not requirements:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No extracted requirements found for document '{request.doc_id}'. "
                "Run requirement extraction first via POST /api/requirements/extract."
            ),
        )

    # Initialize repositories and providers
    drafting_repo = PostgresDraftingRepository()
    store = _scoped_store(ws_uuid)
    embedder = get_embedding_provider()
    try:
        provider = get_rfp_provider()
    except Exception as exc:
        logger.info("AI provider unavailable; drafts will be evidence-gap only: %s", exc)
        provider = None

    # Process each requirement
    evidence_runs_created = 0
    draft_runs_created = 0
    errors = []

    for req in requirements:
        try:
            # Get the requirement's database UUID
            requirement_id = await req_repo.get_requirement_id(ws_uuid, request.doc_id, req.requirement_key)
            if requirement_id is None:
                raise ValueError(f"Requirement {req.requirement_key} not found in database")

            # Build retrieval query
            category_str = req.category.value if hasattr(req.category, 'value') else str(req.category)
            retrieval_query = _build_retrieval_query(
                requirement_text=req.text,
                section_title=req.section_title,
                section_path=req.section_path,
                category=category_str,
            )

            # Create evidence retrieval run
            evidence_run = await drafting_repo.create_evidence_run(
                workspace_id=ws_uuid,
                requirement_id=requirement_id,
                query_text=retrieval_query,
            )
            evidence_runs_created += 1

            # Retrieve evidence
            passages = retrieve_evidence_for_requirement(
                requirement_text=req.text,
                workspace_id=ws_uuid,
                embedder=embedder,
                store=store,
                section_title=req.section_title,
                section_path=req.section_path,
                category=category_str,
                top_k=request.top_k,
            )

            # Complete evidence run
            await drafting_repo.complete_evidence_run(
                run_id=evidence_run.run_id,
                evidence_count=len(passages),
                evidence_passages=passages,
            )

            # Create draft generation run
            draft_run = await drafting_repo.create_draft_run(
                workspace_id=ws_uuid,
                requirement_id=requirement_id,
                retrieval_run_id=evidence_run.run_id,
            )
            draft_runs_created += 1

            # Generate draft
            draft = generate_draft_for_requirement(
                requirement_text=req.text,
                requirement_key=req.requirement_key,
                requirement_id=requirement_id,
                passages=passages,
                section_title=req.section_title,
                category=category_str,
                provider=provider,
            )

            # Independently verify citations against the retrieved evidence
            # (M3.3). Runs after generation; the draft answer is preserved and
            # failed citations are reported, not discarded.
            draft.citation_verifications = verify_citations(
                draft.citations, passages, ws_uuid
            )

            # Complete draft run
            await drafting_repo.complete_draft_run(
                run_id=draft_run.run_id,
                draft=draft,
            )

        except Exception as exc:
            logger.exception("Drafting failed for requirement %s: %s", req.requirement_key, exc)
            errors.append(f"{req.requirement_key}: {type(exc).__name__}: {exc}")

    # Determine overall status
    if errors and len(errors) == len(requirements):
        overall_status = "failed"
    elif errors:
        overall_status = "partial_failure"
    else:
        overall_status = "completed"

    return StartDraftingResponse(
        doc_id=request.doc_id,
        requirements_processed=len(requirements),
        evidence_runs_created=evidence_runs_created,
        draft_runs_created=draft_runs_created,
        status=overall_status,
        errors=errors,
    )


async def list_drafts_controller(
    doc_id: str,
    workspace_id: Optional[str] = None,
) -> DraftListResponse:
    """List all draft responses for a document's requirements."""
    ws_uuid = _resolve_workspace_id(workspace_id)

    # Get requirements for this document
    req_repo = PostgresRequirementRepository()
    try:
        requirements = await list_requirements(doc_id, ws_uuid, req_repo)
    except Exception as exc:
        logger.exception("Failed to list requirements for %s", doc_id)
        raise HTTPException(
            status_code=500, detail=f"Failed to list requirements: {exc}"
        ) from exc

    if not requirements:
        return DraftListResponse(doc_id=doc_id, drafts=[])

    # Get latest draft for each requirement
    drafting_repo = PostgresDraftingRepository()
    drafts = []

    for req in requirements:
        try:
            requirement_id = await req_repo.get_requirement_id(ws_uuid, doc_id, req.requirement_key)
            if requirement_id is None:
                continue

            draft_run = await drafting_repo.get_latest_draft_run(
                workspace_id=ws_uuid,
                requirement_id=requirement_id,
            )
            if draft_run and draft_run.draft_answer is not None:
                drafts.append(
                    RequirementDraftResponse(
                        requirement_id=draft_run.requirement_id,
                        requirement_key=req.requirement_key,
                        requirement_text=req.text,
                        draft_answer=draft_run.draft_answer,
                        citations=[
                            EvidenceCitationResponse(
                                evidence_id=c.get("evidence_id", ""),
                                excerpt=c.get("excerpt", ""),
                                score=c.get("score", 0.0),
                            )
                            for c in draft_run.citations
                        ],
                        evidence_found=draft_run.evidence_found,
                        uncertainty=draft_run.uncertainty,
                        uncertainty_notes=draft_run.uncertainty_notes,
                        citation_verifications=_citation_verifications_from_stored(
                            draft_run.citations
                        ),
                    )
                )
        except Exception as exc:
            logger.warning("Failed to get draft for %s: %s", req.requirement_key, exc)

    return DraftListResponse(doc_id=doc_id, drafts=drafts)


async def get_draft_controller(
    doc_id: str,
    requirement_key: str,
    workspace_id: Optional[str] = None,
) -> SingleDraftResponse:
    """Retrieve a single requirement's draft with full evidence and run details."""
    ws_uuid = _resolve_workspace_id(workspace_id)

    # Get the requirement
    req_repo = PostgresRequirementRepository()
    try:
        requirement = await req_repo.get_requirement(ws_uuid, doc_id, requirement_key)
    except Exception as exc:
        logger.exception("Failed to get requirement %s for %s", requirement_key, doc_id)
        raise HTTPException(
            status_code=500, detail=f"Failed to get requirement: {exc}"
        ) from exc

    if requirement is None:
        raise HTTPException(
            status_code=404,
            detail=f"Requirement '{requirement_key}' not found for document '{doc_id}'",
        )

    requirement_id = await req_repo.get_requirement_id(ws_uuid, doc_id, requirement_key)
    if requirement_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"Requirement '{requirement_key}' not found in database",
        )

    drafting_repo = PostgresDraftingRepository()

    # Get latest draft run
    draft_run = await drafting_repo.get_latest_draft_run(
        workspace_id=ws_uuid,
        requirement_id=requirement_id,
    )

    # Get latest evidence run
    evidence_run = await drafting_repo.get_latest_evidence_run(
        workspace_id=ws_uuid,
        requirement_id=requirement_id,
    )

    # Get evidence passages
    evidence_passages = []
    retrieval_query = ""
    if evidence_run:
        retrieval_query = evidence_run.query_text
        evidence_records = await drafting_repo.list_evidence_for_run(evidence_run.run_id)
        evidence_passages = [
            EvidencePassageResponse(
                text=e.text,
                score=e.score,
                rank=e.rank,
                source_doc_id=e.source_doc_id,
                source_filename=e.source_filename,
                chunk_id=e.chunk_id,
                chunk_index=e.chunk_index,
                page_number=e.page_number,
                character_start=e.character_start,
                character_end=e.character_end,
                section_path=e.section_path,
            )
            for e in evidence_records
        ]

    # Build draft response
    draft_response = None
    if draft_run and draft_run.draft_answer is not None:
        draft_response = RequirementDraftResponse(
            requirement_id=draft_run.requirement_id,
            requirement_key=requirement.requirement_key,
            requirement_text=requirement.text,
            draft_answer=draft_run.draft_answer,
            citations=[
                EvidenceCitationResponse(
                    evidence_id=c.get("evidence_id", ""),
                    excerpt=c.get("excerpt", ""),
                    score=c.get("score", 0.0),
                )
                for c in draft_run.citations
            ],
            evidence_found=draft_run.evidence_found,
            uncertainty=draft_run.uncertainty,
            uncertainty_notes=draft_run.uncertainty_notes,
            citation_verifications=_citation_verifications_from_stored(
                draft_run.citations
            ),
        )
    else:
        draft_response = RequirementDraftResponse(
            requirement_id=requirement_id,
            requirement_key=requirement.requirement_key,
            requirement_text=requirement.text,
            draft_answer=None,
            citations=[],
            evidence_found=False,
            uncertainty="none",
            uncertainty_notes=["Draft not yet generated - use POST /api/drafting/start"],
        )

    return SingleDraftResponse(
        requirement=draft_response,
        retrieval_query=retrieval_query,
        evidence_passages=evidence_passages,
        retrieval_run=evidence_run,
        draft_run=draft_run,
    )


async def get_drafting_status_controller(
    doc_id: str,
    workspace_id: Optional[str] = None,
) -> DraftingStatusResponse:
    """Report drafting status and errors for a document."""
    ws_uuid = _resolve_workspace_id(workspace_id)

    req_repo = PostgresRequirementRepository()
    try:
        requirements = await list_requirements(doc_id, ws_uuid, req_repo)
    except Exception as exc:
        logger.exception("Failed to list requirements for %s", doc_id)
        raise HTTPException(
            status_code=500, detail=f"Failed to list requirements: {exc}"
        ) from exc

    drafting_repo = PostgresDraftingRepository()

    # Count runs by checking each requirement
    requirements_total = len(requirements)
    evidence_runs_completed = 0
    draft_runs_completed = 0
    evidence_runs_failed = 0
    draft_runs_failed = 0

    for req in requirements:
        requirement_id = await req_repo.get_requirement_id(ws_uuid, doc_id, req.requirement_key)
        if requirement_id is None:
            continue

        # Check evidence run
        evidence_run = await drafting_repo.get_latest_evidence_run(ws_uuid, requirement_id)
        if evidence_run:
            if evidence_run.status == "completed":
                evidence_runs_completed += 1
            elif evidence_run.status == "failed":
                evidence_runs_failed += 1

        # Check draft run
        draft_run = await drafting_repo.get_latest_draft_run(ws_uuid, requirement_id)
        if draft_run:
            if draft_run.status == "completed":
                draft_runs_completed += 1
            elif draft_run.status == "failed":
                draft_runs_failed += 1

    # Determine overall status
    if requirements_total == 0:
        overall_status = "pending"
    elif draft_runs_completed == requirements_total and evidence_runs_completed == requirements_total:
        overall_status = "completed"
    elif draft_runs_failed > 0 or evidence_runs_failed > 0:
        overall_status = "partial_failure" if (draft_runs_completed > 0 or evidence_runs_completed > 0) else "failed"
    else:
        overall_status = "running"

    return DraftingStatusResponse(
        doc_id=doc_id,
        requirements_total=requirements_total,
        evidence_runs_completed=evidence_runs_completed,
        draft_runs_completed=draft_runs_completed,
        evidence_runs_failed=evidence_runs_failed,
        draft_runs_failed=draft_runs_failed,
        overall_status=overall_status,
    )