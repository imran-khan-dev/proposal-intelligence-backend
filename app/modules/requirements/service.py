"""Requirement extraction service.

Orchestrates the pipeline:

    stored chunks (or page segments)
        -> reconstruct normalized pages (reusing ingestion source refs)
        -> structured parse
        -> requirement extraction + hybrid classification
        -> persist requirements + run status

The service is persistence-agnostic (takes a
:class:`~app.modules.requirements.repository.RequirementRepository`) and
provider-optional (LLM classification is best-effort).  No background-job
infrastructure is used: extraction runs inline within the request and the run
record captures status/errors for later reporting.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from app.modules.requirements.extraction import (
    ExtractedRequirement,
    extract_requirements,
)
from app.modules.requirements.parsing import (
    StructuredTextParser,
    reconstruct_pages_from_chunks,
)
from app.modules.requirements.repository import RequirementRepository
from app.modules.requirements.schema import (
    ExtractionRunRecord,
    RequirementRecord,
)

logger = logging.getLogger(__name__)

_parser = StructuredTextParser()


async def extract_from_pages(
    doc_id: str,
    filename: str,
    pages: List[dict],
    workspace_id: uuid.UUID,
    repo: RequirementRepository,
    provider=None,
    document_id: Optional[uuid.UUID] = None,
) -> tuple[ExtractionRunRecord, List[ExtractedRequirement], List[str]]:
    """Parse, extract, and persist requirements from page segments.

    Returns:
        ``(run_record, requirements, warnings)``.
    """
    run = await repo.create_run(workspace_id, doc_id)
    try:
        doc = _parser.parse(doc_id, filename, pages)
        requirements = extract_requirements(doc, provider=provider)
        await repo.save_requirements(
            run.run_id,
            workspace_id,
            doc_id,
            requirements,
            document_id=document_id,
        )
        await repo.complete_run(run.run_id, len(requirements))
        run.status = "completed"
        run.requirement_count = len(requirements)
        return run, requirements, list(doc.extraction_warnings)
    except Exception as exc:  # noqa: BLE001 - record and re-raise as failure
        logger.exception("Requirement extraction failed for doc %s", doc_id)
        errors = [f"{type(exc).__name__}: {exc}"]
        await repo.fail_run(run.run_id, errors)
        run.status = "failed"
        run.errors = errors
        raise


async def extract_from_chunks(
    doc_id: str,
    chunks: List[dict],
    workspace_id: uuid.UUID,
    repo: RequirementRepository,
    provider=None,
    filename: Optional[str] = None,
    document_id: Optional[uuid.UUID] = None,
) -> tuple[ExtractionRunRecord, List[ExtractedRequirement], List[str]]:
    """Reconstruct pages from stored chunks, then extract requirements.

    Reuses the source references captured at ingestion; the original file is
    not re-read.
    """
    pages = reconstruct_pages_from_chunks(chunks)
    if not pages:
        raise ValueError(
            f"No usable source references found for document '{doc_id}'"
        )
    if filename is None:
        filename = _first_source_filename(chunks) or doc_id
    return await extract_from_pages(
        doc_id=doc_id,
        filename=filename,
        pages=pages,
        workspace_id=workspace_id,
        repo=repo,
        provider=provider,
        document_id=document_id,
    )


def _first_source_filename(chunks: List[dict]) -> Optional[str]:
    for c in chunks:
        name = c.get("source_filename")
        if name:
            return name
    return None


async def list_requirements(
    doc_id: str,
    workspace_id: uuid.UUID,
    repo: RequirementRepository,
) -> List[RequirementRecord]:
    """Return the requirements for a document's latest run."""
    return await repo.list_requirements(workspace_id, doc_id)


async def get_requirement(
    doc_id: str,
    requirement_key: str,
    workspace_id: uuid.UUID,
    repo: RequirementRepository,
) -> Optional[RequirementRecord]:
    """Return a single requirement by key, or ``None``."""
    return await repo.get_requirement(workspace_id, doc_id, requirement_key)


async def get_status(
    doc_id: str,
    workspace_id: uuid.UUID,
    repo: RequirementRepository,
) -> Optional[ExtractionRunRecord]:
    """Return the latest extraction run status for a document, or ``None``."""
    return await repo.get_latest_run(workspace_id, doc_id)