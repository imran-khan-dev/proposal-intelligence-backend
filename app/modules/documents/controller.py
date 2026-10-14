"""HTTP controller for document ingestion."""

import logging
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.ai.embedding_factory import get_embedding_provider
from app.modules.documents.schema import DocumentIngestResponse
from app.modules.documents.service import (
    SUPPORTED_EXTENSIONS,
    async_ingest_document_from_text,
    derive_doc_id,
    read_text_content,
)
from app.rag.store_factory import get_vector_store

logger = logging.getLogger(__name__)


async def ingest_document_controller(file: UploadFile) -> DocumentIngestResponse:
    """Validate an uploaded document, ingest it into the vector store, and return the result.

    Tenant scope is resolved through the configured vector-store singleton
    (driven by the ``TENANT_ID`` environment variable), never from client
    input.
    """
    # --- Validate file type ---
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if suffix not in SUPPORTED_EXTENSIONS:
        await file.close()
        suffix_display = suffix or "(no extension)"
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{suffix_display}'. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # --- Read file content ---
    try:
        content = await file.read()
    except Exception as exc:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {exc}",
        ) from exc
    finally:
        await file.close()

    # --- Validate content ---
    if not content or not content.strip():
        raise HTTPException(
            status_code=400,
            detail="File is empty or contains no text content",
        )

    text = read_text_content(content)
    doc_id = derive_doc_id(file.filename)

    # --- Resolve singletons (tenant-scoped store + embedder) ---
    # These are resolved through the factory which returns the application
    # singleton if initialized, or creates a new instance (CLI fallback).
    store = get_vector_store()
    embedder = get_embedding_provider()

    # --- Ingest (async) ---
    try:
        chunk_count = await async_ingest_document_from_text(
            text=text,
            doc_id=doc_id,
            store=store,
            embedder=embedder,
            metadata={"source_filename": file.filename or ""},
        )
    except ValueError as exc:
        logger.warning("Document ingestion validation failed: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.warning("Embedding or provider error during ingestion: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during document ingestion")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    return DocumentIngestResponse(
        doc_id=doc_id,
        chunk_count=chunk_count,
        message=f"Successfully ingested '{doc_id}' ({chunk_count} chunks)",
    )
