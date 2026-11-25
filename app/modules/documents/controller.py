"""HTTP controller for document ingestion and queries."""

import logging
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile

from app.ai.embedding_factory import get_embedding_provider
from app.modules.documents.schema import (
    ChunkListResponse,
    DocumentIngestResponse,
    DocumentListResponse,
)
from app.modules.documents.service import (
    SUPPORTED_EXTENSIONS,
    async_get_document,
    async_get_document_chunks,
    async_ingest_document_from_text,
    async_list_documents,
    derive_doc_id,
    read_text_content,
)
from app.rag.store_factory import get_vector_store
from app.rag.stores.postgres_vector_store import PostgreSQLVectorStore

logger = logging.getLogger(__name__)


def _resolve_store_and_embedder(workspace_id: Optional[str]):
    """Resolve the embedding provider and a workspace-scoped vector store.

    When a ``workspace_id`` is supplied and the configured store is a
    PostgreSQL store, a new store instance scoped to that workspace is created
    so queries cannot leak across workspaces. Otherwise the application-wide
    singleton store is used (workspace filtering is then applied via chunk
    metadata for in-memory / persistent stores).
    """
    embedder = get_embedding_provider()
    store = get_vector_store()
    if workspace_id is not None and isinstance(store, PostgreSQLVectorStore):
        store = PostgreSQLVectorStore(workspace_id=workspace_id)
    return store, embedder


async def ingest_document_controller(
    file: UploadFile, workspace_id: Optional[str] = None
) -> DocumentIngestResponse:
    """Validate an uploaded document, ingest it into the vector store, and return the result.

    Supports ``.pdf``, ``.txt`` and ``.md``. PDFs are extracted page by page
    with absolute character offsets so every stored chunk carries a traceable
    source reference.
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

    doc_id = derive_doc_id(file.filename)
    source_filename = file.filename or ""

    # --- Resolve singletons (tenant-scoped store + embedder) ---
    store, embedder = _resolve_store_and_embedder(workspace_id)

    pages = None
    text = ""

    if suffix == ".pdf":
        # Extract PDF page by page with absolute character offsets.
        from app.modules.documents.service import (
            _extract_pdf_pages_from_bytes,
        )
        from app.rag.ingest import build_normalized_text

        try:
            pages = _extract_pdf_pages_from_bytes(content, source_filename)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        text = build_normalized_text(pages)
        if not text.strip():
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in PDF",
            )
    else:
        text = read_text_content(content)

    if not text or not text.strip():
        raise HTTPException(
            status_code=400,
            detail="File is empty or contains no text content",
        )

    metadata = {"source_filename": source_filename}

    # --- Ingest (async) ---
    try:
        chunk_count = await async_ingest_document_from_text(
            text=text,
            doc_id=doc_id,
            store=store,
            embedder=embedder,
            metadata=metadata,
            pages=pages,
            source_filename=source_filename,
            workspace_id=workspace_id,
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
        workspace_id=workspace_id,
    )


async def list_documents_controller(
    workspace_id: Optional[str] = None,
) -> DocumentListResponse:
    """List ingested documents, optionally scoped to a workspace."""
    store = get_vector_store()

    try:
        documents = await async_list_documents(store, workspace_id=workspace_id)
    except NotImplementedError as exc:
        logger.warning("Store does not support document listing: %s", exc)
        raise HTTPException(
            status_code=501,
            detail="Document listing is not supported by the configured store.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error listing documents")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {exc}",
        ) from exc

    return DocumentListResponse(documents=documents)


async def get_document_controller(
    doc_id: str, workspace_id: Optional[str] = None
):
    """Get metadata for a single ingested document."""
    store = get_vector_store()
    try:
        document = await async_get_document(
            store, doc_id, workspace_id=workspace_id
        )
    except NotImplementedError as exc:
        logger.warning("Store does not support document lookup: %s", exc)
        raise HTTPException(
            status_code=501,
            detail="Document lookup is not supported by the configured store.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error getting document")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document: {exc}",
        ) from exc

    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found",
        )
    return document


async def get_document_chunks_controller(
    doc_id: str, workspace_id: Optional[str] = None
) -> ChunkListResponse:
    """Get all chunks for a document with their source references."""
    store = get_vector_store()
    try:
        chunks = await async_get_document_chunks(
            store, doc_id, workspace_id=workspace_id
        )
    except NotImplementedError as exc:
        logger.warning("Store does not support chunk lookup: %s", exc)
        raise HTTPException(
            status_code=501,
            detail="Chunk lookup is not supported by the configured store.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error getting document chunks")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document chunks: {exc}",
        ) from exc

    return ChunkListResponse(doc_id=doc_id, chunks=chunks)
