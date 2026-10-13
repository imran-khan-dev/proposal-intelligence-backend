"""PostgreSQL + pgvector vector store implementation."""

import asyncio
import os
import uuid
from typing import Any, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from app.core.database import get_async_session_factory, is_database_initialized
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.tenant import Tenant
from app.rag.store import SearchResult, VectorRecord, VectorStore


EXPECTED_DIMENSION = 384


def _run_async(coro):
    """Run an async coroutine in a synchronous context.

    Handles both cases: when there's no running event loop (CLI, scripts)
    and when there is one (FastAPI with sync controller).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(coro)

    # There's a running loop - we're likely in an async context but called from sync code
    # Create a new task and run it in the background, then wait for it
    # This is a workaround; ideally callers would be async
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


class PostgreSQLVectorStore(VectorStore):
    """PostgreSQL-backed vector store using pgvector for similarity search.

    Uses the application-scoped async engine and session factory.
    Requires database to be initialized via initialize_database().

    Tenant scope is determined by the TENANT_ID environment variable
    (default: "default-tenant"). This is a configuration-based approach
    since authentication/authorization is not yet implemented.
    """

    def __init__(self, tenant_id: Optional[str] = None):
        if not is_database_initialized():
            raise RuntimeError(
                "Database not initialized. Call initialize_database() at startup."
            )

        self._tenant_id_str = tenant_id or os.environ.get("TENANT_ID", "default-tenant")
        # Generate a deterministic UUID from the tenant string
        self._tenant_id = uuid.uuid5(uuid.NAMESPACE_DNS, self._tenant_id_str)
        self._dimension = EXPECTED_DIMENSION
        self._session_factory = get_async_session_factory()

    async def _get_or_create_tenant(self, session) -> Tenant:
        """Get or create the tenant for this store."""
        stmt = select(Tenant).where(Tenant.id == self._tenant_id)
        result = await session.execute(stmt)
        tenant = result.scalar_one_or_none()

        if tenant is None:
            tenant = Tenant(id=self._tenant_id, name=self._tenant_id_str)
            session.add(tenant)
            await session.flush()

        return tenant

    async def _get_or_create_document(self, session, doc_id: str) -> Document:
        """Get or create a document for the given doc_id."""
        stmt = select(Document).where(
            Document.tenant_id == self._tenant_id,
            Document.doc_id == doc_id,
        )
        result = await session.execute(stmt)
        document = result.scalar_one_or_none()

        if document is None:
            document = Document(tenant_id=self._tenant_id, doc_id=doc_id)
            session.add(document)
            await session.flush()

        return document

    def _validate_embedding_dimension(self, embedding: List[float]) -> None:
        """Validate that embedding has the expected dimension."""
        if len(embedding) != EXPECTED_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EXPECTED_DIMENSION}, "
                f"got {len(embedding)}. This store only supports {EXPECTED_DIMENSION}-dimensional vectors."
            )

    def _record_to_vector_record(self, chunk: DocumentChunk) -> VectorRecord:
        """Convert a DocumentChunk to a VectorRecord."""
        return VectorRecord(
            id=str(chunk.id),
            text=chunk.text,
            embedding=list(chunk.embedding),
            metadata=chunk.chunk_metadata,
        )

    def add(self, records: List[VectorRecord]) -> None:
        """Add records to the store."""
        if not records:
            return

        # Validate all records have embeddings and correct dimension
        for record in records:
            if record.embedding is None or len(record.embedding) == 0:
                raise ValueError(f"Record {record.id} has empty embedding")
            self._validate_embedding_dimension(record.embedding)

        # Group records by doc_id from metadata
        from collections import defaultdict
        records_by_doc: dict[str, List[VectorRecord]] = defaultdict(list)
        for record in records:
            doc_id = record.metadata.get("doc_id")
            if not doc_id:
                raise ValueError(f"Record {record.id} missing required 'doc_id' in metadata")
            records_by_doc[doc_id].append(record)

        async def _add_async():
            async with self._session_factory() as session:
                tenant = await self._get_or_create_tenant(session)

                for doc_id, doc_records in records_by_doc.items():
                    document = await self._get_or_create_document(session, doc_id)

                    # Delete existing chunks for this document
                    await session.execute(
                        delete(DocumentChunk).where(DocumentChunk.document_id == document.id)
                    )

                    # Add new chunks
                    for record in doc_records:
                        chunk_index = record.metadata.get("chunk_index", 0)
                        chunk = DocumentChunk(
                            document_id=document.id,
                            chunk_index=chunk_index,
                            text=record.text,
                            embedding=record.embedding,
                            metadata=record.metadata,
                        )
                        session.add(chunk)

                await session.commit()

        _run_async(_add_async())

    def search(
        self, query_embedding: List[float], k: int = 5
    ) -> List[SearchResult]:
        """Search for the top-k most similar records using pgvector cosine similarity."""
        if k <= 0:
            return []

        self._validate_embedding_dimension(query_embedding)

        async def _search_async():
            async with self._session_factory() as session:
                # Use pgvector cosine distance operator (<=>)
                # Cosine similarity = 1 - cosine_distance
                stmt = (
                    select(
                        DocumentChunk,
                        (1 - DocumentChunk.embedding.cosine_distance(query_embedding)).label(
                            "similarity"
                        ),
                    )
                    .join(Document, DocumentChunk.document_id == Document.id)
                    .where(Document.tenant_id == self._tenant_id)
                    .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
                    .limit(k)
                )

                result = await session.execute(stmt)
                rows = result.all()

                search_results = []
                for chunk, similarity in rows:
                    # Filter out invalid similarities (shouldn't happen with normalized vectors)
                    if similarity is not None and similarity > -1.0:
                        search_results.append(
                            SearchResult(
                                record=self._record_to_vector_record(chunk),
                                score=float(similarity),
                            )
                        )

                return search_results

        return _run_async(_search_async())

    def delete_by_metadata(self, metadata_filter: dict[str, Any]) -> int:
        """Delete records matching a metadata filter."""
        if not metadata_filter:
            return 0

        # We need to find documents whose chunks match the metadata filter
        # For simplicity, we'll filter by doc_id if present, otherwise we need to scan
        doc_id = metadata_filter.get("doc_id")

        async def _delete_async():
            async with self._session_factory() as session:
                if doc_id:
                    # Fast path: delete by doc_id
                    stmt = select(Document).where(
                        Document.tenant_id == self._tenant_id,
                        Document.doc_id == doc_id,
                    )
                    result = await session.execute(stmt)
                    document = result.scalar_one_or_none()

                    if document is None:
                        return 0

                    # Delete chunks for this document
                    delete_stmt = delete(DocumentChunk).where(
                        DocumentChunk.document_id == document.id
                    )
                    result = await session.execute(delete_stmt)
                    deleted_count = result.rowcount

                    # Optionally delete the document if it has no more chunks
                    # For now, keep the document record
                    await session.commit()
                    return deleted_count
                else:
                    # Slow path: scan all chunks for the tenant and filter by metadata
                    # This is inefficient but matches the JSONL store behavior
                    stmt = (
                        select(DocumentChunk)
                        .join(Document, DocumentChunk.document_id == Document.id)
                        .where(Document.tenant_id == self._tenant_id)
                    )
                    result = await session.execute(stmt)
                    chunks = result.scalars().all()

                    deleted = 0
                    for chunk in chunks:
                        if all(
                            chunk.chunk_metadata.get(k) == v for k, v in metadata_filter.items()
                        ):
                            await session.delete(chunk)
                            deleted += 1

                    await session.commit()
                    return deleted

        return _run_async(_delete_async())

    def clear(self) -> None:
        """Remove all records for the current tenant."""

        async def _clear_async():
            async with self._session_factory() as session:
                # Delete all chunks for tenant's documents
                stmt = delete(DocumentChunk).where(
                    DocumentChunk.document_id.in_(
                        select(Document.id).where(Document.tenant_id == self._tenant_id)
                    )
                )
                await session.execute(stmt)

                # Optionally delete documents too
                doc_delete_stmt = delete(Document).where(
                    Document.tenant_id == self._tenant_id
                )
                await session.execute(doc_delete_stmt)

                await session.commit()

        _run_async(_clear_async())

    def count(self) -> int:
        """Return the number of records for the current tenant."""

        async def _count_async():
            async with self._session_factory() as session:
                stmt = select(func.count(DocumentChunk.id)).where(
                    DocumentChunk.document_id.in_(
                        select(Document.id).where(Document.tenant_id == self._tenant_id)
                    )
                )
                result = await session.execute(stmt)
                return result.scalar() or 0

        return _run_async(_count_async())

    @property
    def dimension(self) -> Optional[int]:
        """Return the embedding dimension of stored vectors."""
        return EXPECTED_DIMENSION