"""PostgreSQL + pgvector vector store implementation."""

import asyncio
import uuid
from typing import Any, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from app.core.database import get_async_session_factory, is_database_initialized
from app.core.workspace import resolve_workspace_id
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.tenant import Tenant
from app.rag.store import SearchResult, VectorRecord, VectorStore


EXPECTED_DIMENSION = 384


def _run_async(coro):
    """Run an async coroutine in a synchronous context.

    This is a fallback for CLI/scripts where there's no running event loop.
    In FastAPI (async context), callers should use the async methods directly.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(coro)

    # There's a running loop - we're in an async context but called from sync code.
    # This should not happen in normal FastAPI operation if callers use async methods.
    # Raise an error to catch misuse early.
    raise RuntimeError(
        "PostgreSQLVectorStore sync methods called from an async context. "
        "Use the async_* methods (e.g., async_add, async_search) instead."
    )


def _summarize_chunks(chunks: List[DocumentChunk]) -> List[dict]:
    """Group DocumentChunk rows into per-document summaries."""
    from collections import OrderedDict

    docs: "OrderedDict[str, dict]" = OrderedDict()
    for chunk in chunks:
        metadata = chunk.chunk_metadata or {}
        doc_id = metadata.get("doc_id")
        if doc_id is None:
            continue
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "source_filename": metadata.get("source_filename"),
                "title": metadata.get("title") or metadata.get("source_filename"),
                "chunk_count": 0,
                "workspace_id": metadata.get("workspace_id"),
                "pages": set(),
            }
        entry = docs[doc_id]
        entry["chunk_count"] += 1
        page_number = metadata.get("page_number")
        if page_number is not None:
            entry["pages"].add(page_number)

    result: List[dict] = []
    for entry in docs.values():
        entry["pages"] = sorted(entry["pages"])
        result.append(entry)
    return result


class PostgreSQLVectorStore(VectorStore):
    """PostgreSQL-backed vector store using pgvector for similarity search.

    Uses the application-scoped async engine and session factory.
    Requires database to be initialized via initialize_database().

    Workspace scope is determined by the ``workspace_id`` constructor argument,
    which must be a UUID.  When not provided, the server-resolved workspace ID
    from :func:`app.core.workspace.resolve_workspace_id` is used.
    """

    def __init__(self, workspace_id: Optional[uuid.UUID] = None):
        if not is_database_initialized():
            raise RuntimeError(
                "Database not initialized. Call initialize_database() at startup."
            )

        if workspace_id is None:
            workspace_id = resolve_workspace_id()
        elif isinstance(workspace_id, str):
            raise ValueError(
                "PostgreSQLVectorStore requires a UUID workspace_id, "
                f"got string {workspace_id!r}"
            )

        self._workspace_id: uuid.UUID = workspace_id
        self._workspace_id_str: str = str(workspace_id)
        self._dimension = EXPECTED_DIMENSION
        self._session_factory = get_async_session_factory()

    async def _get_or_create_tenant(self, session) -> Tenant:
        """Get or create the tenant (workspace) for this store."""
        stmt = select(Tenant).where(Tenant.id == self._workspace_id)
        result = await session.execute(stmt)
        tenant = result.scalar_one_or_none()

        if tenant is None:
            tenant = Tenant(id=self._workspace_id, name=self._workspace_id_str)
            session.add(tenant)
            await session.flush()

        return tenant

    async def _get_or_create_document(
        self, session, doc_id: str, source_type: str = "rfp"
    ) -> Document:
        """Get or create a document for the given doc_id within this workspace."""
        stmt = select(Document).where(
            Document.workspace_id == self._workspace_id,
            Document.doc_id == doc_id,
        )
        result = await session.execute(stmt)
        document = result.scalar_one_or_none()

        if document is None:
            document = Document(
                workspace_id=self._workspace_id,
                doc_id=doc_id,
                tenant_id=self._workspace_id,
                source_type=source_type,
            )
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

    # --- Async methods (primary implementation) ---

    async def async_add(self, records: List[VectorRecord]) -> None:
        """Add records to the store asynchronously."""
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
                raise ValueError(
                    f"Record {record.id} missing required 'doc_id' in metadata"
                )
            records_by_doc[doc_id].append(record)

        async with self._session_factory() as session:
            await self._get_or_create_tenant(session)

            for doc_id, doc_records in records_by_doc.items():
                # Source type is a document-level property; take it from the
                # first chunk's metadata (defaults to ``"rfp"`` so RFP
                # ingestion is never misclassified as company knowledge).
                source_type = doc_records[0].metadata.get("source_type", "rfp")
                document = await self._get_or_create_document(
                    session, doc_id, source_type=source_type
                )

                # Delete existing chunks for this document
                await session.execute(
                    delete(DocumentChunk).where(
                        DocumentChunk.document_id == document.id
                    )
                )

                # Add new chunks
                for record in doc_records:
                    chunk_index = record.metadata.get("chunk_index", 0)
                    chunk = DocumentChunk(
                        document_id=document.id,
                        workspace_id=self._workspace_id,
                        chunk_index=chunk_index,
                        text=record.text,
                        embedding=record.embedding,
                        chunk_metadata=record.metadata,
                    )
                    session.add(chunk)

            await session.commit()

    async def async_search(
        self, query_embedding: List[float], k: int = 5,
        source_type: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for the top-k most similar records using pgvector cosine similarity.

        Results are scoped to this store's workspace. When ``source_type`` is
        set, results are additionally restricted to documents of that category
        so requirement drafting never retrieves RFP source passages.
        """
        if k <= 0:
            return []

        self._validate_embedding_dimension(query_embedding)

        async with self._session_factory() as session:
            # Use pgvector cosine distance operator (<=>)
            # Cosine similarity = 1 - cosine_distance
            stmt = (
                select(
                    DocumentChunk,
                    (
                        1
                        - DocumentChunk.embedding.cosine_distance(query_embedding)
                    ).label("similarity"),
                )
                .join(Document, DocumentChunk.document_id == Document.id)
                .where(Document.workspace_id == self._workspace_id)
            )
            if source_type is not None:
                stmt = stmt.where(Document.source_type == source_type)
            stmt = (
                stmt.order_by(
                    DocumentChunk.embedding.cosine_distance(query_embedding)
                )
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

    async def async_delete_by_metadata(self, metadata_filter: dict[str, Any]) -> int:
        """Delete records matching a metadata filter asynchronously.

        Deletions are scoped to this store's workspace.
        """
        if not metadata_filter:
            return 0

        doc_id = metadata_filter.get("doc_id")

        async with self._session_factory() as session:
            if doc_id:
                # Fast path: delete by doc_id within this workspace.
                stmt = select(Document).where(
                    Document.workspace_id == self._workspace_id,
                    Document.doc_id == doc_id,
                )
                result = await session.execute(stmt)
                document = result.scalar_one_or_none()

                if document is None:
                    return 0

                delete_stmt = delete(DocumentChunk).where(
                    DocumentChunk.document_id == document.id
                )
                result = await session.execute(delete_stmt)
                deleted_count = result.rowcount

                await session.commit()
                return deleted_count
            else:
                # Slow path: scan all chunks for the workspace and filter by metadata.
                stmt = (
                    select(DocumentChunk)
                    .join(Document, DocumentChunk.document_id == Document.id)
                    .where(Document.workspace_id == self._workspace_id)
                )
                result = await session.execute(stmt)
                chunks = result.scalars().all()

                deleted = 0
                for chunk in chunks:
                    if all(
                        chunk.chunk_metadata.get(k) == v
                        for k, v in metadata_filter.items()
                    ):
                        await session.delete(chunk)
                        deleted += 1

                await session.commit()
                return deleted

    async def async_clear(self) -> None:
        """Remove all records for the current workspace asynchronously."""
        async with self._session_factory() as session:
            # Delete all chunks for workspace's documents
            stmt = delete(DocumentChunk).where(
                DocumentChunk.document_id.in_(
                    select(Document.id).where(
                        Document.workspace_id == self._workspace_id
                    )
                )
            )
            await session.execute(stmt)

            # Optionally delete documents too
            doc_delete_stmt = delete(Document).where(
                Document.workspace_id == self._workspace_id
            )
            await session.execute(doc_delete_stmt)

            await session.commit()

    async def async_count(self) -> int:
        """Return the number of records for the current workspace asynchronously."""
        async with self._session_factory() as session:
            stmt = select(func.count(DocumentChunk.id)).where(
                DocumentChunk.document_id.in_(
                    select(Document.id).where(
                        Document.workspace_id == self._workspace_id
                    )
                )
            )
            result = await session.execute(stmt)
            return result.scalar() or 0

    # --- Document query operations (list / get / get-chunks) ---

    async def async_list_documents(
        self, workspace_id: Optional[str] = None
    ) -> List[dict]:
        """List ingested documents in this workspace."""
        target_workspace = self._workspace_id
        if workspace_id is not None:
            target_workspace = (
                uuid.uuid5(uuid.NAMESPACE_DNS, str(workspace_id))
                if not isinstance(workspace_id, uuid.UUID)
                else workspace_id
            )

        async with self._session_factory() as session:
            stmt = (
                select(DocumentChunk)
                .join(Document, DocumentChunk.document_id == Document.id)
                .where(Document.workspace_id == target_workspace)
                .order_by(DocumentChunk.chunk_index)
            )
            result = await session.execute(stmt)
            chunks = result.scalars().all()

        return _summarize_chunks(chunks)

    async def async_get_document(
        self, doc_id: str, workspace_id: Optional[str] = None
    ) -> Optional[dict]:
        """Get metadata for a single document within this workspace."""
        documents = await self.async_list_documents(workspace_id=workspace_id)
        for document in documents:
            if document["doc_id"] == doc_id:
                return document
        return None

    async def async_get_document_chunks(
        self, doc_id: str, workspace_id: Optional[str] = None
    ) -> List[dict]:
        """Get all chunks for a document with their source references."""
        target_workspace = self._workspace_id
        if workspace_id is not None:
            target_workspace = (
                uuid.uuid5(uuid.NAMESPACE_DNS, str(workspace_id))
                if not isinstance(workspace_id, uuid.UUID)
                else workspace_id
            )

        async with self._session_factory() as session:
            stmt = (
                select(DocumentChunk)
                .join(Document, DocumentChunk.document_id == Document.id)
                .where(
                    Document.workspace_id == target_workspace,
                    Document.doc_id == doc_id,
                )
                .order_by(DocumentChunk.chunk_index)
            )
            result = await session.execute(stmt)
            chunks = result.scalars().all()

        return [
            {
                "chunk_id": str(chunk.id),
                "document_id": doc_id,
                "doc_id": doc_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "page_number": chunk.chunk_metadata.get("page_number"),
                "character_start": chunk.chunk_metadata.get(
                    "character_start"
                ),
                "character_end": chunk.chunk_metadata.get("character_end"),
                "section_path": chunk.chunk_metadata.get("section_path"),
                "source_filename": chunk.chunk_metadata.get(
                    "source_filename"
                ),
                "workspace_id": chunk.chunk_metadata.get("workspace_id"),
                "metadata": chunk.chunk_metadata,
            }
            for chunk in chunks
        ]

    # --- Sync methods (fallback for CLI/non-async contexts) ---

    def add(self, records: List[VectorRecord]) -> None:
        """Add records to the store (sync fallback)."""
        _run_async(self.async_add(records))

    def search(
        self, query_embedding: List[float], k: int = 5,
        source_type: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for the top-k most similar records (sync fallback)."""
        return _run_async(
            self.async_search(query_embedding, k, source_type=source_type)
        )

    def delete_by_metadata(self, metadata_filter: dict[str, Any]) -> int:
        """Delete records matching a metadata filter (sync fallback)."""
        return _run_async(self.async_delete_by_metadata(metadata_filter))

    def clear(self) -> None:
        """Remove all records for the current workspace (sync fallback)."""
        _run_async(self.async_clear())

    def count(self) -> int:
        """Return the number of records for the current workspace (sync fallback)."""
        return _run_async(self.async_count())

    def list_documents(self, workspace_id: Optional[str] = None) -> List[dict]:
        """List ingested documents in this workspace (sync fallback)."""
        return _run_async(self.async_list_documents(workspace_id=workspace_id))

    def get_document(
        self, doc_id: str, workspace_id: Optional[str] = None
    ) -> Optional[dict]:
        """Get metadata for a single document (sync fallback)."""
        return _run_async(self.async_get_document(doc_id, workspace_id=workspace_id))

    def get_document_chunks(
        self, doc_id: str, workspace_id: Optional[str] = None
    ) -> List[dict]:
        """Get all chunks for a document with source references (sync fallback)."""
        return _run_async(
            self.async_get_document_chunks(doc_id, workspace_id=workspace_id)
        )

    @property
    def dimension(self) -> Optional[int]:
        """Return the embedding dimension of stored vectors."""
        return EXPECTED_DIMENSION
