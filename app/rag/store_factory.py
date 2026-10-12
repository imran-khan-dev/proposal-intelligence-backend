"""Factory for creating vector stores based on environment configuration."""

import os

from app.rag.store import InMemoryVectorStore, PersistentVectorStore, VectorStore
from app.core.singletons import (
    get_vector_store as _get_vector_store_singleton,
    _create_vector_store,
)


def get_vector_store() -> VectorStore:
    """Return the vector store singleton if initialized, otherwise create a new one.

    In the FastAPI application, the singleton is initialized at startup via lifespan.
    In CLI or other contexts, a new instance is created each call (backward compatible).

    Note: For persistent store in CLI context, a new instance is created each call,
    which reloads the JSONL file. This is acceptable for CLI usage but means
    concurrent CLI + API access to persistent store may have consistency issues.
    """
    try:
        return _get_vector_store_singleton()
    except RuntimeError:
        # Singleton not initialized (e.g., CLI context), create new instance
        return _create_vector_store()