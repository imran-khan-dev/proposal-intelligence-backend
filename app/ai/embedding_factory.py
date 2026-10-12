"""Factory for creating embedding providers based on environment configuration."""

import os

from app.ai.embeddings import EmbeddingProvider
from app.ai.embeddings_local import LocalEmbeddingProvider
from app.ai.embeddings_api import ApiEmbeddingProvider
from app.core.singletons import (
    get_embedding_provider as _get_embedding_provider_singleton,
    _create_embedding_provider,
)


def get_embedding_provider() -> EmbeddingProvider:
    """Return the embedding provider singleton if initialized, otherwise create a new one.

    In the FastAPI application, the singleton is initialized at startup via lifespan.
    In CLI or other contexts, a new instance is created each call (backward compatible).
    """
    try:
        return _get_embedding_provider_singleton()
    except RuntimeError:
        # Singleton not initialized (e.g., CLI context), create new instance
        return _create_embedding_provider()