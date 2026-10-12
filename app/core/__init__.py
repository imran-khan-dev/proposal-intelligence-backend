"""Core application infrastructure."""

from app.core.singletons import (
    get_ai_provider,
    get_embedding_provider,
    get_vector_store,
    initialize_singletons,
    shutdown_singletons,
)

__all__ = [
    "get_ai_provider",
    "get_embedding_provider",
    "get_vector_store",
    "initialize_singletons",
    "shutdown_singletons",
]