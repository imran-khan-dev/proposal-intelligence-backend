"""AI provider package."""

from app.ai.embeddings import EmbeddingProvider
from app.ai.embedding_factory import get_embedding_provider
from app.ai.embeddings_local import LocalEmbeddingProvider
from app.ai.embeddings_api import ApiEmbeddingProvider
from app.ai.provider import AIProvider
from app.ai.factory import get_provider

__all__ = [
    "AIProvider",
    "get_provider",
    "EmbeddingProvider",
    "get_embedding_provider",
    "LocalEmbeddingProvider",
    "ApiEmbeddingProvider",
]