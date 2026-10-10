"""Factory for creating embedding providers based on environment configuration."""

import os

from app.ai.embeddings import EmbeddingProvider
from app.ai.embeddings_local import LocalEmbeddingProvider
from app.ai.embeddings_api import ApiEmbeddingProvider


def get_embedding_provider() -> EmbeddingProvider:
    """Select and instantiate an embedding provider based on environment configuration.

    Supported providers:
    - local: Uses sentence-transformers locally (default)
    - api: Uses OpenAI-compatible embeddings API

    Environment variables:
    - EMBEDDING_PROVIDER: "local" or "api" (default: "local")
    - EMBEDDING_MODEL: Model name (provider-specific default if not set)
    - EMBEDDING_API_KEY: API key for API provider (falls back to OPENAI_API_KEY)
    - EMBEDDING_BASE_URL: Base URL for API provider (default: https://api.openai.com/v1)

    Returns:
        An EmbeddingProvider instance.

    Raises:
        RuntimeError: If provider is unknown or API credentials are missing.
    """
    provider_name = (
        os.environ.get("EMBEDDING_PROVIDER", "local")
        .strip()
        .strip('"')
        .strip("'")
        .lower()
    )

    if provider_name == "local":
        model = os.environ.get("EMBEDDING_MODEL")
        return LocalEmbeddingProvider(model_name=model)

    if provider_name == "api":
        return ApiEmbeddingProvider()

    raise RuntimeError(
        f"Invalid EMBEDDING_PROVIDER: '{provider_name}'. "
        "Must be 'local' or 'api'."
    )