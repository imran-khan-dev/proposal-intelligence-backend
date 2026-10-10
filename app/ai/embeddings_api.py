"""API embedding provider using OpenAI-compatible embeddings endpoint."""

import os
from typing import List

from openai import OpenAI

from app.ai.embeddings import EmbeddingProvider


class ApiEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using an OpenAI-compatible API.

    Initializes the client lazily so local mode does not require an API key.
    Uses OpenAI's text-embedding-3-small as the default model.
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        # Priority: explicit arg > EMBEDDING_API_KEY > OPENAI_API_KEY
        self._api_key = api_key or os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self._model = model or os.environ.get("EMBEDDING_MODEL") or self.DEFAULT_MODEL
        self._base_url = base_url or os.environ.get("EMBEDDING_BASE_URL") or self.DEFAULT_BASE_URL
        self._client = None
        self._dimension = None

    def _get_client(self) -> OpenAI:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "API key required for API embedding provider. "
                    "Set EMBEDDING_API_KEY or OPENAI_API_KEY."
                )
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=60.0,
            )
        return self._client

    @property
    def dimension(self) -> int:
        """Return the embedding dimension.

        Note: This requires an API call to determine. For text-embedding-3-small
        the dimension is 1536. We cache it after the first successful call.
        """
        if self._dimension is None:
            # We can't know the dimension without an API call.
            # For text-embedding-3-small it's 1536.
            # We'll determine it on first embed() call.
            pass
        return self._dimension or 1536  # Default for text-embedding-3-small

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts via API.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors, one per input text, in the same order.
            Returns empty list for empty input without making an API call.
        """
        if not texts:
            return []

        client = self._get_client()

        response = client.embeddings.create(
            model=self._model,
            input=texts,
        )

        # Extract embeddings in order
        embeddings = [data.embedding for data in response.data]

        # Cache dimension on first successful call
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])

        return embeddings