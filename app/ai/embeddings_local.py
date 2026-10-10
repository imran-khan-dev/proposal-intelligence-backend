"""Local embedding provider using sentence-transformers."""

from typing import List

from app.ai.embeddings import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using a local sentence-transformers model.

    Loads the model once on first use and reuses it for subsequent calls.
    Normalizes embeddings to unit length.
    """

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._dimension = None

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            # Determine dimension by encoding a test string
            test_embedding = self._model.encode(["test"], normalize_embeddings=True)
            self._dimension = len(test_embedding[0])

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        self._load_model()
        return self._dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors, one per input text, in the same order.
            Returns empty list for empty input.
        """
        if not texts:
            return []

        self._load_model()

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()