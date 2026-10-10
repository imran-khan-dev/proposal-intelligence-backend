"""Tests for embedding providers."""

import os
from unittest.mock import MagicMock, patch

import pytest

from app.ai.embedding_factory import get_embedding_provider
from app.ai.embeddings_local import LocalEmbeddingProvider
from app.ai.embeddings_api import ApiEmbeddingProvider
from app.ai.embeddings import EmbeddingProvider


class TestEmbeddingProviderSelection:
    """Test embedding provider selection via environment variable."""

    def test_local_provider_default(self):
        """EMBEDDING_PROVIDER not set defaults to local."""
        with patch.dict(os.environ, {}, clear=True):
            provider = get_embedding_provider()
            assert isinstance(provider, LocalEmbeddingProvider)

    def test_local_provider_explicit(self):
        """EMBEDDING_PROVIDER=local selects LocalEmbeddingProvider."""
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local"}, clear=True):
            provider = get_embedding_provider()
            assert isinstance(provider, LocalEmbeddingProvider)

    def test_api_provider_explicit(self):
        """EMBEDDING_PROVIDER=api selects ApiEmbeddingProvider."""
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "api", "EMBEDDING_API_KEY": "test-key"}, clear=True):
            provider = get_embedding_provider()
            assert isinstance(provider, ApiEmbeddingProvider)

    def test_unsupported_provider_raises(self):
        """Unknown EMBEDDING_PROVIDER raises RuntimeError."""
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "unknown"}, clear=True):
            with pytest.raises(RuntimeError, match="Invalid EMBEDDING_PROVIDER"):
                get_embedding_provider()

    def test_local_provider_custom_model(self):
        """EMBEDDING_MODEL is passed to local provider."""
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local", "EMBEDDING_MODEL": "custom-model"}, clear=True):
            provider = get_embedding_provider()
            assert isinstance(provider, LocalEmbeddingProvider)
            assert provider._model_name == "custom-model"

    def test_api_provider_reads_env_config(self):
        """API provider reads EMBEDDING_MODEL, EMBEDDING_BASE_URL from env."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_PROVIDER": "api",
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_MODEL": "custom-embedding-model",
                "EMBEDDING_BASE_URL": "https://custom.api/v1",
            },
            clear=True,
        ):
            provider = get_embedding_provider()
            assert isinstance(provider, ApiEmbeddingProvider)
            assert provider._model == "custom-embedding-model"
            assert provider._base_url == "https://custom.api/v1"


class TestLocalEmbeddingProvider:
    """Tests for LocalEmbeddingProvider."""

    def test_empty_input_returns_empty_list(self):
        """Empty input returns empty list without loading model."""
        provider = LocalEmbeddingProvider()
        # Mock the model loading to verify it's not called
        with patch.object(provider, "_load_model") as mock_load:
            result = provider.embed([])
            assert result == []
            mock_load.assert_not_called()

    def test_whitespace_input_returns_embeddings(self):
        """Whitespace-only strings are embedded normally."""
        provider = LocalEmbeddingProvider()
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384])
        provider._model = mock_model
        provider._dimension = 384

        result = provider.embed([" "])
        assert len(result) == 1
        assert len(result[0]) == 384
        mock_model.encode.assert_called_once_with(
            [" "], normalize_embeddings=True, convert_to_numpy=True
        )

    def test_dimension_property(self):
        """dimension property returns correct value after model load."""
        provider = LocalEmbeddingProvider()
        with patch.object(provider, "_load_model") as mock_load:
            # Simulate model load setting dimension
            def set_dimension():
                provider._dimension = 384
            mock_load.side_effect = set_dimension

            dim = provider.dimension
            assert dim == 384

    def test_embed_preserves_order(self):
        """Embeddings returned in same order as input texts."""
        provider = LocalEmbeddingProvider()
        texts = ["text A", "text B", "text C"]
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384,
        ])
        provider._model = mock_model
        provider._dimension = 384

        result = provider.embed(texts)
        assert len(result) == 3
        # Verify order by checking distinct values
        assert result[0][0] == 0.1
        assert result[1][0] == 0.2
        assert result[2][0] == 0.3
        mock_model.encode.assert_called_once_with(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )


class TestApiEmbeddingProvider:
    """Tests for ApiEmbeddingProvider."""

    def test_empty_input_returns_empty_list_no_api_call(self):
        """Empty input returns empty list without API call."""
        provider = ApiEmbeddingProvider(api_key="test-key")
        result = provider.embed([])
        assert result == []
        # Client should not be initialized
        assert provider._client is None

    def test_missing_api_key_raises(self):
        """Missing API key raises RuntimeError on first embed call."""
        provider = ApiEmbeddingProvider(api_key=None)
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="API key required"):
                provider.embed(["test"])

    def test_embed_preserves_order(self):
        """Embeddings returned in same order as input texts."""
        provider = ApiEmbeddingProvider(api_key="test-key")
        texts = ["text A", "text B", "text C"]

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
            MagicMock(embedding=[0.3] * 1536),
        ]
        mock_client.embeddings.create.return_value = mock_response
        provider._client = mock_client

        result = provider.embed(texts)

        assert len(result) == 3
        assert result[0][0] == 0.1
        assert result[1][0] == 0.2
        assert result[2][0] == 0.3

    def test_embed_caches_dimension(self):
        """Dimension is cached after first successful embed call."""
        provider = ApiEmbeddingProvider(api_key="test-key")
        assert provider._dimension is None

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        provider._client = mock_client

        provider.embed(["test"])
        assert provider._dimension == 1536

    def test_uses_embedding_api_key_env_var(self):
        """EMBEDDING_API_KEY is used when set."""
        with patch.dict(os.environ, {"EMBEDDING_API_KEY": "embedding-key"}, clear=True):
            provider = ApiEmbeddingProvider()
            assert provider._api_key == "embedding-key"

    def test_falls_back_to_openai_api_key(self):
        """Falls back to OPENAI_API_KEY when EMBEDDING_API_KEY not set."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            provider = ApiEmbeddingProvider()
            assert provider._api_key == "openai-key"

    def test_explicit_api_key_priority(self):
        """Explicit api_key argument takes priority over env vars."""
        with patch.dict(os.environ, {"EMBEDDING_API_KEY": "env-key", "OPENAI_API_KEY": "openai-key"}, clear=True):
            provider = ApiEmbeddingProvider(api_key="explicit-key")
            assert provider._api_key == "explicit-key"

    def test_lazy_client_initialization(self):
        """Client is initialized only on first embed call."""
        provider = ApiEmbeddingProvider(api_key="test-key")
        assert provider._client is None

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch("app.ai.embeddings_api.OpenAI", return_value=mock_client) as mock_openai:
            provider.embed(["test"])
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                timeout=60.0,
            )

    def test_custom_base_url(self):
        """Custom base URL is used when provided."""
        provider = ApiEmbeddingProvider(api_key="test-key", base_url="https://custom.api/v1")
        assert provider._base_url == "https://custom.api/v1"

    def test_dimension_default_before_first_call(self):
        """dimension returns default 1536 before first embed call."""
        provider = ApiEmbeddingProvider(api_key="test-key")
        assert provider.dimension == 1536


class TestEmbeddingProviderInterface:
    """Tests verifying the EmbeddingProvider interface contract."""

    def test_local_implements_interface(self):
        """LocalEmbeddingProvider implements EmbeddingProvider."""
        provider = LocalEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_api_implements_interface(self):
        """ApiEmbeddingProvider implements EmbeddingProvider."""
        provider = ApiEmbeddingProvider(api_key="test-key")
        assert isinstance(provider, EmbeddingProvider)

    def test_embed_signature(self):
        """embed method accepts List[str] and returns List[List[float]]."""
        from typing import get_type_hints
        import typing

        hints = get_type_hints(EmbeddingProvider.embed)
        # Handle both list[str] (Python 3.9+) and typing.List[str]
        assert hints["texts"] in (list[str], typing.List[str])
        assert hints["return"] in (list[list[float]], typing.List[typing.List[float]])