\"\"\"Singleton management for expensive resources.

Provides application-scoped instances for AI providers, embedding providers,
and vector stores. Instances are created once at startup and reused across
requests. Request-scoped mutable state (AgentService, conversation messages,
tool-call records) remains request-scoped.
\"\"\"

import os
import threading
from typing import Optional

from app.ai.provider import AIProvider
from app.ai.embeddings import EmbeddingProvider
from app.rag.store import VectorStore
from app.ai.kilo import KiloProvider
from app.ai.ollama import OllamaProvider
from app.ai.openai import OpenAIProvider
from app.ai.google import GoogleProvider
from app.ai.groq import GroqProvider
from app.ai.embeddings_local import LocalEmbeddingProvider
from app.ai.embeddings_api import ApiEmbeddingProvider
from app.rag.store import InMemoryVectorStore, PersistentVectorStore


_ai_provider: Optional[AIProvider] = None
_embedding_provider: Optional[EmbeddingProvider] = None
_vector_store: Optional[VectorStore] = None

_lock = threading.Lock()
_initialized = False


def _create_ai_provider() -> AIProvider:
    \"\"\"Create a new AI provider instance based on environment configuration.\"\"\"
    # Read and validate AI_TIMEOUT
    _ai_timeout_env = os.environ.get(\"AI_TIMEOUT\")
    if _ai_timeout_env is not None:
        try:
            _ai_timeout = float(_ai_timeout_env)
            if _ai_timeout <= 0:
                raise ValueError
        except ValueError:
            raise RuntimeError(\"AI_TIMEOUT must be a positive float\")
    else:
        _ai_timeout = None

    # Read and validate AI_MAX_RETRIES
    _ai_max_retries_env = os.environ.get(\"AI_MAX_RETRIES\")
    if _ai_max_retries_env is not None:
        try:
            _ai_max_retries = int(_ai_max_retries_env)
            if _ai_max_retries < 0:
                raise ValueError
        except ValueError:
            raise RuntimeError(\"AI_MAX_RETRIES must be a non-negative integer\")
    else:
        _ai_max_retries = None

    provider_name = (
        os.environ.get(\"AI_PROVIDER\", \"openai\")
        .strip()
        .strip('\"')
        .strip(\"'\")
        .lower()
    )

    if provider_name == \"ollama\":
        base_url = os.environ.get(
            \"OLLAMA_BASE_URL\", \"http://localhost:11434\"
        )
        model = os.environ.get(\"OLLAMA_MODEL\")
        if not model:
            raise RuntimeError(\"OLLAMA_MODEL environment variable is not set\")
        # For Ollama, we preserve the current hardcoded timeout of 120.0 when AI_TIMEOUT is not set
        client_timeout = _ai_timeout
        if client_timeout is None:
            client_timeout = 120.0  # Preserve existing behavior
        client_kwargs = {
            \"api_key\": \"ollama\",  # Ollama uses a fixed API key
            \"base_url\": base_url,
            \"timeout\": client_timeout,
        }
        if _ai_max_retries is not None:
            client_kwargs[\"max_retries\"] = _ai_max_retries
        return OllamaProvider(
            base_url=base_url,
            model=model,
            timeout=client_kwargs[\"timeout\"],
            max_retries=client_kwargs.get(\"max_retries\"),
            max_tokens=None  # Singleton doesn't have task-specific max_tokens
        )

    if provider_name == \"openai\":
        api_key = os.environ.get(\"OPENAI_API_KEY\")
        if not api_key:
            raise RuntimeError(\"OPENAI_API_KEY environment variable is not set\")
        model = os.environ.get(\"OPENAI_MODEL\", \"gpt-4o-mini\")
        # For OpenAI, we don't have a hardcoded timeout default
        client_kwargs = {
            \"api_key\": api_key,
        }
        if _ai_timeout is not None:
            client_kwargs[\"timeout\"] = _ai_timeout
        if _ai_max_retries is not None:
            client_kwargs[\"max_retries\"] = _ai_max_retries
        return OpenAIProvider(
            api_key=api_key,
            model=model,
            timeout=client_kwargs.get(\"timeout\"),
            max_retries=client_kwargs.get(\"max_retries\"),
            max_tokens=None  # Singleton doesn't have task-specific max_tokens
        )

    if provider_name == \"kilo\":
        api_key = os.environ.get(\"KILO_API_KEY\")
        if not api_key:
            raise RuntimeError(\"KILO_API_KEY environment variable is not set\")
        base_url = os.environ.get(
            \"KILO_BASE_URL\", \"https://api.kilo.ai/api/gateway\"
        )
        model = os.environ.get(\"KILO_MODEL\")
        if not model:
            raise RuntimeError(\"KILO_MODEL environment variable is not set\")
        # For Kilo, we preserve the current hardcoded timeout of 120.0 when AI_TIMEOUT is not set
        client_timeout = _ai_timeout
        if client_timeout is None:
            client_timeout = 120.0  # Preserve existing behavior
        client_kwargs = {
            \"api_key\": api_key,
            \"base_url\": base_url,
            \"timeout\": client_timeout,
        }
        if _ai_max_retries is not None:
            client_kwargs[\"max_retries\"] = _ai_max_retries
        return KiloProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=client_kwargs[\"timeout\"],
            max_retries=client_kwargs.get(\"max_retries\"),
            max_tokens=None  # Singleton doesn't have task-specific max_tokens
        )

    if provider_name == \"google\":
        api_key = os.environ.get(\"GOOGLE_API_KEY\")
        if not api_key:
            raise RuntimeError(\"GOOGLE_API_KEY environment variable is not set\")
        model = os.environ.get(\"GOOGLE_MODEL\", \"gemini-3.8-flash\")
        # For Google, we pass the timeout and max_retries but note they may not be used by the SDK
        return GoogleProvider(
            api_key=api_key,
            model=model,
            timeout=_ai_timeout,
            max_retries=_ai_max_retries,
            max_tokens=None  # Singleton doesn't have task-specific max_tokens
        )

    if provider_name == \"groq\":
        api_key = os.environ.get(\"GROQ_API_KEY\")
        if not api_key:
            raise RuntimeError(\"GROQ_API_KEY environment variable is not set\")
        base_url = os.environ.get(
            \"GROQ_BASE_URL\", \"https://api.groq.com/openai/v1\"
        )
        model = os.environ.get(\"GROQ_MODEL\", \"openai/gpt-oss-20b\")
        # For Groq, we preserve the current hardcoded timeout of 120.0 when AI_TIMEOUT is not set
        client_timeout = _ai_timeout
        if client_timeout is None:
            client_timeout = 120.0  # Preserve existing behavior
        client_kwargs = {
            \"api_key\": api_key,
            \"base_url\": base_url,
            \"timeout\": client_timeout,
        }
        if _ai_max_retries is not None:
            client_kwargs[\"max_retries\"] = _ai_max_retries
        return GroqProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=client_kwargs[\"timeout\"],
            max_retries=client_kwargs.get(\"max_retries\"),
            max_tokens=None  # Singleton doesn't have task-specific max_tokens
        )

    raise RuntimeError(
        f\"Invalid AI_PROVIDER: '{provider_name}'. \"
        \"Must be 'ollama', 'openai', 'kilo', 'google', or 'groq'.\"
    )


def _create_embedding_provider() -> EmbeddingProvider:
    \"\"\"Create a new embedding provider instance based on environment configuration.\"\"\"
    provider_name = (
        os.environ.get(\"EMBEDDING_PROVIDER\", \"local\")
        .strip()
        .strip('\"')
        .strip(\"'\")
        .lower()
    )

    if provider_name == \"local\":
        model = os.environ.get(\"EMBEDDING_MODEL\")
        return LocalEmbeddingProvider(model_name=model)

    if provider_name == \"api\":
        return ApiEmbeddingProvider()

    raise RuntimeError(
        f\"Invalid EMBEDDING_PROVIDER: '{provider_name}'. \"
        \"Must be 'local' or 'api'.\"
    )


def _create_vector_store() -> VectorStore:
    \"\"\"Create a new vector store instance based on environment configuration.\"\"\"
    store_type = (
        os.environ.get(\"VECTOR_STORE_TYPE\", \"memory\")
        .strip()
        .strip('\"')
        .strip(\"'\")
        .lower()
    )

    if store_type == \"memory\":
        return InMemoryVectorStore()

    if store_type == \"persistent\":
        path = os.environ.get(\"VECTOR_STORE_PATH\")
        if not path:
            # Default to a local data directory
            path = os.path.join(os.getcwd(), \"data\", \"vector_store.jsonl\")
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return PersistentVectorStore(path)

    if store_type == \"postgresql\":
        # Lazy import to avoid circular dependency
        from app.rag.stores.postgres_vector_store import PostgreSQLVectorStore
        return PostgreSQLVectorStore()

    raise RuntimeError(
        f\"Invalid VECTOR_STORE_TYPE: '{store_type}'. \"
        \"Must be 'memory', 'persistent', or 'postgresql'.\"
    )


def initialize_singletons() -> None:
    \"\"\"Initialize all singleton resources.

    Called once at application startup via FastAPI lifespan.
    Thread-safe: can be called concurrently but only first call initializes.
    \"\"\"
    global _ai_provider, _embedding_provider, _vector_store, _initialized

    with _lock:
        if _initialized:
            return

        _ai_provider = _create_ai_provider()
        _embedding_provider = _create_embedding_provider()
        _vector_store = _create_vector_store()
        _initialized = True


def shutdown_singletons() -> None:
    \"\"\"Shutdown singleton resources.

    Called once at application shutdown via FastAPI lifespan.
    Currently a no-op for most resources since underlying clients
    (OpenAI sync client, sentence-transformers) don't expose explicit close.
    Provided for future extensibility (e.g., async clients, DB pools).
    \"\"\"
    global _ai_provider, _embedding_provider, _vector_store, _initialized

    with _lock:
        # No explicit close() on current implementations.
        # OpenAI sync client relies on __del__; sentence-transformers
        # model is just a Python object. If we switch to AsyncOpenAI
        # or add DB pools, close them here.
        _ai_provider = None
        _embedding_provider = None
        _vector_store = None
        _initialized = False


def get_ai_provider() -> AIProvider:
    \"\"\"Return the application-scoped AI provider singleton.

    Raises:
        RuntimeError: If called before initialize_singletons().
    \"\"\"
    if _ai_provider is None:
        raise RuntimeError(
            \"AI provider not initialized. Call initialize_singletons() at startup.\"
        )
    return _ai_provider


def get_embedding_provider() -> EmbeddingProvider:
    \"\"\"Return the application-scoped embedding provider singleton.

    Raises:
        RuntimeError: If called before initialize_singletons().
    \"\"\"
    if _embedding_provider is None:
        raise RuntimeError(
            \"Embedding provider not initialized. Call initialize_singletons() at startup.\"
        )
    return _embedding_provider


def get_vector_store() -> VectorStore:
    \"\"\"Return the application-scoped vector store singleton.

    Raises:
        RuntimeError: If called before initialize_singletons().
    \"\"\"
    if _vector_store is None:
        raise RuntimeError(
            \"Vector store not initialized. Call initialize_singletons() at startup.\"
        )
    return _vector_store
