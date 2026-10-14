import os

from app.ai.google import GoogleProvider
from app.ai.groq import GroqProvider
from app.ai.kilo import KiloProvider
from app.ai.ollama import OllamaProvider
from app.ai.openai import OpenAIProvider
from app.ai.provider import AIProvider
from app.core.singletons import get_ai_provider as _get_ai_provider_singleton


def _create_ai_provider_from_env(prefix: str = "") -> AIProvider:
    """Create a new AI provider instance from environment variables.
    
    Args:
        prefix: If non-empty, use env vars like PREFIX_PROVIDER, PREFIX_MODEL, etc.
                If empty, fall back to AI_PROVIDER, OPENAI_API_KEY, etc.
    """
    if prefix:
        # Convert prefix to uppercase and underscore format for env vars
        env_prefix = prefix.upper()
        provider_name = (
            os.environ.get(f"{env_prefix}_AI_PROVIDER", "")
            .strip()
            .strip('"')
            .strip("'")
            .lower()
        )
        # Fall back to common AI_PROVIDER if task-specific one is not set
        if not provider_name:
            provider_name = (
                os.environ.get("AI_PROVIDER", "openai")
                .strip()
                .strip('"')
                .strip("'")
                .lower()
            )
    else:
        # No prefix, use common AI_PROVIDER
        provider_name = (
            os.environ.get("AI_PROVIDER", "openai")
            .strip()
            .strip('"')
            .strip("'")
            .lower()
        )

    if provider_name == "ollama":
        if prefix:
            base_url = os.environ.get(
                f"{env_prefix}_OLLAMA_BASE_URL", "http://localhost:11434"
            )
            model = os.environ.get(f"{env_prefix}_OLLAMA_MODEL")
            # Fall back to common OLLAMA_MODEL if task-specific one is not set
            if not model:
                model = os.environ.get("OLLAMA_MODEL")
        else:
            base_url = os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            )
            model = os.environ.get("OLLAMA_MODEL")
        
        if not model:
            raise RuntimeError("OLLAMA_MODEL environment variable is not set")
        return OllamaProvider(base_url=base_url, model=model)

    if provider_name == "openai":
        if prefix:
            api_key = os.environ.get(f"{env_prefix}_OPENAI_API_KEY")
            # Fall back to common OPENAI_API_KEY if task-specific one is not set
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY")
            model = os.environ.get(f"{env_prefix}_OPENAI_MODEL")
            # Fall back to common OPENAI_MODEL if task-specific one is not set
            if not model:
                model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        return OpenAIProvider(api_key=api_key, model=model)

    if provider_name == "kilo":
        if prefix:
            api_key = os.environ.get(f"{env_prefix}_KILO_API_KEY")
            # Fall back to common KILO_API_KEY if task-specific one is not set
            if not api_key:
                api_key = os.environ.get("KILO_API_KEY")
            base_url = os.environ.get(
                f"{env_prefix}_KILO_BASE_URL", "https://api.kilo.ai/api/gateway"
            )
            # Fall back to common KILO_BASE_URL if task-specific one is not set
            model = os.environ.get(f"{env_prefix}_KILO_MODEL")
            # Fall back to common KILO_MODEL if task-specific one is not set
            if not model:
                model = os.environ.get("KILO_MODEL")
        else:
            api_key = os.environ.get("KILO_API_KEY")
            base_url = os.environ.get(
                "KILO_BASE_URL", "https://api.kilo.ai/api/gateway"
            )
            model = os.environ.get("KILO_MODEL")
        
        if not api_key:
            raise RuntimeError("KILO_API_KEY environment variable is not set")
        if not model:
            raise RuntimeError("KILO_MODEL environment variable is not set")
        return KiloProvider(api_key=api_key, base_url=base_url, model=model)

    if provider_name == "google":
        if prefix:
            api_key = os.environ.get(f"{env_prefix}_GOOGLE_API_KEY")
            # Fall back to common GOOGLE_API_KEY if task-specific one is not set
            if not api_key:
                api_key = os.environ.get("GOOGLE_API_KEY")
            model = os.environ.get(f"{env_prefix}_GOOGLE_MODEL")
            # Fall back to common GOOGLE_MODEL if task-specific one is not set
            if not model:
                model = os.environ.get("GOOGLE_MODEL", "gemini-3.8-flash")
        else:
            api_key = os.environ.get("GOOGLE_API_KEY")
            model = os.environ.get("GOOGLE_MODEL", "gemini-3.8-flash")
        
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set")
        return GoogleProvider(api_key=api_key, model=model)

    if provider_name == "groq":
        if prefix:
            api_key = os.environ.get(f"{env_prefix}_GROQ_API_KEY")
            # Fall back to common GROQ_API_KEY if task-specific one is not set
            if not api_key:
                api_key = os.environ.get("GROQ_API_KEY")
            base_url = os.environ.get(
                f"{env_prefix}_GROQ_BASE_URL", "https://api.groq.com/openai/v1"
            )
            # Fall back to common GROQ_BASE_URL if task-specific one is not set
            model = os.environ.get(f"{env_prefix}_GROQ_MODEL")
            # Fall back to common GROQ_MODEL if task-specific one is not set
            if not model:
                model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        else:
            api_key = os.environ.get("GROQ_API_KEY")
            base_url = os.environ.get(
                "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
            )
            model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        return GroqProvider(api_key=api_key, base_url=base_url, model=model)

    raise RuntimeError(
        f"Invalid AI_PROVIDER: '{provider_name}'. "
        "Must be 'ollama', 'openai', 'kilo', 'google', or 'groq'."
    )


def _create_ai_provider() -> AIProvider:
    """Create a new AI provider instance (not singleton)."""
    return _create_ai_provider_from_env("")


def get_provider() -> AIProvider:
    """Return the AI provider singleton if initialized, otherwise create a new one.

    In the FastAPI application, the singleton is initialized at startup via lifespan.
    In CLI or other contexts, a new instance is created each call (backward compatible).
    """
    try:
        return _get_ai_provider_singleton()
    except RuntimeError:
        # Singleton not initialized (e.g., CLI context), create new instance
        return _create_ai_provider()


def get_rfp_provider() -> AIProvider:
    """Get AI provider for RFP analysis, checking RFP_AI_PROVIDER first."""
    return _create_ai_provider_from_env("RFP")


def get_agent_provider() -> AIProvider:
    """Get AI provider for agent calls, checking AGENT_AI_PROVIDER first."""
    return _create_ai_provider_from_env("AGENT")
