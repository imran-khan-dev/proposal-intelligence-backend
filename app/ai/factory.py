import os
from typing import Optional

from app.ai.google import GoogleProvider
from app.ai.groq import GroqProvider
from app.ai.kilo import KiloProvider
from app.ai.ollama import OllamaProvider
from app.ai.openai import OpenAIProvider
from app.ai.provider import AIProvider
from app.core.singletons import get_ai_provider as _get_ai_provider_singleton


# Read and validate AI_TIMEOUT
_ai_timeout_env = os.environ.get("AI_TIMEOUT")
if _ai_timeout_env is not None:
    try:
        _ai_timeout = float(_ai_timeout_env)
        if _ai_timeout <= 0:
            raise ValueError
    except ValueError:
        raise RuntimeError("AI_TIMEOUT must be a positive float")
else:
    _ai_timeout = None

# Read and validate AI_MAX_RETRIES
_ai_max_retries_env = os.environ.get("AI_MAX_RETRIES")
if _ai_max_retries_env is not None:
    try:
        _ai_max_retries = int(_ai_max_retries_env)
        if _ai_max_retries < 0:
            raise ValueError
    except ValueError:
        raise RuntimeError("AI_MAX_RETRIES must be a non-negative integer")
else:
    _ai_max_retries = None

# Read and validate RFP_AI_MAX_TOKENS
_rfp_ai_max_tokens_env = os.environ.get("RFP_AI_MAX_TOKENS")
if _rfp_ai_max_tokens_env is not None:
    try:
        _rfp_ai_max_tokens = int(_rfp_ai_max_tokens_env)
        if _rfp_ai_max_tokens <= 0:
            raise ValueError
    except ValueError:
        raise RuntimeError("RFP_AI_MAX_TOKENS must be a positive integer")
else:
    _rfp_ai_max_tokens = None

# Read and validate AGENT_AI_MAX_TOKENS
_agent_ai_max_tokens_env = os.environ.get("AGENT_AI_MAX_TOKENS")
if _agent_ai_max_tokens_env is not None:
    try:
        _agent_ai_max_tokens = int(_agent_ai_max_tokens_env)
        if _agent_ai_max_tokens <= 0:
            raise ValueError
    except ValueError:
        raise RuntimeError("AGENT_AI_MAX_TOKENS must be a positive integer")
else:
    _agent_ai_max_tokens = None


def _create_ai_provider_from_env(prefix: str = "", timeout: Optional[float] = None, max_retries: Optional[int] = None, max_tokens: Optional[int] = None) -> AIProvider:
    """Create a new AI provider instance from environment variables.
    
    Args:
        prefix: If non-empty, use env vars like PREFIX_PROVIDER, PREFIX_MODEL, etc.
                If empty, fall back to AI_PROVIDER, OPENAI_API_KEY, etc.
        timeout: Request timeout in seconds. If None, use provider's default.
        max_retries: Maximum retry attempts. If None, use provider's default.
        max_tokens: Maximum output tokens. If None, no limit.
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

    # Determine effective timeout: if explicitly passed, use it; otherwise use provider's default when not configured via AI_TIMEOUT
    effective_timeout = timeout
    if effective_timeout is None and _ai_timeout is not None:
        effective_timeout = _ai_timeout
    
    # Determine effective max_retries: if explicitly passed, use it; otherwise use from AI_MAX_RETRIES if set
    effective_max_retries = max_retries
    if effective_max_retries is None and _ai_max_retries is not None:
        effective_max_retries = _ai_max_retries

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
        # For Ollama, we pass timeout and max_retries to the OpenAI client
        client_kwargs = {
            "api_key": "ollama",  # Ollama uses a fixed API key
            "base_url": base_url,
        }
        if effective_timeout is not None:
            client_kwargs["timeout"] = effective_timeout
        else:
            # Preserve current hardcoded timeout of 120.0 when not configured
            client_kwargs["timeout"] = 120.0
        if effective_max_retries is not None:
            client_kwargs["max_retries"] = effective_max_retries
        return OllamaProvider(
            base_url=base_url,
            model=model,
            timeout=client_kwargs["timeout"],
            max_retries=client_kwargs.get("max_retries"),
            max_tokens=max_tokens
        )

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
        # For OpenAI, we pass timeout and max_retries to the OpenAI client
        client_kwargs = {
            "api_key": api_key,
        }
        if effective_timeout is not None:
            client_kwargs["timeout"] = effective_timeout
        if effective_max_retries is not None:
            client_kwargs["max_retries"] = effective_max_retries
        return OpenAIProvider(
            api_key=api_key,
            model=model,
            timeout=client_kwargs.get("timeout"),
            max_retries=client_kwargs.get("max_retries"),
            max_tokens=max_tokens
        )

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
        # For Kilo, we pass timeout and max_retries to the OpenAI client
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
        }
        if effective_timeout is not None:
            client_kwargs["timeout"] = effective_timeout
        else:
            # Preserve current hardcoded timeout of 120.0 when not configured
            client_kwargs["timeout"] = 120.0
        if effective_max_retries is not None:
            client_kwargs["max_retries"] = effective_max_retries
        return KiloProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=client_kwargs["timeout"],
            max_retries=client_kwargs.get("max_retries"),
            max_tokens=max_tokens
        )

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
        # For Google, we currently only support timeout and max_retries at the client level if the SDK allows it.
        # We'll pass them to the constructor but note that the Google provider may not use them.
        return GoogleProvider(
            api_key=api_key,
            model=model,
            timeout=effective_timeout,
            max_retries=effective_max_retries,
            max_tokens=max_tokens
        )

    raise RuntimeError(
        f"Invalid AI_PROVIDER: '{provider_name}'. "
        "Must be 'ollama', 'openai', 'kilo', 'google', or 'groq'."
    )


def _create_ai_provider() -> AIProvider:
    """Create a new AI provider instance (not singleton)."""
    return _create_ai_provider_from_env("", timeout=_ai_timeout, max_retries=_ai_max_retries, max_tokens=None)


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
    return _create_ai_provider_from_env("RFP", timeout=_ai_timeout, max_retries=_ai_max_retries, max_tokens=_rfp_ai_max_tokens)


def get_agent_provider() -> AIProvider:
    """Get AI provider for agent calls, checking AGENT_AI_PROVIDER first."""
    return _create_ai_provider_from_env("AGENT", timeout=_ai_timeout, max_retries=_ai_max_retries, max_tokens=_agent_ai_max_tokens)
