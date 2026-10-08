import os

from app.ai.kilo import KiloProvider
from app.ai.ollama import OllamaProvider
from app.ai.openai import OpenAIProvider
from app.ai.provider import AIProvider


def get_provider() -> AIProvider:
    """Select and instantiate an AI provider based on environment configuration."""
    provider_name = (
        os.environ.get("AI_PROVIDER", "openai")
        .strip()
        .strip('"')
        .strip("'")
        .lower()
    )

    if provider_name == "ollama":
        base_url = os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        model = os.environ.get("OLLAMA_MODEL")
        if not model:
            raise RuntimeError("OLLAMA_MODEL environment variable is not set")
        return OllamaProvider(base_url=base_url, model=model)

    if provider_name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIProvider(api_key=api_key, model=model)

    if provider_name == "kilo":
        api_key = os.environ.get("KILO_API_KEY")
        if not api_key:
            raise RuntimeError("KILO_API_KEY environment variable is not set")
        base_url = os.environ.get(
            "KILO_BASE_URL", "https://api.kilo.ai/api/gateway"
        )
        model = os.environ.get("KILO_MODEL")
        if not model:
            raise RuntimeError("KILO_MODEL environment variable is not set")
        return KiloProvider(api_key=api_key, base_url=base_url, model=model)

    raise RuntimeError(
        f"Invalid AI_PROVIDER: '{provider_name}'. "
        "Must be 'ollama', 'openai', or 'kilo'."
    )