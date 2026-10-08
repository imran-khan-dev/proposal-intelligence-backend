"""Test provider selection with AI_PROVIDER=kilo."""
import os
import sys

# Override for testing
os.environ["AI_PROVIDER"] = "kilo"
os.environ["KILO_API_KEY"] = "test-key-123"
os.environ["KILO_MODEL"] = "kilo-auto/free"
os.environ["KILO_BASE_URL"] = "https://api.kilo.ai/api/gateway"

sys.path.insert(0, ".")
from app.ai.factory import get_provider
from app.ai.kilo import KiloProvider
from app.ai.ollama import OllamaProvider
from app.ai.openai import OpenAIProvider
from app.ai.provider import AIProvider

provider = get_provider()
print(f"AI_PROVIDER=kilo -> Selected provider: {type(provider).__name__}")
print(f"Is KiloProvider: {isinstance(provider, KiloProvider)}")
print(f"Implements AIProvider: {isinstance(provider, AIProvider)}")
print(f"Base URL: {provider._client.base_url}")
print(f"Model: {provider._model}")
print(f"API Key set: {bool(provider._client.api_key)}")

assert isinstance(provider, KiloProvider), "Should be KiloProvider"
assert isinstance(provider, AIProvider), "Should implement AIProvider"
assert str(provider._client.base_url).rstrip("/") == "https://api.kilo.ai/api/gateway"
assert provider._model == "kilo-auto/free"
assert provider._client.api_key == "test-key-123"

print("\nPASS: AI_PROVIDER=kilo correctly selects KiloProvider")