"""Verify .env loading and provider selection."""
import os
import sys

sys.path.insert(0, ".")
from dotenv import load_dotenv

load_dotenv()

print("=== .env values loaded ===")
print(f"AI_PROVIDER: {os.environ.get('AI_PROVIDER')}")
print(f"OLLAMA_BASE_URL: {os.environ.get('OLLAMA_BASE_URL')}")
print(f"OLLAMA_MODEL: {os.environ.get('OLLAMA_MODEL')}")
print(f"KILO_BASE_URL: {os.environ.get('KILO_BASE_URL')}")
print(f"KILO_API_KEY: {'SET' if os.environ.get('KILO_API_KEY') else 'NOT SET'}")
print(f"KILO_MODEL: {os.environ.get('KILO_MODEL')}")
print(f"OPENAI_API_KEY: {'SET' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'}")
print(f"OPENAI_MODEL: {os.environ.get('OPENAI_MODEL')}")

print("\n=== Provider selection ===")
from app.ai.factory import get_provider
from app.ai.kilo import KiloProvider
from app.ai.ollama import OllamaProvider
from app.ai.openai import OpenAIProvider

provider = get_provider()
print(f"Selected provider: {type(provider).__name__}")
print(f"Is KiloProvider: {isinstance(provider, KiloProvider)}")
print(f"Is OllamaProvider: {isinstance(provider, OllamaProvider)}")
print(f"Is OpenAIProvider: {isinstance(provider, OpenAIProvider)}")

if isinstance(provider, KiloProvider):
    print(f"Kilo base_url: {provider._client.base_url}")
    print(f"Kilo model: {provider._model}")
    print(f"Kilo api_key set: {bool(provider._client.api_key)}")

print("\n=== All checks passed ===")