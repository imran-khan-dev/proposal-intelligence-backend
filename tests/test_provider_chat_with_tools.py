"""Tests for chat_with_tools() on OpenAI-compatible providers.

These tests verify that the request_kwargs dictionary is correctly built
with model, messages, tools, tool_choice, and optional max_tokens, and that
the OpenAI client's chat.completions.create is invoked with those kwargs.
No network calls are made — the OpenAI client is replaced with a mock.
"""

from unittest.mock import MagicMock

import pytest
from types import SimpleNamespace

from app.ai.ollama import OllamaProvider
from app.ai.openai import OpenAIProvider
from app.ai.kilo import KiloProvider
from app.ai.groq import GroqProvider


_PROVIDER_SPECS = [
    pytest.param(
        OllamaProvider,
        {"base_url": "http://localhost:11434", "model": "qwen3:4b"},
        id="ollama",
    ),
    pytest.param(
        OpenAIProvider,
        {"api_key": "test-key", "model": "gpt-4o-mini"},
        id="openai",
    ),
    pytest.param(
        KiloProvider,
        {"api_key": "test-key", "base_url": "https://api.kilo.ai/api/gateway", "model": "kilo-auto/free"},
        id="kilo",
    ),
    pytest.param(
        GroqProvider,
        {"api_key": "test-key", "base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-20b"},
        id="groq",
    ),
]


def _make_provider(provider_cls, kwargs):
    """Instantiate a provider and replace its OpenAI client with a mock."""
    provider = provider_cls(**kwargs)
    provider._client = MagicMock()
    return provider


class TestChatWithToolsRequest:
    @pytest.mark.parametrize("provider_cls,kwargs", _PROVIDER_SPECS)
    def test_builds_correct_request_kwargs(self, provider_cls, kwargs):
        provider = _make_provider(provider_cls, kwargs)
        messages = [{"role": "user", "content": "hello"}]
        tools = [{"type": "function", "function": {"name": "do_thing", "parameters": {"type": "object"}}}]

        provider.chat_with_tools(messages, tools)

        provider._client.chat.completions.create.assert_called_once()
        call_kwargs = provider._client.chat.completions.create.call_args.kwargs

        assert call_kwargs["model"] == kwargs["model"]
        assert call_kwargs["messages"] == messages
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.parametrize("provider_cls,kwargs", _PROVIDER_SPECS)
    def test_includes_max_tokens_when_set(self, provider_cls, kwargs):
        provider = _make_provider(provider_cls, {**kwargs, "max_tokens": 128})

        provider.chat_with_tools([], [])

        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 128

    @pytest.mark.parametrize("provider_cls,kwargs", _PROVIDER_SPECS)
    def test_omits_max_tokens_when_none(self, provider_cls, kwargs):
        provider = _make_provider(provider_cls, kwargs)

        provider.chat_with_tools([], [])

        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in call_kwargs

    @pytest.mark.parametrize("provider_cls,kwargs", _PROVIDER_SPECS)
    def test_returns_completion_object(self, provider_cls, kwargs):
        provider = _make_provider(provider_cls, kwargs)
        fake_completion = SimpleNamespace(id="test-id")
        provider._client.chat.completions.create.return_value = fake_completion

        result = provider.chat_with_tools([], [])

        assert result is fake_completion


class TestChatWithToolsEmptyMessages:
    @pytest.mark.parametrize("provider_cls,kwargs", _PROVIDER_SPECS)
    def test_empty_messages_still_sent(self, provider_cls, kwargs):
        provider = _make_provider(provider_cls, kwargs)

        provider.chat_with_tools([], [])

        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == []
        assert call_kwargs["tools"] == []
