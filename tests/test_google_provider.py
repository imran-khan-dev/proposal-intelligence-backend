"""Focused tests for the Google Gemini Interactions provider.

These tests mock ``genai.Client.interactions.create`` and assert on the shape of
the requests the provider sends and the adapters it returns. No live Google API
calls are made, so these tests CANNOT verify real server behavior (status
transitions, multi-turn continuation, or model availability).
"""

from types import SimpleNamespace
import inspect
import json

import pytest
from pydantic import BaseModel

from app.agent_core.tools import Tool, ToolRegistry
from app.ai.factory import _create_ai_provider, _create_ai_provider_from_env
from app.ai.google import GoogleProvider
from app.ai.groq import GroqProvider
from app.ai.kilo import KiloProvider
from app.ai.ollama import OllamaProvider
from app.ai.openai import OpenAIProvider
from app.modules.agent.schema import AgentRunRequest
from app.modules.agent.service import AgentService


class _FakeStep(SimpleNamespace):
    pass


class _FakeInteraction(SimpleNamespace):
    pass


class _FakeResp(SimpleNamespace):
    pass


def _make_response(status_code, request=None):
    return _FakeResp(
        status_code=status_code,
        request=request or SimpleNamespace(),
    )


def _provider(create_fn):
    """Build a GoogleProvider whose SDK client is fully mocked (no network)."""
    provider = GoogleProvider(api_key="test-key", model="gemini-3.8-flash")
    provider._client = SimpleNamespace(
        interactions=SimpleNamespace(create=create_fn)
    )
    return provider


def _count_items_tool():
    return Tool(
        name="count_items",
        description="Return the number of items in the list.",
        handler=lambda items: len(items),
        parameters={
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["items"],
        },
    )


def _agent(provider, message="count: a,b,c", max_rounds=5):
    registry = ToolRegistry()
    registry.register(_count_items_tool())
    return AgentService(provider=provider, registry=registry)


# ---------------------------------------------------------------------------
# 1 & 2. Structured output
# ---------------------------------------------------------------------------


class _Recipe(BaseModel):
    recipe_name: str
    ingredients: list[str]


def _structured_provider(payload_text):
    def create(**kwargs):
        assert kwargs["response_format"]["type"] == "text"
        assert kwargs["response_format"]["mime_type"] == "application/json"
        assert kwargs["response_format"]["schema"]["type"] == "object"
        return SimpleNamespace(output_text=payload_text, status="completed", steps=[])
    return create


def test_generate_structured_parses_valid_json_into_model():
    p = _provider(_structured_provider('{"recipe_name":"Soup","ingredients":["carrot","onion"]}'))
    out = p.generate_structured(
        system_prompt="system",
        user_text="extract a recipe",
        response_model=_Recipe,
    )
    assert out == _Recipe(recipe_name="Soup", ingredients=["carrot", "onion"])


def test_generate_structured_empty_response_raises():
    p = _provider(_structured_provider(""))
    with pytest.raises(RuntimeError, match="empty response"):
        p.generate_structured("s", "u", _Recipe)


def test_generate_structured_invalid_json_raises():
    p = _provider(_structured_provider("not json"))
    with pytest.raises(RuntimeError, match="Failed to validate"):
        p.generate_structured("s", "u", _Recipe)


# ---------------------------------------------------------------------------
# 3. Function call -> OpenAI-compatible tool-call shape
# ---------------------------------------------------------------------------


def test_function_call_converted_to_openai_shape():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _FakeInteraction(
            id="int-1",
            status="requires_action",
            output_text="",
            steps=[
                _FakeStep(
                    type="function_call",
                    id="fc-1",
                    name="count_items",
                    arguments={"items": ["a", "b", "c"]},
                )
            ],
        )

    p = _provider(create)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "count: a,b,c"},
    ]
    from app.modules.agent.service import Conversation  # verify helper exists
    messages = Conversation([{"role": "system", "content": "sys"},
                             {"role": "user", "content": "count: a,b,c"}],
                            conversation_id="cid-1")
    completion = p.chat_with_tools(messages, [
        {"type": "function", "function": {"name": "count_items", "description": "d", "parameters": {"type": "object"}}}
    ])

    choice = completion.choices[0]
    tc = choice.message.tool_calls[0]
    assert tc.id == "fc-1"
    assert tc.type == "function"
    assert tc.function.name == "count_items"
    # Google returns arguments as a dict; AgentService accepts that directly.
    assert isinstance(tc.function.arguments, dict)
    assert tc.function.arguments == {"items": ["a", "b", "c"]}
    assert choice.finish_reason == "tool_calls"
    assert captured["input"] == "count: a,b,c"
    assert captured["response_format"] is None if "response_format" in captured else True


# ---------------------------------------------------------------------------
# 4. Single tool call round-trip through AgentService
# ---------------------------------------------------------------------------


async def test_single_tool_call_round_trip():
    turns = {"n": 0}

    def create(**kwargs):
        if kwargs.get("previous_interaction_id") is None:
            # Fresh first turn -> make a function call.
            return _FakeInteraction(
                id="int-1", status="requires_action", output_text="",
                steps=[_FakeStep(type="function_call", id="fc-1",
                                name="count_items", arguments={"items": ["a", "b", "c"]})])
        # Continuation: the agent fed back the tool result.
        fr = kwargs["input"][0]
        assert fr["type"] == "function_result"
        assert fr["call_id"] == "fc-1"
        assert fr["name"] == "count_items"
        assert json.loads(fr["result"][0]["text"]) == 3
        assert kwargs["previous_interaction_id"] == "int-1"
        return _FakeInteraction(id="int-2", status="completed",
                                output_text="3 items", steps=[])

    p = _provider(create)
    resp = await _agent(p).run(AgentRunRequest(message="count: a,b,c"))
    assert resp.final_answer == "3 items"
    assert resp.rounds_used == 2
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].tool_name == "count_items"
    assert resp.tool_calls[0].arguments == {"items": ["a", "b", "c"]}
    assert resp.tool_calls[0].result == 3


# ---------------------------------------------------------------------------
# 5. Multiple tool calls in one turn, results submitted in order
# ---------------------------------------------------------------------------


async def test_multiple_tool_calls_order_preserved():
    def create(**kwargs):
        if kwargs.get("previous_interaction_id") is None:
            return _FakeInteraction(
                id="int-1", status="requires_action", output_text="",
                steps=[
                    _FakeStep(type="function_call", id="fc-A", name="count_items",
                              arguments={"items": ["a", "b"]}),
                    _FakeStep(type="function_call", id="fc-B", name="count_items",
                              arguments={"items": ["a", "b", "c", "d"]}),
                ])
        results = kwargs["input"]
        assert len(results) == 2
        assert results[0]["call_id"] == "fc-A"
        assert json.loads(results[0]["result"][0]["text"]) == 2
        assert results[0]["name"] == "count_items"
        assert results[1]["call_id"] == "fc-B"
        assert json.loads(results[1]["result"][0]["text"]) == 4
        assert kwargs["previous_interaction_id"] == "int-1"
        return _FakeInteraction(id="int-2", status="completed",
                                output_text="2 and 4", steps=[])

    p = _provider(create)
    resp = await _agent(p, "count a,b then a,b,c,d").run(
        AgentRunRequest(message="count a,b then a,b,c,d"))
    assert resp.final_answer == "2 and 4"
    assert [tc.tool_name for tc in resp.tool_calls] == ["count_items", "count_items"]
    assert [tc.result for tc in resp.tool_calls] == [2, 4]


# ---------------------------------------------------------------------------
# 6. Separate agent runs do not share interaction state
# ---------------------------------------------------------------------------


async def test_conversation_state_isolated_between_runs():
    log = {"last_fresh": None, "continuations": []}

    def create(**kwargs):
        prev = kwargs.get("previous_interaction_id")
        if prev is None:
            # Fresh first turn: mint a new interaction id for this run.
            fresh_id = f"int-{len(log['continuations']) + 1}"
            log["last_fresh"] = fresh_id
            return _FakeInteraction(
                id=fresh_id, status="requires_action", output_text="",
                steps=[_FakeStep(type="function_call", id="fc-1",
                                name="count_items", arguments={"items": ["a", "b"]})])
        # Continuation: MUST chain to THIS run's own fresh interaction, not a
        # stale id from a previous run.
        assert prev == log["last_fresh"], f"leaked previous_interaction_id: {prev}"
        log["continuations"].append(prev)
        return _FakeInteraction(id=f"{prev}-next", status="completed",
                                output_text="done", steps=[])

    p = _provider(create)
    r1 = await _agent(p, "first run").run(AgentRunRequest(message="count a,b"))
    r2 = await _agent(p, "second run").run(AgentRunRequest(message="count a,b"))
    assert r1.final_answer == "done"
    assert r2.final_answer == "done"
    assert len(log["continuations"]) == 2
    # The two runs must have chained to two DIFFERENT interaction ids.
    assert log["continuations"][0] != log["continuations"][1]


# ---------------------------------------------------------------------------
# 7. Async tools (e.g. search_company_knowledge) are awaited by async_execute
# ---------------------------------------------------------------------------


class _FakeAsyncProvider:
    """Provider whose chat_with_tools returns a tool call first, then a final answer."""

    def __init__(self) -> None:
        self.calls = 0

    def chat_with_tools(self, messages, tools):
        self.calls += 1
        has_tool_result = any(
            isinstance(m, dict) and m.get("role") == "tool" for m in messages
        )
        if not has_tool_result:
            tc = SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(
                    name="search_company_knowledge",
                    arguments='{"query": "hello"}',
                ),
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None, tool_calls=[tc])
                    )
                ]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Found answer via knowledge base", tool_calls=None
                    )
                )
            ]
        )


async def _async_search_handler(query: str, top_k: int = 5):
    """Async handler mimicking search_company_knowledge's return shape."""
    return [
        {"text": "secret info", "score": 0.9, "metadata": {"doc_id": "d1"}}
    ]


async def test_async_tool_is_awaited_and_result_used():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="search_company_knowledge",
            description="Search the company knowledge base.",
            handler=_async_search_handler,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
        )
    )
    provider = _FakeAsyncProvider()
    service = AgentService(provider=provider, registry=registry)

    resp = await service.run(AgentRunRequest(message="what is the policy?"))

    assert provider.calls == 2
    assert resp.final_answer == "Found answer via knowledge base"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].tool_name == "search_company_knowledge"
    assert resp.tool_calls[0].arguments == {"query": "hello"}
    # The result must be the awaited list, not a leaked coroutine.
    assert resp.tool_calls[0].result == [
        {"text": "secret info", "score": 0.9, "metadata": {"doc_id": "d1"}}
    ]
    assert not inspect.isawaitable(resp.tool_calls[0].result)


async def test_sync_tool_still_supported_via_async_run():
    """Synchronous handlers must keep working through the async execution path."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="count_items",
            description="Count items.",
            handler=lambda items: len(items),
            parameters={
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["items"],
            },
        )
    )

    class _SyncProvider:
        def __init__(self) -> None:
            self._answered = False

        def chat_with_tools(self, messages, tools):
            has_tool_result = any(
                isinstance(m, dict) and m.get("role") == "tool" for m in messages
            )
            if not has_tool_result:
                tc = SimpleNamespace(
                    id="c1", type="function",
                    function=SimpleNamespace(
                        name="count_items", arguments='{"items":["a","b","c"]}'
                    ),
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=None, tool_calls=[tc])
                    )]
                )
            self._answered = True
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="3 items", tool_calls=None)
                )]
            )

    provider = _SyncProvider()
    service = AgentService(provider=provider, registry=registry)
    resp = await service.run(AgentRunRequest(message="count a,b,c"))
    assert provider._answered
    assert resp.final_answer == "3 items"
    assert resp.tool_calls[0].result == 3


# ---------------------------------------------------------------------------
# 8. Google API errors are handled without leaking the API key
# ---------------------------------------------------------------------------


def test_api_error_does_not_leak_key():
    from google.genai._gaos.lib.compat_errors import AuthenticationError

    def create(**kwargs):
        raise AuthenticationError(
            "Unauthorized",
            response=_make_response(401),
            body={"error": {"message": "API key not valid"}},
        )

    p = _provider(create)
    with pytest.raises(RuntimeError) as exc_info:
        p.generate_structured("s", "u", _Recipe)
    msg = str(exc_info.value)
    assert "test-key" not in msg
    assert "status=401" in msg


# ---------------------------------------------------------------------------
# 9. Existing providers still initialize through their own configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider, env",
    [
        ("ollama", {"OLLAMA_MODEL": "test-model"}),
        ("openai", {"OPENAI_API_KEY": "unit-test-key"}),
        ("kilo", {"KILO_API_KEY": "unit-test-key", "KILO_MODEL": "kilo-auto/free"}),
        ("google", {"GOOGLE_API_KEY": "unit-test-key", "GOOGLE_MODEL": "gemini-3.8-flash"}),
        ("groq", {"GROQ_API_KEY": "unit-test-key"}),
    ],
)
def test_existing_providers_still_initialize(monkeypatch, provider, env):
    monkeypatch.setenv("AI_PROVIDER", provider)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    instance = _create_ai_provider()
    expected = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "kilo": KiloProvider,
        "google": GoogleProvider,
        "groq": GroqProvider,
    }[provider]
    assert isinstance(instance, expected)
    if provider == "google":
        assert instance._model == "gemini-3.8-flash"
    if provider == "groq":
        assert instance._model == "openai/gpt-oss-20b"
        assert str(instance._client.base_url) == "https://api.groq.com/openai/v1/"


def test_google_model_default_and_config(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-key")
    monkeypatch.delenv("GOOGLE_MODEL", raising=False)
    instance = _create_ai_provider()
    assert isinstance(instance, GoogleProvider)
    assert instance._model == "gemini-3.8-flash"


def test_groq_model_default_and_config(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    instance = _create_ai_provider()
    assert isinstance(instance, GroqProvider)
    assert instance._model == "openai/gpt-oss-20b"
    assert str(instance._client.base_url) == "https://api.groq.com/openai/v1/"


def test_groq_custom_model_and_base_url(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_BASE_URL", "https://custom.example/v1")
    instance = _create_ai_provider()
    assert isinstance(instance, GroqProvider)
    assert instance._model == "llama-3.3-70b-versatile"
    assert str(instance._client.base_url) == "https://custom.example/v1/"


def test_groq_missing_api_key_raises(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        _create_ai_provider()


def test_unsupported_provider_raises():
    import builtins
    orig = builtins.__import__
    # Ensure env is not accidentally google from conftest defaults.
    try:
        import importlib
        import os
        os.environ["AI_PROVIDER"] = "bogus_provider"
        with pytest.raises(RuntimeError, match="Invalid AI_PROVIDER"):
            _create_ai_provider()
    finally:
        os.environ.pop("AI_PROVIDER", None)


# ---------------------------------------------------------------------------
# 10. Google provider timeout and retry configuration via HttpOptions
# ---------------------------------------------------------------------------


def test_google_provider_accepts_timeout_and_max_retries():
    provider = GoogleProvider(
        api_key="test-key", model="gemini-3.8-flash", timeout=45.0, max_retries=3
    )
    assert provider._timeout == 45.0
    assert provider._max_retries == 3
    assert provider._max_tokens is None


def test_google_provider_without_timeout_and_retries():
    provider = GoogleProvider(api_key="test-key", model="gemini-3.8-flash")
    assert provider._timeout is None
    assert provider._max_retries is None


def test_google_provider_accepts_max_tokens():
    provider = GoogleProvider(
        api_key="test-key", model="gemini-3.8-flash", max_tokens=512
    )
    assert provider._max_tokens == 512


# ---------------------------------------------------------------------------
# 11. Task-specific factory provider selection (RFP and AGENT prefixes)
# ---------------------------------------------------------------------------


def test_rfp_provider_uses_prefix_env(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("RFP_AI_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    instance = _create_ai_provider_from_env(
        "RFP", timeout=None, max_retries=None, max_tokens=None
    )
    assert isinstance(instance, OllamaProvider)
    assert instance._model == "test-model"


def test_rfp_provider_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.delenv("RFP_AI_PROVIDER", raising=False)
    instance = _create_ai_provider_from_env(
        "RFP", timeout=None, max_retries=None, max_tokens=None
    )
    assert isinstance(instance, OpenAIProvider)


def test_agent_provider_uses_prefix_env(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("AGENT_AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-key")
    instance = _create_ai_provider_from_env(
        "AGENT", timeout=None, max_retries=None, max_tokens=None
    )
    assert isinstance(instance, GroqProvider)


def test_rfp_provider_with_empty_prefix_falls_back(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "kilo")
    monkeypatch.setenv("KILO_API_KEY", "unit-test-key")
    monkeypatch.setenv("KILO_MODEL", "kilo-auto/free")
    monkeypatch.setenv("RFP_AI_PROVIDER", "")
    instance = _create_ai_provider_from_env(
        "RFP", timeout=None, max_retries=None, max_tokens=None
    )
    assert isinstance(instance, KiloProvider)


def test_rfp_max_tokens_propagation(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    instance = _create_ai_provider_from_env(
        "RFP", timeout=None, max_retries=None, max_tokens=100
    )
    assert isinstance(instance, OpenAIProvider)
    assert instance._max_tokens == 100


def test_agent_max_tokens_propagation(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    instance = _create_ai_provider_from_env(
        "AGENT", timeout=None, max_retries=None, max_tokens=200
    )
    assert isinstance(instance, OpenAIProvider)
    assert instance._max_tokens == 200


def test_rfp_provider_task_specific_model(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("RFP_AI_PROVIDER", "openai")
    monkeypatch.setenv("RFP_OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("RFP_OPENAI_API_KEY", "rfp-specific-key")
    instance = _create_ai_provider_from_env(
        "RFP", timeout=None, max_retries=None, max_tokens=None
    )
    assert isinstance(instance, OpenAIProvider)
    assert instance._model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# 12. Google provider forwards max_tokens to the Interactions API
# ---------------------------------------------------------------------------


def _structured_provider_captures_kwargs():
    """Mock interactions.create that returns a valid structured response
    and captures the kwargs it received."""
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            output_text='{"recipe_name":"Soup","ingredients":["carrot","onion"]}',
            status="completed",
            steps=[],
        )
    create.captured = captured
    return create


def test_generate_structured_includes_generation_config_when_max_tokens_set():
    mock_create = _structured_provider_captures_kwargs()
    provider = GoogleProvider(api_key="test-key", model="gemini-3.8-flash", max_tokens=256)
    provider._client = SimpleNamespace(interactions=SimpleNamespace(create=mock_create))

    provider.generate_structured(
        system_prompt="system",
        user_text="extract a recipe",
        response_model=_Recipe,
    )

    assert mock_create.captured["model"] == "gemini-3.8-flash"
    assert mock_create.captured["generation_config"] == {"max_output_tokens": 256}


def test_generate_structured_omits_generation_config_when_max_tokens_none():
    mock_create = _structured_provider_captures_kwargs()
    provider = GoogleProvider(api_key="test-key", model="gemini-3.8-flash", max_tokens=None)
    provider._client = SimpleNamespace(interactions=SimpleNamespace(create=mock_create))

    provider.generate_structured(
        system_prompt="system",
        user_text="extract a recipe",
        response_model=_Recipe,
    )

    assert "generation_config" not in mock_create.captured


def _tool_call_provider_captures_kwargs():
    """Mock interactions.create that returns a response with a function_call step
    and captures the kwargs it received."""
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="test-interaction-id",
            status="completed",
            steps=[],
            output_text="",
        )
    create.captured = captured
    return create


def test_chat_with_tools_includes_generation_config_when_max_tokens_set():
    mock_create = _tool_call_provider_captures_kwargs()
    provider = GoogleProvider(api_key="test-key", model="gemini-3.8-flash", max_tokens=128)
    provider._client = SimpleNamespace(interactions=SimpleNamespace(create=mock_create))

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count the items: a, b, c"},
    ]
    provider.chat_with_tools(messages, [])

    assert mock_create.captured["model"] == "gemini-3.8-flash"
    assert mock_create.captured["generation_config"] == {"max_output_tokens": 128}


def test_chat_with_tools_omits_generation_config_when_max_tokens_none():
    mock_create = _tool_call_provider_captures_kwargs()
    provider = GoogleProvider(api_key="test-key", model="gemini-3.8-flash", max_tokens=None)
    provider._client = SimpleNamespace(interactions=SimpleNamespace(create=mock_create))

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count the items: a, b, c"},
    ]
    provider.chat_with_tools(messages, [])

    assert "generation_config" not in mock_create.captured


