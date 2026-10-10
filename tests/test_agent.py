"""Tests for the AgentService with mocked provider responses."""
import json
from unittest.mock import MagicMock

import pytest

from app.agent.service import AgentService
from app.agent.tools import ToolRegistry
from app.schemas.agent import AgentRunRequest


def _make_tool_call_response(tool_name, arguments, call_id="call_1", content=None):
    """Build a mock OpenAI-style completion with a tool call."""
    func_mock = MagicMock()
    func_mock.name = tool_name
    func_mock.arguments = json.dumps(arguments)

    msg = MagicMock()
    msg.content = content
    msg.tool_calls = [
        MagicMock(
            id=call_id,
            function=func_mock,
        )
    ]
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _make_text_response(text):
    """Build a mock completion with no tool calls."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _make_service(provider):
    """Build an AgentService with a mock provider and default tools."""
    from app.agent.builtin_tools import calculate_total, count_requirements
    from app.agent.tools import Tool

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="count_requirements",
            description="Count items in a list.",
            handler=count_requirements,
            parameters={
                "type": "object",
                "properties": {
                    "requirements": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["requirements"],
            },
        )
    )
    registry.register(
        Tool(
            name="calculate_total",
            description="Calculate total cost.",
            handler=calculate_total,
            parameters={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "quantity": {"type": "number"},
                                "unit_price": {"type": "number"},
                            },
                            "required": [
                                "description",
                                "quantity",
                                "unit_price",
                            ],
                        },
                    }
                },
                "required": ["items"],
            },
        )
    )
    return AgentService(provider=provider, registry=registry)


class TestSuccessfulToolCall:
    def test_count_requirements_tool_call(self):
        provider = MagicMock()
        provider.chat_with_tools.side_effect = [
            _make_tool_call_response(
                "count_requirements",
                {"requirements": ["a", "b", "c"]},
            ),
            _make_text_response("There are 3 requirements."),
        ]
        service = _make_service(provider)
        request = AgentRunRequest(message="How many requirements are there?")
        response = service.run(request)

        assert response.final_answer == "There are 3 requirements."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "count_requirements"
        assert response.tool_calls[0].arguments == {
            "requirements": ["a", "b", "c"]
        }
        assert response.tool_calls[0].result == 3
        assert response.rounds_used == 2

    def test_calculate_total_tool_call(self):
        provider = MagicMock()
        provider.chat_with_tools.side_effect = [
            _make_tool_call_response(
                "calculate_total",
                {
                    "items": [
                        {
                            "description": "Widget",
                            "quantity": 3,
                            "unit_price": 10.0,
                        }
                    ]
                },
            ),
            _make_text_response("The total cost is $30.00."),
        ]
        service = _make_service(provider)
        request = AgentRunRequest(message="What is the total cost?")
        response = service.run(request)

        assert response.final_answer == "The total cost is $30.00."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "calculate_total"
        assert response.tool_calls[0].result == 30.0


class TestNoToolCall:
    def test_direct_answer(self):
        provider = MagicMock()
        provider.chat_with_tools.return_value = _make_text_response(
            "Hello! I can help with calculations."
        )
        service = _make_service(provider)
        request = AgentRunRequest(message="Say hello")
        response = service.run(request)

        assert response.final_answer == "Hello! I can help with calculations."
        assert len(response.tool_calls) == 0
        assert response.rounds_used == 1


class TestUnknownTool:
    def test_unknown_tool_raises_runtime_error(self):
        provider = MagicMock()
        provider.chat_with_tools.return_value = _make_tool_call_response(
            "nonexistent_tool",
            {"arg": "value"},
        )
        service = _make_service(provider)
        request = AgentRunRequest(message="Use nonexistent tool")

        with pytest.raises(RuntimeError) as exc_info:
            service.run(request)
        assert "nonexistent_tool" in str(exc_info.value)


class TestInvalidArguments:
    def test_invalid_json_arguments_raises(self):
        provider = MagicMock()
        msg = MagicMock()
        msg.content = None
        func_mock = MagicMock()
        func_mock.name = "count_requirements"
        func_mock.arguments = "not valid json"
        msg.tool_calls = [
            MagicMock(
                id="call_1",
                function=func_mock,
            )
        ]
        choice = MagicMock()
        choice.message = msg
        completion = MagicMock()
        completion.choices = [choice]
        provider.chat_with_tools.return_value = completion

        service = _make_service(provider)
        request = AgentRunRequest(message="Count something")

        with pytest.raises(RuntimeError) as exc_info:
            service.run(request)
        assert "Failed to parse" in str(exc_info.value)

    def test_type_error_in_tool_raises(self):
        provider = MagicMock()
        provider.chat_with_tools.return_value = _make_tool_call_response(
            "count_requirements",
            {"requirements": "not a list"},
        )
        service = _make_service(provider)
        request = AgentRunRequest(message="Count items")

        with pytest.raises(RuntimeError) as exc_info:
            service.run(request)
        assert "count_requirements" in str(exc_info.value)


class TestMaxRounds:
    def test_max_rounds_reached(self):
        provider = MagicMock()
        # Always respond with a tool call
        provider.chat_with_tools.return_value = _make_tool_call_response(
            "count_requirements",
            {"requirements": ["a"]},
        )
        service = _make_service(provider)
        request = AgentRunRequest(
            message="Keep calling tools", max_tool_rounds=2
        )
        response = service.run(request)

        assert "Maximum tool-call rounds" in response.final_answer
        assert len(response.tool_calls) == 2
        assert response.rounds_used == 2


class TestMultipleToolCallsInOneResponse:
    """Model returns multiple tool calls in a single assistant message."""

    def test_multiple_tools_executed(self):
        provider = MagicMock()
        provider.chat_with_tools.side_effect = [
            _make_multi_tool_call_response(),
            _make_text_response("Done: 3 items and $30.00 total."),
        ]
        service = _make_service(provider)
        request = AgentRunRequest(
            message="Count [a,b,c] and calculate total of 3x$10."
        )
        response = service.run(request)

        assert response.final_answer == "Done: 3 items and $30.00 total."
        assert len(response.tool_calls) == 2
        names = {tc.tool_name for tc in response.tool_calls}
        assert names == {"count_requirements", "calculate_total"}
        # Verify results
        results = {tc.tool_name: tc.result for tc in response.tool_calls}
        assert results["count_requirements"] == 3
        assert results["calculate_total"] == 30.0
        assert response.rounds_used == 2

    def test_multiple_tools_correct_tool_call_ids(self):
        """Tool result messages must reference the correct tool_call_id."""
        provider = MagicMock()
        provider.chat_with_tools.side_effect = [
            _make_multi_tool_call_response(),
            _make_text_response("ok"),
        ]
        service = _make_service(provider)
        request = AgentRunRequest(message="test")

        # Run and inspect the messages sent to the provider
        service.run(request)

        # Second call to chat_with_tools should include tool result messages
        # with matching tool_call_ids
        second_call_messages = provider.chat_with_tools.call_args_list[1].kwargs[
            "messages"
        ]
        tool_msgs = [m for m in second_call_messages if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        ids = {m["tool_call_id"] for m in tool_msgs}
        assert ids == {"call_1", "call_2"}


def _make_multi_tool_call_response():
    """Build a mock response with two tool calls in one message."""
    func1 = MagicMock()
    func1.name = "count_requirements"
    func1.arguments = json.dumps({"requirements": ["a", "b", "c"]})

    func2 = MagicMock()
    func2.name = "calculate_total"
    func2.arguments = json.dumps(
        {
            "items": [
                {"description": "Widget", "quantity": 3, "unit_price": 10.0}
            ]
        }
    )

    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [
        MagicMock(id="call_1", function=func1),
        MagicMock(id="call_2", function=func2),
    ]
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


class TestEmptyToolCalls:
    """Model returns an explicit empty tool_calls list."""

    def test_empty_tool_calls_treated_as_final_answer(self):
        provider = MagicMock()
        msg = MagicMock()
        msg.content = "Final answer with empty tool_calls."
        msg.tool_calls = []  # explicit empty list
        choice = MagicMock()
        choice.message = msg
        completion = MagicMock()
        completion.choices = [choice]
        provider.chat_with_tools.return_value = completion

        service = _make_service(provider)
        request = AgentRunRequest(message="Hello")
        response = service.run(request)

        assert response.final_answer == "Final answer with empty tool_calls."
        assert len(response.tool_calls) == 0
        assert response.rounds_used == 1


class TestAssistantContentNone:
    """Assistant message has content=None but no tool calls."""

    def test_content_none_no_tool_calls(self):
        provider = MagicMock()
        msg = MagicMock()
        msg.content = None
        msg.tool_calls = None
        choice = MagicMock()
        choice.message = msg
        completion = MagicMock()
        completion.choices = [choice]
        provider.chat_with_tools.return_value = completion

        service = _make_service(provider)
        request = AgentRunRequest(message="Hello")
        response = service.run(request)

        assert response.final_answer == ""
        assert len(response.tool_calls) == 0
        assert response.rounds_used == 1


class TestProviderException:
    """Provider raises an exception during chat_with_tools."""

    def test_provider_exception_propagates(self):
        provider = MagicMock()
        provider.chat_with_tools.side_effect = RuntimeError("API error")
        service = _make_service(provider)
        request = AgentRunRequest(message="test")

        with pytest.raises(RuntimeError) as exc_info:
            service.run(request)
        assert "API error" in str(exc_info.value)


class TestAssistantMessageSerialization:
    """Verify assistant tool-call messages are normalized to plain dicts."""

    def test_assistant_message_normalized_to_dicts(self):
        provider = MagicMock()
        provider.chat_with_tools.side_effect = [
            _make_tool_call_response(
                "count_requirements",
                {"requirements": ["a"]},
            ),
            _make_text_response("ok"),
        ]
        service = _make_service(provider)
        request = AgentRunRequest(message="test")
        service.run(request)

        # Inspect the messages sent on the second call
        second_call_messages = provider.chat_with_tools.call_args_list[1].kwargs[
            "messages"
        ]
        assistant_msgs = [
            m for m in second_call_messages if m["role"] == "assistant"
        ]
        # The history includes the first assistant message (with tool_calls)
        # and the second assistant message (final text response)
        assert len(assistant_msgs) == 2
        # First assistant message must have normalized tool_calls as plain dicts
        am = assistant_msgs[0]
        assert isinstance(am, dict)
        assert "tool_calls" in am
        tc = am["tool_calls"][0]
        assert isinstance(tc, dict)
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "count_requirements"
        assert isinstance(tc["function"]["arguments"], str)

    def test_no_tool_calls_omits_field(self):
        provider = MagicMock()
        provider.chat_with_tools.return_value = _make_text_response("hi")
        service = _make_service(provider)
        request = AgentRunRequest(message="test")
        service.run(request)

        sent_messages = provider.chat_with_tools.call_args.kwargs["messages"]
        assistant_msgs = [
            m for m in sent_messages if m["role"] == "assistant"
        ]
        assert len(assistant_msgs) == 1
        am = assistant_msgs[0]
        assert "tool_calls" not in am


class TestProviderResponseValidation:
    """Provider returns malformed responses."""

    def test_provider_returns_none(self):
        provider = MagicMock()
        provider.chat_with_tools.return_value = None
        service = _make_service(provider)
        request = AgentRunRequest(message="test")

        with pytest.raises(RuntimeError) as exc_info:
            service.run(request)
        assert "None completion" in str(exc_info.value)

    def test_provider_returns_empty_choices(self):
        provider = MagicMock()
        completion = MagicMock()
        completion.choices = []
        provider.chat_with_tools.return_value = completion
        service = _make_service(provider)
        request = AgentRunRequest(message="test")

        with pytest.raises(RuntimeError) as exc_info:
            service.run(request)
        assert "no choices" in str(exc_info.value)

    def test_provider_returns_none_choice(self):
        provider = MagicMock()
        completion = MagicMock()
        completion.choices = [None]
        provider.chat_with_tools.return_value = completion
        service = _make_service(provider)
        request = AgentRunRequest(message="test")

        with pytest.raises(RuntimeError) as exc_info:
            service.run(request)
        assert "None choice" in str(exc_info.value)

    def test_provider_returns_choice_with_none_message(self):
        provider = MagicMock()
        choice = MagicMock()
        choice.message = None
        completion = MagicMock()
        completion.choices = [choice]
        provider.chat_with_tools.return_value = completion
        service = _make_service(provider)
        request = AgentRunRequest(message="test")

        with pytest.raises(RuntimeError) as exc_info:
            service.run(request)
        assert "no message" in str(exc_info.value)


class TestUnexpectedToolException:
    """Tool handler raises an exception beyond TypeError/ValueError."""

    def test_key_error_in_tool_raises_runtime_error(self):
        """Tool raising KeyError is wrapped in RuntimeError with tool name."""
        provider = MagicMock()
        provider.chat_with_tools.return_value = _make_tool_call_response(
            "count_requirements",
            {"requirements": ["a", "b"]},
        )
        service = _make_service(provider)

        # Replace the tool handler with one that raises KeyError
        def bad_handler(requirements):
            raise KeyError("missing key")

        service._registry._tools["count_requirements"].handler = bad_handler

        request = AgentRunRequest(message="Count items")
        with pytest.raises(RuntimeError) as exc_info:
            service.run(request)
        assert "count_requirements" in str(exc_info.value)
        assert "unexpected error" in str(exc_info.value)
        assert "missing key" in str(exc_info.value)

    def test_custom_exception_in_tool_raises_runtime_error(self):
        """Tool raising a custom exception is wrapped in RuntimeError."""

        class CustomToolError(Exception):
            pass

        provider = MagicMock()
        provider.chat_with_tools.return_value = _make_tool_call_response(
            "calculate_total",
            {"items": [{"description": "X", "quantity": 1, "unit_price": 10.0}]},
        )
        service = _make_service(provider)

        def bad_handler(items):
            raise CustomToolError("custom failure")

        service._registry._tools["calculate_total"].handler = bad_handler

        request = AgentRunRequest(message="Calculate total")
        with pytest.raises(RuntimeError) as exc_info:
            service.run(request)
        assert "calculate_total" in str(exc_info.value)
        assert "unexpected error" in str(exc_info.value)
        assert "custom failure" in str(exc_info.value)