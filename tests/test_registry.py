"""Tests for the ToolRegistry."""
import pytest

from app.agent.builtin_tools import calculate_total, count_requirements
from app.agent.tools import Tool, ToolRegistry


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = Tool(
            name="test_tool",
            description="A test tool",
            handler=lambda x: x,
            parameters={"type": "object"},
        )
        registry.register(tool)
        assert registry.get("test_tool") is tool

    def test_get_unknown_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_tools(self):
        registry = ToolRegistry()
        t1 = Tool("a", "desc a", lambda: None, {})
        t2 = Tool("b", "desc b", lambda: None, {})
        registry.register(t1)
        registry.register(t2)
        assert len(registry.list_tools()) == 2

    def test_names(self):
        registry = ToolRegistry()
        registry.register(Tool("a", "", lambda: None, {}))
        registry.register(Tool("b", "", lambda: None, {}))
        assert set(registry.names()) == {"a", "b"}

    def test_to_openai_format(self):
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="my_tool",
                description="Does something",
                handler=lambda: None,
                parameters={"type": "object", "properties": {}},
            )
        )
        result = registry.to_openai_format()
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "my_tool"
        assert result[0]["function"]["description"] == "Does something"

    def test_execute_known_tool(self):
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="double",
                description="Doubles a number",
                handler=lambda x: x * 2,
                parameters={"type": "object"},
            )
        )
        result = registry.execute("double", {"x": 5})
        assert result == 10

    def test_execute_unknown_tool_raises_value_error(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError) as exc_info:
            registry.execute("nonexistent", {})
        assert "nonexistent" in str(exc_info.value)

    def test_execute_with_invalid_arguments_raises(self):
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="strict_tool",
                description="Requires x",
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
        with pytest.raises(TypeError):
            registry.execute("strict_tool", {"requirements": "not a list"})

    def test_calculate_total_via_registry(self):
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="calculate_total",
                description="Calculate total",
                handler=calculate_total,
                parameters={"type": "object"},
            )
        )
        items = [
            {"description": "A", "quantity": 2, "unit_price": 5.0}
        ]
        result = registry.execute("calculate_total", {"items": items})
        assert result == 10.0