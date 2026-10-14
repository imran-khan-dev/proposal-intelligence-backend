from dataclasses import dataclass
from typing import Any, Callable, Optional, Awaitable
import inspect


@dataclass
class Tool:
    """A registered tool with a name, description, and callable handler."""

    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict


class ToolRegistry:
    """Registry of explicitly allowed tools.

    Only tools registered here can be executed. The model can only
    request tools by name; arbitrary code execution is never allowed.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def to_openai_format(self) -> list[dict]:
        """Convert registered tools to OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict) -> Any:
        """Execute a registered tool by name with validated arguments."""
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(
                f"Unknown tool: '{name}'. "
                f"Available tools: {self.names()}"
            )
        result = tool.handler(**arguments)
        if inspect.isawaitable(result):
            raise RuntimeError(
                f"Tool '{name}' is async but ToolRegistry.execute() is synchronous. "
                f"Use ToolRegistry.async_execute() instead."
            )
        return result

    async def async_execute(self, name: str, arguments: dict) -> Any:
        """Asynchronously execute a registered tool by name with validated arguments."""
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(
                f"Unknown tool: '{name}'. "
                f"Available tools: {self.names()}"
            )
        result = tool.handler(**arguments)
        if inspect.isawaitable(result):
            return await result
        return result