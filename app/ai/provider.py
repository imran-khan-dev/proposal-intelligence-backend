from abc import ABC, abstractmethod
from typing import Type, TypeVar, Any

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    @abstractmethod
    def generate_structured(
        self,
        system_prompt: str,
        user_text: str,
        response_model: Type[T],
    ) -> T:
        """Generate a response validated against a Pydantic model."""
        ...

    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> Any:
        """Send a chat completion request with tool definitions.

        Returns the raw completion object from the underlying SDK.
        The caller is responsible for inspecting tool_calls and
        executing them, then sending results back via another call.
        """
        ...