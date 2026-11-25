from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar

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
        """Generate a structured response from the AI provider."""
        pass

    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> Any:
        """Chat with the AI provider using tools."""
        pass