from typing import Any, Type, TypeVar, Optional

from openai import OpenAI
from pydantic import BaseModel

from app.ai.provider import AIProvider

T = TypeVar("T", bound=BaseModel)


class OpenAIProvider(AIProvider):
    """AI provider backed by the OpenAI API."""

    def __init__(self, api_key: str, model: str, timeout: Optional[float] = None, max_retries: Optional[int] = None, max_tokens: Optional[int] = None):
        # For OpenAI, we use the provided timeout and max_retries if set, otherwise use defaults
        client_kwargs = {
            "api_key": api_key,
        }
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        self._client = OpenAI(**client_kwargs)
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_tokens = max_tokens

    def generate_structured(
        self,
        system_prompt: str,
        user_text: str,
        response_model: Type[T],
    ) -> T:
        request_kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "response_model": response_model,
        }
        if self._max_tokens is not None:
            request_kwargs["max_tokens"] = self._max_tokens
        completion = self._client.beta.chat.completions.parse(**request_kwargs)

        result = completion.choices[0].message.parsed
        if result is None:
            raise RuntimeError("OpenAI returned no structured output")

        return result

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> Any:
        request_kwargs = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        if self._max_tokens is not None:
            request_kwargs["max_tokens"] = self._max_tokens
        return self._client.chat.completions.create(**request_kwargs)