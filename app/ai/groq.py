from typing import Any, Type, TypeVar, Optional

from openai import OpenAI
from pydantic import BaseModel

from app.ai.provider import AIProvider

T = TypeVar("T", bound=BaseModel)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(AIProvider):
    """AI provider backed by the Groq API (OpenAI-compatible)."""

    def __init__(self, api_key: str, base_url: str, model: str, 
                 timeout: Optional[float] = None, 
                 max_retries: Optional[int] = None,
                 max_tokens: Optional[int] = None):
        # For Groq, we preserve the current hardcoded timeout of 120.0 when timeout is not explicitly set via AI_TIMEOUT
        # But note: the factory passes timeout=None when AI_TIMEOUT is not set, and we want to use 120.0 in that case
        client_timeout = timeout
        if client_timeout is None:
            client_timeout = 120.0  # Preserve existing behavior
        
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": client_timeout,
        }
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
            
        self._client = OpenAI(**client_kwargs)
        self._model = model
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
            "response_format": {"type": "json_object"},
        }
        if self._max_tokens is not None:
            request_kwargs["max_tokens"] = self._max_tokens
            
        completion = self._client.chat.completions.create(**request_kwargs)

        raw = completion.choices[0].message.content
        if not raw:
            raise RuntimeError("Groq returned empty response")

        try:
            return response_model.model_validate_json(raw)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to validate Groq response as "
                f"{response_model.__name__}: {exc}"
            ) from exc

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