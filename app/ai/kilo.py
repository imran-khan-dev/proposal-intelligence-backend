from typing import Any, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from app.ai.provider import AIProvider

T = TypeVar("T", bound=BaseModel)


class KiloProvider(AIProvider):
    """AI provider backed by the Kilo AI Gateway (OpenAI-compatible)."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120.0,
        )
        self._model = model

    def generate_structured(
        self,
        system_prompt: str,
        user_text: str,
        response_model: Type[T],
    ) -> T:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            response_format={"type": "json_object"},
        )

        raw = completion.choices[0].message.content
        if not raw:
            raise RuntimeError("Kilo returned empty response")

        try:
            return response_model.model_validate_json(raw)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to validate Kilo response as "
                f"{response_model.__name__}: {exc}"
            ) from exc

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> Any:
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )