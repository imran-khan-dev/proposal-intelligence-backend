from typing import Any, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from app.ai.provider import AIProvider

T = TypeVar("T", bound=BaseModel)


class OpenAIProvider(AIProvider):
    """AI provider backed by the OpenAI API."""

    def __init__(self, api_key: str, model: str):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate_structured(
        self,
        system_prompt: str,
        user_text: str,
        response_model: Type[T],
    ) -> T:
        completion = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            response_model=response_model,
        )

        result = completion.choices[0].message.parsed
        if result is None:
            raise RuntimeError("OpenAI returned no structured output")

        return result

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