"""Google Gemini provider backed by the Gemini Interactions API.

Uses ``genai.Client.interactions.create`` (API revision 2026-05-20). This is the
Interactions API, not ``client.models.generate_content``.

Structured output: ``response_format={"type":"text","mime_type":"application/json",
"schema": <JSON Schema>}`` with the validated text read from ``interaction.output_text``.

Function calling: tools are declared as ``{"type":"function","name","description",
"parameters":<JSON Schema>``. The model's calls are returned as steps with
``step.type == "function_call"`` exposing ``id``, ``name`` and ``arguments`` (a dict,
not a JSON string). Turns are linked with ``previous_interaction_id`` and function
results are submitted back as ``function_result`` steps.

Conversation-state note (important): the Interactions API is server-stateful, so
turns are chained with ``previous_interaction_id``. This cannot be replayed
statelessly through the existing ``AgentService`` message loop because that loop
round-trips the transcript through plain OpenAI-style dicts, which strips the
``signature`` fields the API requires for stateless continuation. Instead this
provider keeps the current interaction id in a per-run map keyed by a durable
``conversation_id`` that ``AgentService`` attaches to its shared message list
(see ``Conversation``). A fresh first turn (no ``role=="tool"`` messages) always
clears the entry for that id, so a finished run's id is never reused by the next
run. This does not rely on ``id()`` and isolates each run even under the
synchronous, event-loop-blocking controller (per-worker serialization).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel

from google import genai
from google.genai import errors as _genai_errors

from app.ai.provider import AIProvider

try:
    from google.genai._gaos.lib.compat_errors import APIError as _NextGenAPIError
except Exception:  # pragma: no cover - resilient to SDK layout changes
    _NextGenAPIError = _genai_errors.APIError

T = TypeVar("T", bound=BaseModel)

_API_ERRORS: tuple = tuple(
    {c for c in (_genai_errors.APIError, _NextGenAPIError, httpx.HTTPError) if isinstance(c, type)}
)

_FAILURE_STATUSES = {"failed", "cancelled", "incomplete", "budget_exceeded"}


@dataclass
class _Conversation:
    interaction_id: str | None = None


class _Function:
    __slots__ = ("name", "arguments")

    def __init__(self, name: str, arguments: Any) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    """OpenAI-shaped tool call expected by AgentService."""

    __slots__ = ("id", "type", "function")

    def __init__(self, call_id: str, name: str, arguments: Any) -> None:
        self.id = call_id
        self.type = "function"
        self.function = _Function(name=name, arguments=arguments)


class _Message:
    __slots__ = ("content", "tool_calls")

    def __init__(
        self,
        content: str | None,
        tool_calls: list[_ToolCall] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    __slots__ = ("message", "finish_reason")

    def __init__(self, message: _Message, finish_reason: str) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Completion:
    """Minimal OpenAI-compatible view over a Gemini interaction response."""

    __slots__ = ("id", "choices")

    def __init__(self, completion_id: str, message: _Message, finish_reason: str) -> None:
        self.id = completion_id
        self.choices = [_Choice(message=message, finish_reason=finish_reason)]


class GoogleProvider(AIProvider):
    """AI provider backed by the Google Gemini Interactions API."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._conversations: dict[str, _Conversation] = {}
        self._lock = threading.Lock()

    def generate_structured(
        self,
        system_prompt: str,
        user_text: str,
        response_model: Type[T],
    ) -> T:
        schema = response_model.model_json_schema()
        response_format = {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema,
        }
        try:
            interaction = self._client.interactions.create(
                model=self._model,
                input=user_text,
                system_instruction=system_prompt,
                response_format=response_format,
            )
        except _API_ERRORS as exc:
            raise RuntimeError(
                f"Google API call failed: {_sanitize_error(exc)}"
            ) from exc

        _check_status(interaction)
        raw = (getattr(interaction, "output_text", None) or "").strip()
        if not raw:
            raise RuntimeError(
                f"Google returned empty response for {response_model.__name__}"
            )
        try:
            return response_model.model_validate_json(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"Failed to validate Google response as "
                f"{response_model.__name__}: {exc}"
            ) from exc

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> Any:
        messages = messages or []
        conversation_id = getattr(messages, "conversation_id", None)
        key = conversation_id if conversation_id is not None else "__default__"
        is_fresh = not _has_tool_messages(messages)

        with self._lock:
            if is_fresh:
                self._conversations.pop(key, None)
                conv = _Conversation()
            else:
                conv = self._conversations.get(key)
                if conv is None or not conv.interaction_id:
                    raise RuntimeError(
                        "Google provider: continuation requested but no active "
                        "interaction exists (state was reset by a concurrent run)."
                    )

        system_instruction = _extract_system(messages)
        converted_tools = _convert_tools(tools)

        if is_fresh:
            user_text = _extract_user_text(messages)
            create_kwargs = {
                "model": self._model,
                "input": user_text,
                "system_instruction": system_instruction,
                "tools": converted_tools,
            }
        else:
            result_steps = []
            for call_id, name, result in _extract_trailing_function_results(messages):
                result_steps.append(
                    {
                        "type": "function_result",
                        "name": name,
                        "call_id": call_id,
                        "result": [{"type": "text", "text": result}],
                    }
                )
            create_kwargs = {
                "model": self._model,
                "input": result_steps,
                "system_instruction": system_instruction,
                "tools": converted_tools,
                "previous_interaction_id": conv.interaction_id,
            }

        try:
            interaction = self._client.interactions.create(**create_kwargs)
        except _API_ERRORS as exc:
            raise RuntimeError(
                f"Google API call failed: {_sanitize_error(exc)}"
            ) from exc

        _check_status(interaction)
        calls = _extract_function_calls(interaction)
        text = _extract_output_text(interaction)

        with self._lock:
            if calls:
                conv.interaction_id = getattr(interaction, "id", None) or conv.interaction_id
                self._conversations[key] = conv
            else:
                self._conversations.pop(key, None)

        if calls:
            tool_calls = [_ToolCall(cid, name, args) for cid, name, args in calls]
            message = _Message(content=text or None, tool_calls=tool_calls)
            finish_reason = "tool_calls"
        else:
            message = _Message(content=text or "", tool_calls=None)
            finish_reason = "stop"

        return _Completion(
            completion_id=getattr(interaction, "id", None) or "",
            message=message,
            finish_reason=finish_reason,
        )


def _has_tool_messages(messages: list[dict]) -> bool:
    return any(
        isinstance(m, dict) and m.get("role") == "tool" for m in messages
    )


def _extract_system(messages: list[dict]) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            return m.get("content") or ""
    return ""


def _extract_user_text(messages: list[dict]) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _convert_tools(tools: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function":
            fn = t.get("function") or {}
            params = fn.get("parameters")
            if params is None:
                params = {"type": "object"}
            converted.append(
                {
                    "type": "function",
                    "name": fn.get("name"),
                    "description": fn.get("description"),
                    "parameters": params,
                }
            )
        else:
            converted.append(t)
    return converted


def _find_call_name(messages: list[dict], call_id: str) -> str:
    for m in messages:
        if not (isinstance(m, dict) and m.get("role") == "assistant"):
            continue
        for tc in m.get("tool_calls", []) or []:
            if isinstance(tc, dict):
                if tc.get("id") == call_id:
                    return (tc.get("function") or {}).get("name")
            elif getattr(tc, "id", None) == call_id:
                return getattr(getattr(tc, "function", None), "name", None)
    raise RuntimeError(
        f"Google provider: tool result references unknown call id '{call_id}'"
    )


def _extract_trailing_function_results(messages: list[dict]) -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []
    i = len(messages)
    while (
        i > 0
        and isinstance(messages[i - 1], dict)
        and messages[i - 1].get("role") == "tool"
    ):
        i -= 1
        msg = messages[i]
        call_id = msg.get("tool_call_id")
        result = msg.get("content", "")
        name = _find_call_name(messages, call_id)
        results.append((call_id, name, result))
    results.reverse()
    return results


def _step_type(step: Any) -> str | None:
    if isinstance(step, dict):
        return step.get("type")
    return getattr(step, "type", None)


def _step_attr(step: Any, name: str, default: Any = None) -> Any:
    if isinstance(step, dict):
        return step.get(name, default)
    return getattr(step, name, default)


def _extract_function_calls(interaction: Any) -> list[tuple[str, str, Any]]:
    calls: list[tuple[str, str, Any]] = []
    for step in getattr(interaction, "steps", None) or []:
        if _step_type(step) == "function_call":
            call_id = _step_attr(step, "id")
            name = _step_attr(step, "name")
            args = _step_attr(step, "arguments", None) or {}
            calls.append((call_id, name, args))
    return calls


def _extract_output_text(interaction: Any) -> str:
    text = getattr(interaction, "output_text", None) or ""
    if text:
        return text
    collected: list[str] = []
    for step in getattr(interaction, "steps", None) or []:
        if _step_type(step) != "model_output":
            continue
        content = _step_attr(step, "content", None)
        blocks = content if isinstance(content, list) else [content]
        for block in blocks:
            blk_text = _step_attr(block, "text", None)
            if blk_text:
                collected.append(blk_text)
    return "\n".join(collected)


def _check_status(interaction: Any) -> None:
    status = getattr(interaction, "status", None)
    if status in _FAILURE_STATUSES:
        errors = getattr(interaction, "errors", None)
        raise RuntimeError(
            f"Google interaction ended with status '{status}': {errors}"
        )


def _sanitize_error(exc: BaseException) -> str:
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    message = (
        getattr(exc, "message", None)
        or getattr(exc, "details", None)
        or ""
    )
    parts: list[str] = []
    if status_code is not None:
        parts.append(f"status={status_code}")
    if status:
        parts.append(f"reason={status}")
    if message:
        parts.append(str(message)[:300])
    return ", ".join(parts) if parts else "unknown error"
