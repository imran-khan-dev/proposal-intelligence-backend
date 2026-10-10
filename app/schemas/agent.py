from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    """Record of a single tool call and its result."""

    tool_name: str = Field(..., description="Name of the tool that was called")
    arguments: dict[str, Any] = Field(
        ..., description="Arguments passed to the tool"
    )
    result: Any = Field(..., description="Result returned by the tool")


class AgentRunRequest(BaseModel):
    """Request body for POST /api/agent/run."""

    message: str = Field(
        ..., min_length=1, description="User message for the agent"
    )
    max_tool_rounds: int = Field(
        5, ge=1, le=10, description="Maximum number of tool-call rounds"
    )


class AgentRunResponse(BaseModel):
    """Response body for POST /api/agent/run."""

    final_answer: str = Field(
        ..., description="Final answer produced by the model"
    )
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="Record of all tool calls made during the run",
    )
    rounds_used: int = Field(
        ..., description="Number of tool-call rounds executed"
    )