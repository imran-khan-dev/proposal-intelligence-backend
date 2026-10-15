import logging

from fastapi import HTTPException

from app.modules.agent.service import create_agent_service
from app.modules.agent.schema import AgentRunRequest, AgentRunResponse

logger = logging.getLogger(__name__)


async def run_agent_controller(
    request: AgentRunRequest,
) -> AgentRunResponse:
    """Handle the HTTP-level agent execution flow."""

    try:
        service = create_agent_service()
        return await service.run(request)
    except Exception as exc:
        from app.ai.error_handling import handle_ai_exception
        raise handle_ai_exception(exc)
