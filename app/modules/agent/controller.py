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

    except RuntimeError as exc:
        logger.exception("Agent execution failed")
        raise HTTPException(
            status_code=502,
            detail="The AI agent could not complete the request.",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error during agent execution")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while running the agent.",
        ) from exc