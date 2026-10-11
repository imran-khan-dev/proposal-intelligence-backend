
from fastapi import APIRouter

from app.modules.agent.controller import run_agent_controller
from app.modules.agent.schema import AgentRunRequest, AgentRunResponse


router = APIRouter(tags=["agent"])


@router.post("/run", response_model=AgentRunResponse)
async def run_agent_route(
    request: AgentRunRequest,
) -> AgentRunResponse:
    return await run_agent_controller(request)
