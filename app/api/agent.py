from fastapi import APIRouter, HTTPException

from app.agent.service import create_agent_service
from app.schemas.agent import AgentRunRequest, AgentRunResponse

router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(request: AgentRunRequest):
    """Run the agent with a user message.

    The agent may call registered tools (count_requirements, calculate_total)
    and return the final answer along with a record of tool calls made.
    """
    try:
        service = create_agent_service()
        return service.run(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Agent run failed: {exc}"
        ) from exc