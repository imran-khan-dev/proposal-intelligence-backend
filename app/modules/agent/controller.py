import logging
import uuid

from fastapi import HTTPException

from app.core.workspace import resolve_workspace_id
from app.modules.agent.service import create_agent_service
from app.modules.agent.schema import AgentRunRequest, AgentRunResponse
from app.rag.store_factory import get_vector_store
from app.rag.stores.postgres_vector_store import PostgreSQLVectorStore

logger = logging.getLogger(__name__)


async def run_agent_controller(
    request: AgentRunRequest,
) -> AgentRunResponse:
    """Handle the HTTP-level agent execution flow.

    The agent's knowledge-base search uses the same server-resolved workspace
    as the document ingestion endpoints, so document search results are scoped
    to the same workspace and cannot leak across workspaces.
    """
    try:
        ws_uuid = resolve_workspace_id()
        store = get_vector_store()
        if isinstance(store, PostgreSQLVectorStore):
            store = PostgreSQLVectorStore(workspace_id=ws_uuid)
        service = create_agent_service(store=store)
        return await service.run(request)
    except Exception as exc:
        from app.ai.error_handling import handle_ai_exception

        raise handle_ai_exception(exc)
