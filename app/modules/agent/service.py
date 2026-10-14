"""Agent service that orchestrates tool calling.

Flow:
  1. Build the initial message list (system + user).
  2. Call provider.chat_with_tools(messages, tools).
  3. Inspect the response for tool_calls.
  4. If no tool_calls → return the model'"'"'s final answer.
  5. If tool_calls → validate tool name and arguments, execute,
     append tool results to messages, and repeat from step 2.
  6. Stop after max_tool_rounds to prevent infinite loops.
"""

from typing import Any, Optional
import uuid

from app.ai.embedding_factory import get_embedding_provider
from app.ai.embeddings import EmbeddingProvider
from app.ai.factory import get_agent_provider
from app.ai.provider import AIProvider
from app.agent_core.tools import ToolRegistry
from app.rag.store import VectorStore
from app.rag.store_factory import get_vector_store
from app.modules.agent.schema import AgentRunRequest, AgentRunResponse, ToolCallRecord


class Conversation(list):
    """Per-run message list carrying a durable ``conversation_id``.

    ``AgentService.run`` builds the message history once and reuses the same list
    object across every provider call in the run. The AI provider is a shared
    singleton and ``chat_with_tools`` takes no run-scoped handle, so the stable
    per-run identity is attached here as ``conversation_id``. Stateful providers
    (e.g. Google Gemini) use it to scope ``previous_interaction_id`` to this run
    without changing the ``AIProvider`` interface, so Ollama/OpenAI/Kilo are
    untouched.
    """

    conversation_id: str

    def __init__(self, items=None, conversation_id: str = "") -> None:
        super().__init__(items if items is not None else [])
        self.conversation_id = conversation_id



SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

When the user asks a question that requires calculation or counting,
use the appropriate tool. Always prefer using a tool over guessing.

Available tools:
- count_requirements: Count items in a list of requirements.
- calculate_total: Calculate the total cost of line items.
- search_company_knowledge: Search the company knowledge base for relevant information.

If a tool is not needed, answer directly from your knowledge."""


# Agent Step 2: Agent Register Here and Agent is Used from Here
class AgentService:
    """Orchestrates tool-calling with a configured AI provider."""

    def __init__(self, provider: AIProvider, registry: ToolRegistry):
        self._provider = provider
        self._registry = registry
        # Optional dependencies exposed for ingestion/retrieval workflows.
        # These are the same instances used by the registered search_company_knowledge tool.
        self.embedder: Optional[EmbeddingProvider] = None
        self.store: Optional[VectorStore] = None

    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        messages: list[dict[str, Any]] = Conversation(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message},
            ],
            conversation_id=str(uuid.uuid4()),
        )
        tools = self._registry.to_openai_format()

        tool_call_records: list[ToolCallRecord] = []

        for round_num in range(1, request.max_tool_rounds + 1):
            completion = self._provider.chat_with_tools(
                messages=messages,
                tools=tools,
            )

            if completion is None:
                raise RuntimeError("Provider returned None completion")

            choices = getattr(completion, "choices", None)
            if not choices:
                raise RuntimeError("Provider returned completion with no choices")

            choice = choices[0]
            if choice is None:
                raise RuntimeError("Provider returned completion with None choice")

            assistant_message = getattr(choice, "message", None)
            if assistant_message is None:
                raise RuntimeError("Provider returned choice with no message")

            # Normalize assistant message into plain dicts for provider compatibility.
            # OpenAI-compatible APIs require tool_calls as dicts with id, type,
            # and function{name, arguments}. Raw SDK objects may not serialize.
            raw_tool_calls = getattr(assistant_message, "tool_calls", None)
            if raw_tool_calls:
                normalized_tool_calls = []
                for tc in raw_tool_calls:
                    normalized_tool_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": normalized_tool_calls,
                    }
                )
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                    }
                )

            tool_calls = raw_tool_calls
            if not tool_calls:
                # Model produced a final answer with no tool calls
                final_answer = assistant_message.content or ""
                return AgentRunResponse(
                    final_answer=final_answer,
                    tool_calls=tool_call_records,
                    rounds_used=round_num,
                )

            # Execute each requested tool call
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                raw_args = tool_call.function.arguments

                # Parse arguments
                try:
                    import json

                    arguments = (
                        json.loads(raw_args)
                        if isinstance(raw_args, str)
                        else raw_args
                    )
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Failed to parse tool arguments for '{tool_name}': {exc}"
                    ) from exc

                # Validate tool name
                if not self._registry.get(tool_name):
                    raise RuntimeError(
                        f"Model requested unknown tool: '{tool_name}'. "
                        f"Available: {self._registry.names()}"
                    )

                # Execute the tool
                try:
                    result = await self._registry.async_execute(
                        tool_name, arguments
                    )
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Tool '{tool_name}' failed: {exc}"
                    ) from exc
                except Exception as exc:
                    raise RuntimeError(
                        f"Tool '{tool_name}' raised an unexpected error: {exc}"
                    ) from exc

                tool_call_records.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        arguments=arguments,
                        result=result,
                    )
                )

                # Append tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

        # Max rounds reached without a final answer
        return AgentRunResponse(
            final_answer="Maximum tool-call rounds reached without a final answer.",
            tool_calls=tool_call_records,
            rounds_used=request.max_tool_rounds,
        )


# Agent Step 1: Build the AgentService
def create_agent_service(
    provider: AIProvider = None,
    embedder: EmbeddingProvider = None,
    store: VectorStore = None,
) -> AgentService:
    """Build an AgentService with the configured provider and default tools.

    Args:
        provider: AI provider for chat completions. If None, uses get_agent_provider().
        embedder: Embedding provider for RAG. If None, uses get_embedding_provider().
        store: Vector store for RAG. If None, creates InMemoryVectorStore().
               Pass a PersistentVectorStore for persistent knowledge.

    Returns:
        Configured AgentService with count_requirements, calculate_total,
        and search_company_knowledge tools registered.
    """
    if provider is None:
        provider = get_agent_provider()
    registry = ToolRegistry()

    # Create shared embedding provider and vector store for RAG
    if embedder is None:
        embedder = get_embedding_provider()
    if store is None:
        store = get_vector_store()

    from app.agent_core.builtin_tools import (
        calculate_total,
        count_requirements,
        create_search_company_knowledge_tool,
    )
    from app.agent_core.tools import Tool

    registry.register(
        Tool(
            name="count_requirements",
            description=(
                "Count the number of items in a list of requirements. "
                "Use this when the user asks how many requirements there are."
            ),
            handler=count_requirements,
            parameters={
                "type": "object",
                "properties": {
                    "requirements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of requirement strings.",
                    }
                },
                "required": ["requirements"],
            },
        )
    )

    registry.register(
        Tool(
            name="calculate_total",
            description=(
                "Calculate the total cost of a list of line items. "
                "Each item has a description, quantity, and unit_price. "
                "Use this when the user asks for a total cost or sum."
            ),
            handler=calculate_total,
            parameters={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "quantity": {
                                    "type": "number",
                                    "minimum": 0,
                                },
                                "unit_price": {
                                    "type": "number",
                                    "minimum": 0,
                                },
                            },
                            "required": [
                                "description",
                                "quantity",
                                "unit_price",
                            ],
                        },
                    }
                },
                "required": ["items"],
            },
        )
    )

    # Register RAG search tool with injected dependencies
    search_tool_handler = create_search_company_knowledge_tool(
        embedder=embedder,
        store=store,
    )

    registry.register(
        Tool(
            name="search_company_knowledge",
            description=(
                "Search the company knowledge base for relevant information. "
                "Use this when the user asks questions about company documents, "
                "policies, or previously ingested RFPs."
            ),
            handler=search_tool_handler,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query text.",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": "Maximum number of results to return (default: 5).",
                    },
                },
                "required": ["query"],
            },
        )
    )

    service = AgentService(provider=provider, registry=registry)
    service.embedder = embedder
    service.store = store
    return service
