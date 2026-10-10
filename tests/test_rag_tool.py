"""Tests for RAG search tool integration."""

import json
from unittest.mock import MagicMock

import pytest

from app.agent.builtin_tools import (
    create_search_company_knowledge_tool,
)
from app.agent.service import AgentService, create_agent_service
from app.agent.tools import ToolRegistry
from app.ai.embeddings import EmbeddingProvider
from app.rag.store import InMemoryVectorStore, SearchResult, VectorRecord
from app.schemas.agent import AgentRunRequest


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 384, should_fail: bool = False, query_embedding: list[float] = None):
        self._dimension = dimension
        self._should_fail = should_fail
        self._query_embedding = query_embedding
        self.last_query = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._should_fail:
            raise RuntimeError("Embedding API error")
        if self._query_embedding is not None:
            return [self._query_embedding]
        return [[0.1] * self._dimension] * len(texts)


def _make_test_service(registry: ToolRegistry = None, provider=None):
    """Create an AgentService with a mock provider and given registry."""
    from app.ai.provider import AIProvider
    if provider is None:
        mock_provider = MagicMock(spec=AIProvider)
    else:
        mock_provider = provider
    if registry is None:
        registry = ToolRegistry()
    return AgentService(provider=mock_provider, registry=registry)


def _make_registry_with_rag():
    """Create a ToolRegistry with the RAG tool registered."""
    registry = ToolRegistry()
    embedder = MockEmbeddingProvider(dimension=4)
    store = InMemoryVectorStore()
    from app.agent.builtin_tools import create_search_company_knowledge_tool
    from app.agent.tools import Tool

    search_tool_handler = create_search_company_knowledge_tool(embedder=embedder, store=store)
    registry.register(
        Tool(
            name="search_company_knowledge",
            description="Search the company knowledge base.",
            handler=search_tool_handler,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query text."},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
                "required": ["query"],
            },
        )
    )
    return registry


class TestSearchCompanyKnowledgeTool:
    """Tests for the search_company_knowledge tool handler."""

    def test_tool_returns_json_serializable_results(self):
        """Tool returns list of dicts with text, score, metadata."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4, query_embedding=[1.0, 0.0, 0.0, 0.0])

        store.add([
            VectorRecord(
                id="doc-1_chunk_0",
                text="Company policy on expenses",
                embedding=[1.0, 0.0, 0.0, 0.0],
                metadata={"doc_id": "doc-1", "chunk_index": 0, "source": "policy.pdf"},
            ),
        ])

        tool = create_search_company_knowledge_tool(embedder, store)
        results = tool(query="expense policy", top_k=3)

        assert isinstance(results, list)
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, dict)
        assert "text" in result
        assert "score" in result
        assert "metadata" in result
        assert result["text"] == "Company policy on expenses"
        assert result["score"] == 1.0
        assert result["metadata"]["doc_id"] == "doc-1"
        assert result["metadata"]["chunk_index"] == 0
        assert result["metadata"]["source"] == "policy.pdf"

        # Verify JSON serializable
        json.dumps(results)

    def test_tool_respects_top_k(self):
        """Tool respects top_k parameter."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4, query_embedding=[0.1] * 4)

        for i in range(5):
            store.add([
                VectorRecord(
                    id=f"doc_chunk_{i}",
                    text=f"Content {i}",
                    embedding=[0.1] * 4,
                    metadata={"doc_id": "doc", "chunk_index": i},
                )
            ])

        tool = create_search_company_knowledge_tool(embedder, store)
        results = tool(query="test", top_k=2)
        assert len(results) == 2

        results = tool(query="test", top_k=10)
        assert len(results) == 5

    def test_tool_empty_store_returns_empty_list(self):
        """Tool returns empty list for empty store."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4)

        tool = create_search_company_knowledge_tool(embedder, store)
        results = tool(query="anything", top_k=5)

        assert results == []

    def test_tool_invalid_top_k_raises(self):
        """Tool raises ValueError for non-positive top_k."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4)

        tool = create_search_company_knowledge_tool(embedder, store)

        with pytest.raises(ValueError, match="top_k must be positive"):
            tool(query="test", top_k=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            tool(query="test", top_k=-1)

    def test_tool_embedding_failure_propagates(self):
        """Tool propagates embedding failures as RuntimeError."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4, should_fail=True)

        tool = create_search_company_knowledge_tool(embedder, store)

        with pytest.raises(RuntimeError, match="Query embedding failed"):
            tool(query="test", top_k=5)

    def test_tool_dimension_mismatch_raises(self):
        """Tool propagates dimension mismatch from store."""
        store = InMemoryVectorStore()
        # Store has 3D embeddings
        store.add([
            VectorRecord(id="1", text="test", embedding=[1.0, 0.0, 0.0], metadata={}),
        ])

        # Embedder returns 4D embedding
        embedder = MockEmbeddingProvider(dimension=4, query_embedding=[0.1] * 4)

        tool = create_search_company_knowledge_tool(embedder, store)

        with pytest.raises(ValueError, match="Query embedding dimension mismatch"):
            tool(query="test", top_k=5)

    def test_same_store_instance_used_across_calls(self):
        """Tool uses the same store instance across multiple calls."""
        store = InMemoryVectorStore()
        embedder = MockEmbeddingProvider(dimension=4, query_embedding=[1.0, 0.0, 0.0, 0.0])

        tool = create_search_company_knowledge_tool(embedder, store)

        # First call - empty store
        results1 = tool(query="test", top_k=5)
        assert results1 == []

        # Add data directly to store
        store.add([
            VectorRecord(
                id="doc-1_chunk_0",
                text="New content",
                embedding=[1.0, 0.0, 0.0, 0.0],
                metadata={"doc_id": "doc-1", "chunk_index": 0},
            ),
        ])

        # Second call - should see the new data
        results2 = tool(query="test", top_k=5)
        assert len(results2) == 1
        assert results2[0]["text"] == "New content"


class TestSearchCompanyKnowledgeToolRegistration:
    """Tests for tool registration in the agent service."""

    def test_tool_registered_in_agent_service(self):
        """search_company_knowledge tool is registered and discoverable."""
        registry = _make_registry_with_rag()

        assert "search_company_knowledge" in registry.names()
        tool = registry.get("search_company_knowledge")
        assert tool is not None
        assert tool.name == "search_company_knowledge"
        assert "company knowledge base" in tool.description.lower()

    def test_tool_parameters_schema(self):
        """Tool has correct parameter schema for OpenAI function calling."""
        registry = _make_registry_with_rag()
        tool = registry.get("search_company_knowledge")

        assert tool.parameters["type"] == "object"
        assert "query" in tool.parameters["properties"]
        assert tool.parameters["properties"]["query"]["type"] == "string"
        assert "top_k" in tool.parameters["properties"]
        assert tool.parameters["properties"]["top_k"]["type"] == "integer"
        assert tool.parameters["properties"]["top_k"]["minimum"] == 1
        assert tool.parameters["properties"]["top_k"]["maximum"] == 20
        assert tool.parameters["properties"]["top_k"]["default"] == 5
        assert tool.parameters["required"] == ["query"]

    def test_tool_in_openai_format(self):
        """Tool appears in OpenAI format for the model."""
        registry = _make_registry_with_rag()
        openai_tools = registry.to_openai_format()

        search_tool = next((t for t in openai_tools if t["function"]["name"] == "search_company_knowledge"), None)
        assert search_tool is not None
        assert search_tool["type"] == "function"
        assert "query" in search_tool["function"]["parameters"]["properties"]
        assert "top_k" in search_tool["function"]["parameters"]["properties"]


def _make_tool_call_response(tool_name, arguments, call_id="call_1", content=None):
    """Build a mock OpenAI-style completion with a tool call."""
    func_mock = MagicMock()
    func_mock.name = tool_name
    func_mock.arguments = json.dumps(arguments)

    msg = MagicMock()
    msg.content = content
    msg.tool_calls = [
        MagicMock(
            id=call_id,
            function=func_mock,
        )
    ]
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _make_text_response(text):
    """Build a mock completion with no tool calls."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


class TestAgentWithRagTool:
    """Integration tests for agent using the RAG tool (mocked provider)."""

    def test_agent_can_call_search_tool(self):
        """Agent can invoke search_company_knowledge tool via mock provider."""
        mock_provider = MagicMock()

        # First call: model requests search_company_knowledge
        # Second call: model gives final answer
        mock_provider.chat_with_tools.side_effect = [
            # First response: tool call
            _make_tool_call_response(
                "search_company_knowledge",
                {"query": "expense policy", "top_k": 3},
            ),
            # Second response: final answer
            _make_text_response("Found expense policy"),
        ]

        registry = _make_registry_with_rag()
        service = _make_test_service(registry, provider=mock_provider)
        request = AgentRunRequest(message="What is the expense policy?")
        response = service.run(request)

        assert response.final_answer == "Found expense policy"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "search_company_knowledge"
        assert response.tool_calls[0].arguments == {"query": "expense policy", "top_k": 3}

    def test_agent_search_tool_empty_results(self):
        """Agent handles empty search results gracefully."""
        mock_provider = MagicMock()
        mock_provider.chat_with_tools.side_effect = [
            # First response: tool call
            _make_tool_call_response(
                "search_company_knowledge",
                {"query": "unknown topic", "top_k": 5},
            ),
            # Second response: final answer based on empty results
            _make_text_response("No information found"),
        ]

        registry = _make_registry_with_rag()
        service = _make_test_service(registry, provider=mock_provider)
        request = AgentRunRequest(message="What about unknown topic?")
        response = service.run(request)

        assert response.final_answer == "No information found"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].result == []

    def test_agent_search_tool_error_propagates(self):
        """Agent propagates tool errors correctly."""
        mock_provider = MagicMock()
        mock_provider.chat_with_tools.return_value = _make_tool_call_response(
            "search_company_knowledge",
            {"query": "test", "top_k": -1},
        )

        registry = _make_registry_with_rag()
        service = _make_test_service(registry, provider=mock_provider)
        request = AgentRunRequest(message="Search with invalid top_k")

        with pytest.raises(RuntimeError, match="top_k must be positive"):
            service.run(request)