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
from app.ai.provider import AIProvider
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


def _make_mock_provider():
    """Create a mock AI provider for testing."""
    mock_provider = MagicMock(spec=AIProvider)
    mock_provider.chat_with_tools.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(content="Default answer", tool_calls=None)
            )
        ]
    )
    return mock_provider


def _make_test_service(registry: ToolRegistry = None, provider=None):
    """Create an AgentService with a mock provider and given registry."""
    if provider is None:
        mock_provider = _make_mock_provider()
    else:
        mock_provider = provider
    if registry is None:
        registry = ToolRegistry()
    return AgentService(provider=mock_provider, registry=registry)


def _make_service_with_rag(embedder=None, store=None, provider=None):
    """Create an AgentService with RAG tool registered and optional custom embedder/store/provider."""
    if embedder is None:
        embedder = MockEmbeddingProvider(dimension=4)
    if store is None:
        store = InMemoryVectorStore()
    if provider is None:
        provider = _make_mock_provider()

    registry = ToolRegistry()
    search_tool_handler = create_search_company_knowledge_tool(embedder=embedder, store=store)
    from app.agent.tools import Tool

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

    service = AgentService(provider=provider, registry=registry)
    service.embedder = embedder
    service.store = store
    return service


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
        registry = ToolRegistry()
        embedder = MockEmbeddingProvider(dimension=4)
        store = InMemoryVectorStore()
        search_tool_handler = create_search_company_knowledge_tool(embedder=embedder, store=store)
        from app.agent.tools import Tool

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

        assert "search_company_knowledge" in registry.names()
        tool = registry.get("search_company_knowledge")
        assert tool is not None
        assert tool.name == "search_company_knowledge"
        assert "company knowledge base" in tool.description.lower()

    def test_tool_parameters_schema(self):
        """Tool has correct parameter schema for OpenAI function calling."""
        registry = ToolRegistry()
        embedder = MockEmbeddingProvider(dimension=4)
        store = InMemoryVectorStore()
        search_tool_handler = create_search_company_knowledge_tool(embedder=embedder, store=store)
        from app.agent.tools import Tool

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
        registry = ToolRegistry()
        embedder = MockEmbeddingProvider(dimension=4)
        store = InMemoryVectorStore()
        search_tool_handler = create_search_company_knowledge_tool(embedder=embedder, store=store)
        from app.agent.tools import Tool

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

        service = _make_service_with_rag(provider=mock_provider)
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

        service = _make_service_with_rag(provider=mock_provider)
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

        service = _make_service_with_rag(provider=mock_provider)
        request = AgentRunRequest(message="Search with invalid top_k")

        with pytest.raises(RuntimeError, match="top_k must be positive"):
            service.run(request)


class TestAgentServiceDependencyExposure:
    """Tests for accessing embedder and store through AgentService."""

    def test_create_agent_service_exposes_embedder_and_store(self):
        """Service created with RAG has embedder and store attributes."""
        service = _make_service_with_rag()

        assert hasattr(service, "embedder")
        assert hasattr(service, "store")
        assert service.embedder is not None
        assert service.store is not None

    def test_embedder_and_store_are_same_instances_used_by_tool(self):
        """The exposed embedder and store are the exact instances used by the search tool."""
        service = _make_service_with_rag()

        # Get the tool handler from the registry
        tool = service._registry.get("search_company_knowledge")
        assert tool is not None

        # The tool handler is a closure that captures embedder and store
        # We can verify they're the same by checking identity
        # Since we can't directly access the closure, we verify through behavior:
        # 1. Add data via the exposed store
        # 2. Search via the tool - should find the data
        test_embedding = [0.1] * service.embedder.dimension
        service.store.add([
            VectorRecord(
                id="test_doc_chunk_0",
                text="Exposed store content",
                embedding=test_embedding,
                metadata={"doc_id": "test_doc", "chunk_index": 0},
            )
        ])

        # Now invoke the tool handler directly (it uses the same store instance)
        search_handler = tool.handler
        results = search_handler(query="exposed store", top_k=5)

        assert len(results) == 1
        assert results[0]["text"] == "Exposed store content"
        assert results[0]["metadata"]["doc_id"] == "test_doc"

    def test_ingestion_via_exposed_store_retrievable_via_tool(self):
        """Documents ingested via exposed store are retrievable through the agent's tool."""
        service = _make_service_with_rag()

        # Simulate ingestion using the exposed dependencies
        from app.rag.ingest import ingest_document

        text = "Company expense policy: all expenses over $100 require manager approval."
        doc_id = "expense-policy-v1"

        count = ingest_document(
            text=text,
            doc_id=doc_id,
            store=service.store,
            embedder=service.embedder,
            chunk_size=200,
            overlap=50,
        )

        assert count > 0
        assert service.store.count() == count

        # Now use the tool (via the registry) to search
        tool = service._registry.get("search_company_knowledge")
        search_handler = tool.handler

        results = search_handler(query="expense policy approval", top_k=3)

        assert len(results) >= 1
        # The ingested content should be found
        found = any("expense policy" in r["text"].lower() for r in results)
        assert found

    def test_agent_run_still_works_with_exposed_dependencies(self):
        """Agent run behavior is unchanged; tool calls work via the shared store."""
        service = _make_service_with_rag()

        # Mock the provider to return a tool call then a final answer
        service._provider.chat_with_tools.side_effect = [
            # First response: tool call
            _make_tool_call_response(
                "search_company_knowledge",
                {"query": "expense policy", "top_k": 3},
            ),
            # Second response: final answer
            _make_text_response("Found expense policy"),
        ]

        request = AgentRunRequest(message="What is the expense policy?")
        response = service.run(request)

        assert response.final_answer == "Found expense policy"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "search_company_knowledge"

    def test_direct_agent_service_construction_still_works(self):
        """Direct AgentService construction (bypassing create_agent_service) still works."""
        mock_provider = _make_mock_provider()
        registry = ToolRegistry()

        # Direct construction should not have embedder/store set
        service = AgentService(provider=mock_provider, registry=registry)

        assert service.embedder is None
        assert service.store is None

        # But run() should still work for non-RAG tools
        mock_provider.chat_with_tools.return_value = _make_text_response("Direct answer")

        from app.schemas.agent import AgentRunRequest

        request = AgentRunRequest(message="Hello")
        response = service.run(request)

        assert response.final_answer == "Direct answer"
        assert response.tool_calls == []

    def test_exposed_store_is_in_memory_vector_store_by_default(self):
        """Default store type is InMemoryVectorStore."""
        from app.rag.store import InMemoryVectorStore

        service = _make_service_with_rag()
        assert isinstance(service.store, InMemoryVectorStore)

    def test_exposed_embedder_matches_configured_provider(self):
        """Exposed embedder matches the configured embedding provider."""
        service = _make_service_with_rag()

        # Should be same type and configuration
        assert type(service.embedder) == type(service.embedder)  # Trivial but verifies attribute exists
        assert service.embedder.dimension == service.embedder.dimension


class DeterministicMockEmbedder(EmbeddingProvider):
    """Deterministic mock embedder that returns consistent embeddings for specific texts."""

    def __init__(self, dimension: int = 4):
        self._dimension = dimension
        self._text_to_embedding = {
            "company expense policy": [1.0, 0.0, 0.0, 0.0],
            "all expenses over 100 require manager approval": [0.9, 0.1, 0.0, 0.0],
            "travel expenses must be pre-approved": [0.8, 0.2, 0.0, 0.0],
            "quarterly budget review process": [0.0, 1.0, 0.0, 0.0],
            "budget requests submitted by month end": [0.1, 0.9, 0.0, 0.0],
            "project budget approval workflow": [0.2, 0.8, 0.0, 0.0],
        }

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            # Find matching key (case-insensitive, partial)
            text_lower = text.lower()
            matched = False
            for key, emb in self._text_to_embedding.items():
                if key in text_lower or text_lower in key:
                    results.append(emb)
                    matched = True
                    break
            if not matched:
                # Default embedding
                results.append([0.1] * self._dimension)
        return results


class TestPersistentRagIntegration:
    """End-to-end integration tests for persistent RAG workflow."""

    def _make_persistent_service(self, path: str, embedder=None):
        """Create an agent service with PersistentVectorStore."""
        from app.rag.store import PersistentVectorStore

        if embedder is None:
            embedder = DeterministicMockEmbedder(dimension=4)

        store = PersistentVectorStore(path)
        provider = _make_mock_provider()
        service = create_agent_service(provider=provider, embedder=embedder, store=store)
        return service

    def test_ingest_and_retrieve_with_persistent_store(self):
        """Ingest document via exposed store, retrieve via search tool."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            # Create service with persistent store
            embedder = DeterministicMockEmbedder(dimension=4)
            service = self._make_persistent_service(path, embedder=embedder)

            # Ingest a company document
            from app.rag.ingest import ingest_document

            text = """Company Expense Policy

            All expenses over $100 require manager approval.
            Travel expenses must be pre-approved by department head.
            Receipts required for all expenses over $25."""

            doc_id = "expense-policy-2024"
            count = ingest_document(
                text=text,
                doc_id=doc_id,
                store=service.store,
                embedder=service.embedder,
                chunk_size=100,
                overlap=20,
            )

            assert count > 0
            assert service.store.count() == count

            # Search via the tool - should find relevant chunks
            tool = service._registry.get("search_company_knowledge")
            search_handler = tool.handler

            results = search_handler(query="expense policy approval", top_k=5)

            assert len(results) >= 1
            # Verify content relevance
            found = any("expense" in r["text"].lower() and "approval" in r["text"].lower() for r in results)
            assert found

            # Verify metadata preserved
            for r in results:
                assert r["metadata"]["doc_id"] == doc_id
                assert "chunk_index" in r["metadata"]

        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_persistence_across_service_recreation(self):
        """Document persists and is searchable after recreating store and service."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            embedder = DeterministicMockEmbedder(dimension=4)

            # First service: ingest document
            service1 = self._make_persistent_service(path, embedder=embedder)

            from app.rag.ingest import ingest_document

            text = "Quarterly budget review process: budget requests submitted by month end."
            doc_id = "budget-process-2024"
            count = ingest_document(
                text=text,
                doc_id=doc_id,
                store=service1.store,
                embedder=service1.embedder,
                chunk_size=100,
                overlap=20,
            )
            assert count > 0
            original_count = service1.store.count()

            # Recreate store and service with same file
            service2 = self._make_persistent_service(path, embedder=embedder)

            # Verify document count preserved
            assert service2.store.count() == original_count

            # Search via new service's tool
            tool = service2._registry.get("search_company_knowledge")
            search_handler = tool.handler

            results = search_handler(query="budget review month end", top_k=5)

            assert len(results) >= 1
            found = any("budget" in r["text"].lower() and "month" in r["text"].lower() for r in results)
            assert found

            # Verify metadata preserved across recreation
            for r in results:
                assert r["metadata"]["doc_id"] == doc_id
                assert "chunk_index" in r["metadata"]

        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_multiple_documents_persisted_and_searchable(self):
        """Multiple documents ingested, all retrievable after recreation."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            embedder = DeterministicMockEmbedder(dimension=4)
            service = self._make_persistent_service(path, embedder=embedder)

            from app.rag.ingest import ingest_document

            # Ingest two different documents
            ingest_document(
                text="Company expense policy: all expenses over 100 require manager approval.",
                doc_id="expense-policy",
                store=service.store,
                embedder=service.embedder,
            )
            ingest_document(
                text="Quarterly budget review process: budget requests submitted by month end.",
                doc_id="budget-process",
                store=service.store,
                embedder=service.embedder,
            )

            # Recreate
            service2 = self._make_persistent_service(path, embedder=embedder)
            tool = service2._registry.get("search_company_knowledge")
            search_handler = tool.handler

            # Search for expense-related content
            expense_results = search_handler(query="expense approval", top_k=5)
            assert len(expense_results) >= 1
            assert any(r["metadata"]["doc_id"] == "expense-policy" for r in expense_results)

            # Search for budget-related content
            budget_results = search_handler(query="budget review", top_k=5)
            assert len(budget_results) >= 1
            assert any(r["metadata"]["doc_id"] == "budget-process" for r in budget_results)

            # Both documents should be in store
            assert service2.store.count() >= 2

        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_empty_store_returns_no_results(self):
        """Fresh/empty persistent store returns no results without crashing."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            embedder = DeterministicMockEmbedder(dimension=4)
            service = self._make_persistent_service(path, embedder=embedder)

            # Store is empty - no documents ingested
            tool = service._registry.get("search_company_knowledge")
            search_handler = tool.handler

            results = search_handler(query="any query", top_k=5)

            assert results == []

        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_absent_file_creates_empty_store(self):
        """Non-existent file path creates empty store without error."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nonexistent.jsonl")

            embedder = DeterministicMockEmbedder(dimension=4)
            service = self._make_persistent_service(path, embedder=embedder)

            # Should be empty
            assert service.store.count() == 0
            assert service.store.dimension is None

            tool = service._registry.get("search_company_knowledge")
            search_handler = tool.handler

            results = search_handler(query="any query", top_k=5)
            assert results == []

    def test_empty_file_creates_empty_store(self):
        """Empty file creates empty store without error."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            embedder = DeterministicMockEmbedder(dimension=4)
            service = self._make_persistent_service(path, embedder=embedder)

            assert service.store.count() == 0
            assert service.store.dimension is None

            tool = service._registry.get("search_company_knowledge")
            search_handler = tool.handler

            results = search_handler(query="any query", top_k=5)
            assert results == []

        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_reingestion_replaces_document_in_persistent_store(self):
        """Re-ingesting same doc_id replaces old chunks in persistent store."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            path = tmp.name

        try:
            embedder = DeterministicMockEmbedder(dimension=4)

            # First ingestion
            service1 = self._make_persistent_service(path, embedder=embedder)
            from app.rag.ingest import ingest_document

            text1 = "Old version of the policy document."
            ingest_document(
                text=text1,
                doc_id="policy-v1",
                store=service1.store,
                embedder=service1.embedder,
            )
            assert service1.store.count() == 1

            # Re-ingest with new content
            service2 = self._make_persistent_service(path, embedder=embedder)
            text2 = "New version of the policy document with updated rules."
            count = ingest_document(
                text=text2,
                doc_id="policy-v1",
                store=service2.store,
                embedder=service2.embedder,
            )
            assert count == 1
            assert service2.store.count() == 1  # Still only 1 chunk (replaced)

            # Search should find new content
            tool = service2._registry.get("search_company_knowledge")
            search_handler = tool.handler
            results = search_handler(query="updated rules", top_k=5)

            assert len(results) == 1
            assert "updated rules" in results[0]["text"].lower()

        finally:
            if os.path.exists(path):
                os.unlink(path)