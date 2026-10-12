"""Tests for resource lifecycle and request-state isolation."""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from app.ai.factory import get_provider as get_ai_provider_factory, _create_ai_provider
from app.ai.embedding_factory import get_embedding_provider as get_embedding_provider_factory, _create_embedding_provider
from app.rag.store_factory import get_vector_store as get_vector_store_factory, _create_vector_store
from app.core.singletons import (
    initialize_singletons,
    shutdown_singletons,
    get_ai_provider,
    get_embedding_provider,
    get_vector_store,
)
from app.modules.agent.service import create_agent_service, AgentService
from app.modules.agent.schema import AgentRunRequest
from app.rag.store import InMemoryVectorStore, VectorRecord
from app.ai.embeddings import EmbeddingProvider

# New import for HTTP integration test
from fastapi.testclient import TestClient
from main import app
from unittest.mock import MagicMock


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 4):
        self._dimension = dimension
        self.call_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [[0.1] * self._dimension for _ in texts]


class TestSingletonInitialization:
    """Test singleton initialization and shutdown."""

    def test_initialize_and_shutdown_singletons(self):
        """Test that singletons can be initialized and shutdown."""
        # Initialize
        initialize_singletons()

        # Verify singletons are accessible
        ai_provider = get_ai_provider()
        embedding_provider = get_embedding_provider()
        vector_store = get_vector_store()

        assert ai_provider is not None
        assert embedding_provider is not None
        assert vector_store is not None

        # Shutdown
        shutdown_singletons()

        # Verify singletons are cleared
        with pytest.raises(RuntimeError, match="not initialized"):
            get_ai_provider()
        with pytest.raises(RuntimeError, match="not initialized"):
            get_embedding_provider()
        with pytest.raises(RuntimeError, match="not initialized"):
            get_vector_store()

    def test_double_initialize_is_idempotent(self):
        """Test that calling initialize_singletons twice is safe."""
        initialize_singletons()
        first_ai = get_ai_provider()
        first_embedding = get_embedding_provider()
        first_store = get_vector_store()

        initialize_singletons()  # Should not create new instances

        assert get_ai_provider() is first_ai
        assert get_embedding_provider() is first_embedding
        assert get_vector_store() is first_store

        shutdown_singletons()

    def test_shutdown_without_initialize_is_safe(self):
        """Test that shutdown without initialize is safe."""
        shutdown_singletons()  # Should not raise


class TestFactoryFallbackBehavior:
    """Test factory fallback when singletons not initialized."""

    def test_get_provider_fallback(self):
        """Test get_provider creates new instance when singleton not initialized."""
        shutdown_singletons()  # Ensure clean state

        provider1 = get_ai_provider_factory()
        provider2 = get_ai_provider_factory()

        # Should create new instances each call (fallback behavior)
        assert provider1 is not provider2

    def test_get_embedding_provider_fallback(self):
        """Test get_embedding_provider creates new instance when singleton not initialized."""
        shutdown_singletons()

        provider1 = get_embedding_provider_factory()
        provider2 = get_embedding_provider_factory()

        assert provider1 is not provider2

    def test_get_vector_store_fallback(self):
        """Test get_vector_store creates new instance when singleton not initialized."""
        shutdown_singletons()

        store1 = get_vector_store_factory()
        store2 = get_vector_store_factory()

        assert store1 is not store2

    def test_create_functions_still_work(self):
        """Test that _create_* functions still create new instances."""
        p1 = _create_ai_provider()
        p2 = _create_ai_provider()
        assert p1 is not p2

        e1 = _create_embedding_provider()
        e2 = _create_embedding_provider()
        assert e1 is not e2

        s1 = _create_vector_store()
        s2 = _create_vector_store()
        assert s1 is not s2


class TestSingletonReuseInFastAPIContext:
    """Test singleton reuse when initialized (simulating FastAPI lifespan)."""

    def setup_method(self):
        """Initialize singletons before each test."""
        shutdown_singletons()
        initialize_singletons()

    def teardown_method(self):
        """Shutdown singletons after each test."""
        shutdown_singletons()

    def test_ai_provider_singleton_reuse(self):
        """Test that get_provider returns same instance after initialization."""
        provider1 = get_ai_provider_factory()
        provider2 = get_ai_provider_factory()
        assert provider1 is provider2

    def test_embedding_provider_singleton_reuse(self):
        """Test that get_embedding_provider returns same instance after initialization."""
        provider1 = get_embedding_provider_factory()
        provider2 = get_embedding_provider_factory()
        assert provider1 is provider2

    def test_vector_store_singleton_reuse(self):
        """Test that get_vector_store returns same instance after initialization."""
        store1 = get_vector_store_factory()
        store2 = get_vector_store_factory()
        assert store1 is store2

    def test_concurrent_access_returns_same_instance(self):
        """Test that concurrent access to singletons returns same instances."""
        results = []

        def access_singletons():
            results.append((
                get_ai_provider_factory(),
                get_embedding_provider_factory(),
                get_vector_store_factory(),
            ))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_singletons) for _ in range(50)]
            for f in as_completed(futures):
                f.result()

        # All results should have the same instances
        first_ai, first_embedding, first_store = results[0]
        for ai, embedding, store in results[1:]:
            assert ai is first_ai
            assert embedding is first_embedding
            assert store is first_store


class TestAgentServiceRequestIsolation:
    """Test that AgentService request state is properly isolated."""

    def setup_method(self):
        """Initialize singletons before each test."""
        shutdown_singletons()
        initialize_singletons()

    def teardown_method(self):
        """Shutdown singletons after each test."""
        shutdown_singletons()

    def test_agent_service_uses_singleton_provider(self):
        """Test that AgentService uses singleton AI provider."""
        service = create_agent_service()
        singleton_provider = get_ai_provider()
        assert service._provider is singleton_provider

    def test_agent_service_uses_singleton_embedding_provider(self):
        """Test that AgentService uses singleton embedding provider."""
        service = create_agent_service()
        singleton_embedding = get_embedding_provider()
        assert service.embedder is singleton_embedding

    def test_agent_service_uses_singleton_vector_store(self):
        """Test that AgentService uses singleton vector store."""
        service = create_agent_service()
        singleton_store = get_vector_store()
        assert service.store is singleton_store

    def test_multiple_agent_services_share_infrastructure(self):
        """Test that multiple AgentService instances share infrastructure."""
        service1 = create_agent_service()
        service2 = create_agent_service()

        # Infrastructure should be shared
        assert service1._provider is service2._provider
        assert service1.embedder is service2.embedder
        assert service1.store is service2.store

        # But ToolRegistry should be separate per request
        assert service1._registry is not service2._registry

    def test_request_state_is_isolated(self):
        """Test that request-specific state is isolated between runs."""
        service = create_agent_service()

        # Mock the provider to avoid actual API calls
        from unittest.mock import MagicMock

        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Test answer"
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]

        service._provider.chat_with_tools = MagicMock(return_value=mock_completion)

        # Run first request
        request1 = AgentRunRequest(message="Hello", max_tool_rounds=5)
        response1 = service.run(request1)

        # Run second request with different message
        request2 = AgentRunRequest(message="World", max_tool_rounds=5)
        response2 = service.run(request2)

        # Responses should be independent
        assert response1.final_answer == "Test answer"
        assert response2.final_answer == "Test answer"

        # Each run should have its own message history
        assert service._provider.chat_with_tools.call_count == 2

    def test_tool_call_records_are_request_scoped(self):
        """Test that tool call records don't leak between requests."""
        service = create_agent_service()

        from unittest.mock import MagicMock

        # First request: tool call
        mock_completion1 = MagicMock()
        mock_choice1 = MagicMock()
        mock_message1 = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "count_requirements"
        mock_tool_call.function.arguments = '{"requirements": ["a", "b"]}'
        mock_message1.content = None
        mock_message1.tool_calls = [mock_tool_call]
        mock_choice1.message = mock_message1
        mock_completion1.choices = [mock_choice1]

        # Second response: final answer
        mock_completion2 = MagicMock()
        mock_choice2 = MagicMock()
        mock_message2 = MagicMock()
        mock_message2.content = "Final answer"
        mock_message2.tool_calls = None
        mock_choice2.message = mock_message2
        mock_completion2.choices = [mock_choice2]

        service._provider.chat_with_tools = MagicMock(
            side_effect=[mock_completion1, mock_completion2]
        )

        request = AgentRunRequest(message="Count these", max_tool_rounds=5)
        response = service.run(request)

        # Should have one tool call record
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "count_requirements"

        # Run second request - should not have previous tool calls
        service._provider.chat_with_tools = MagicMock(return_value=mock_completion2)
        request2 = AgentRunRequest(message="Another request", max_tool_rounds=5)
        response2 = service.run(request2)

        assert len(response2.tool_calls) == 0
        assert response2.final_answer == "Final answer"


class TestVectorStoreRequestIsolation:
    """Test that vector store operations are correct with shared store."""

    def setup_method(self):
        """Initialize singletons before each test."""
        shutdown_singletons()
        initialize_singletons()

    def teardown_method(self):
        """Shutdown singletons after each test."""
        shutdown_singletons()

    def test_shared_vector_store_persists_across_requests(self):
        """Test that data written to shared store persists across requests."""
        store = get_vector_store()

        # Clear any existing data
        store.clear()

        # Add records via first "request"
        records1 = [
            VectorRecord(
                id="doc1_chunk_0",
                text="First document",
                embedding=[0.1, 0.2, 0.3, 0.4],
                metadata={"doc_id": "doc1", "chunk_index": 0},
            )
        ]
        store.add(records1)

        # Verify data is accessible via same store instance (simulating second request)
        assert store.count() == 1
        results = store.search([0.1, 0.2, 0.3, 0.4], k=1)
        assert len(results) == 1
        assert results[0].record.text == "First document"

    def test_concurrent_reads_safe(self):
        """Test that concurrent reads from shared store are safe."""
        store = get_vector_store()
        store.clear()

        # Add some data
        records = [
            VectorRecord(
                id=f"doc_{i}_chunk_0",
                text=f"Document {i}",
                embedding=[0.1 * i, 0.2, 0.3, 0.4],
                metadata={"doc_id": f"doc_{i}", "chunk_index": 0},
            )
            for i in range(10)
        ]
        store.add(records)

        # Concurrent reads
        def search_store():
            return store.search([0.5, 0.2, 0.3, 0.4], k=3)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(search_store) for _ in range(50)]
            for f in as_completed(futures):
                results = f.result()
                assert len(results) <= 3  # Should not crash

    def test_inmemory_store_not_threadsafe_for_writes(self):
        """Test that InMemoryVectorStore concurrent writes may have issues (known limitation)."""
        # This test documents the known limitation - InMemoryVectorStore
        # is not thread-safe for concurrent writes. This is acceptable
        # since we'll migrate to PostgreSQL+pgvector for production.
        store = get_vector_store()

        if isinstance(store, InMemoryVectorStore):
            store.clear()

            def add_records():
                records = [
                    VectorRecord(
                        id=f"concurrent_{threading.current_thread().ident}_chunk_0",
                        text="Concurrent write",
                        embedding=[0.1, 0.2, 0.3, 0.4],
                        metadata={"doc_id": f"concurrent_{threading.current_thread().ident}"},
                    )
                ]
                store.add(records)

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(add_records) for _ in range(10)]
                for f in as_completed(futures):
                    f.result()  # May raise or lose data - known limitation


class TestEmbeddingProviderModelCaching:
    """Test that LocalEmbeddingProvider caches model correctly."""

    def setup_method(self):
        """Initialize singletons before each test."""
        shutdown_singletons()
        initialize_singletons()

    def teardown_method(self):
        """Shutdown singletons after each test."""
        shutdown_singletons()

    def test_local_embedding_model_loaded_once(self):
        """Test that LocalEmbeddingProvider loads model only once."""
        provider = get_embedding_provider()

        if hasattr(provider, '_model'):
            # First access to dimension triggers load
            dim1 = provider.dimension
            # Second access should use cached model
            dim2 = provider.dimension
            assert dim1 == dim2
            # Model should be loaded
            assert provider._model is not None

    def test_embedding_provider_reused_across_requests(self):
        """Test that same embedding provider instance is used across requests."""
        service1 = create_agent_service()
        service2 = create_agent_service()

        # Both services should use the same embedding provider instance
        assert service1.embedder is service2.embedder
        assert service1.embedder is get_embedding_provider()


class TestRFPAnalyzeServiceUsesSingleton:
    """Test that RFP analyze service uses singleton provider."""

    def setup_method(self):
        """Initialize singletons before each test."""
        shutdown_singletons()
        initialize_singletons()

    def teardown_method(self):
        """Shutdown singletons after each test."""
        shutdown_singletons()

    def test_analyze_text_uses_singleton(self):
        """Test that analyze_text uses singleton provider."""
        from app.modules.rfp_analyze.service import analyze_text
        from unittest.mock import MagicMock, patch

        singleton_provider = get_ai_provider()

        # Mock the provider's generate_structured
        mock_analysis = MagicMock()
        mock_analysis.title = "Test"
        mock_analysis.client = "Client"
        mock_analysis.deadline = "2024-01-01"
        mock_analysis.budget = "$1000"
        mock_analysis.requirements = ["Req 1"]
        mock_analysis.deliverables = ["Del 1"]
        mock_analysis.evaluation_criteria = ["Crit 1"]
        mock_analysis.submission_requirements = ["Sub 1"]

        singleton_provider.generate_structured = MagicMock(return_value=mock_analysis)

        # Call analyze_text - should use singleton provider
        result = analyze_text("Test RFP content")

        assert result is mock_analysis
        singleton_provider.generate_structured.assert_called_once()


class TestLifespanIntegration:
    """Test lifespan integration with FastAPI."""

    def test_lifespan_initializes_singletons(self):
        """Test that lifespan context manager initializes singletons."""
        from main import lifespan
        from fastapi import FastAPI

        app = FastAPI(lifespan=lifespan)

        # Manually test lifespan
        import asyncio

        async def test():
            async with lifespan(app):
                # Singletons should be initialized
                ai = get_ai_provider()
                emb = get_embedding_provider()
                vs = get_vector_store()
                assert ai is not None
                assert emb is not None
                assert vs is not None
            # After lifespan, singletons should be shutdown
            with pytest.raises(RuntimeError):
                get_ai_provider()

        asyncio.run(test())


class TestHttpIntegration:
    """HTTP integration tests via TestClient."""

    def test_agent_endpoint_via_lifespan(self):
        """Test that the agent endpoint works within the lifespan and uses singletons."""
        # Use TestClient to manage the lifespan
        with TestClient(app) as client:
            # Within this block, the lifespan has started (singletons initialized)
            # Get the singleton AI provider and mock its chat_with_tools method
            ai_provider = get_ai_provider()
            # We'll mock the method to return a fixed response
            mock_completion = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Test answer from agent"
            mock_message.tool_calls = None
            mock_choice.message = mock_message
            mock_completion.choices = [mock_choice]
            ai_provider.chat_with_tools = MagicMock(return_value=mock_completion)

            # Make a request to the agent endpoint
            response = client.post(
                "/api/agent/run",
                json={"message": "Hello", "max_tool_rounds": 5},
            )

            # Verify the response
            assert response.status_code == 200
            data = response.json()
            assert data["final_answer"] == "Test answer from agent"
            assert data["tool_calls"] == []
            assert data["rounds_used"] == 1

            # Verify the mock was called
            assert ai_provider.chat_with_tools.call_count == 1

        # After the TestClient block, the lifespan has shut down (singletons cleared)
        # Verify that we cannot get the provider (optional, but good to confirm)
        with pytest.raises(RuntimeError, match="not initialized"):
            get_ai_provider()