"""Pytest configuration for Proposal Intelligence tests."""

import os
import sys

# Set test environment variables before any imports
os.environ.setdefault("AI_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "test-model")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
os.environ.setdefault("VECTOR_STORE_TYPE", "memory")

# Database test configuration
# Use NullPool to avoid connection pooling issues in tests
# Use test database URL with asyncpg driver for integration tests
# For unit tests, this will be overridden by individual test modules if needed
os.environ.setdefault("DATABASE_USE_NULL_POOL", "true")
os.environ.setdefault("DATABASE_ECHO", "false")
# Default test database URL - use asyncpg driver
# This can be overridden by TEST_DATABASE_URL for integration tests
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:45.86.86.86@localhost:5432/proposal_intelligence_test")

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Pytest markers and asyncio configuration
def pytest_configure(config):
    """Register custom markers and configure async test support."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests requiring PostgreSQL")
    config.addinivalue_line("markers", "unit: marks tests as unit tests not requiring external services")
    # Enable asyncio auto mode so async fixtures and tests are handled automatically.
    # pytest-asyncio registers both command-line and ini versions of asyncio_mode;
    # setting config.option is sufficient for the mode.
    if hasattr(config.option, "asyncio_mode"):
        config.option.asyncio_mode = "auto"