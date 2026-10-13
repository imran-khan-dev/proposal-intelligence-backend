"""Unit tests for database configuration and models (no PostgreSQL required)."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure test environment is set before importing app modules
os.environ.setdefault("DATABASE_USE_NULL_POOL", "true")
os.environ.setdefault("DATABASE_ECHO", "false")


class TestDatabaseConfig:
    """Test database configuration."""

    def test_database_config_defaults(self):
        """Test that database config reads environment variables correctly."""
        from app.core.database import DatabaseConfig

        config = DatabaseConfig()

        # Check defaults
        assert config.echo is False
        assert config.pool_size == 10
        assert config.max_overflow == 5
        assert config.pool_recycle == 1800
        assert config.use_null_pool is True  # Set in conftest

    def test_database_config_from_env(self):
        """Test that database config reads custom environment variables."""
        from app.core.database import DatabaseConfig

        # Save original
        original_url = os.environ.get("DATABASE_URL")
        original_pool = os.environ.get("DATABASE_POOL_SIZE")
        original_max_overflow = os.environ.get("DATABASE_MAX_OVERFLOW")
        original_pool_recycle = os.environ.get("DATABASE_POOL_RECYCLE")
        original_echo = os.environ.get("DATABASE_ECHO")
        original_null_pool = os.environ.get("DATABASE_USE_NULL_POOL")

        try:
            os.environ["DATABASE_URL"] = "postgresql+asyncpg://custom:pass@localhost:5432/test"
            os.environ["DATABASE_POOL_SIZE"] = "20"
            os.environ["DATABASE_MAX_OVERFLOW"] = "10"
            os.environ["DATABASE_POOL_RECYCLE"] = "3600"
            os.environ["DATABASE_ECHO"] = "true"
            os.environ["DATABASE_USE_NULL_POOL"] = "false"

            # Need to create new instance to pick up new env vars
            config = DatabaseConfig()
            # Reset cached URL
            config._url = None

            assert config.url == "postgresql+asyncpg://custom:pass@localhost:5432/test"
            assert config.pool_size == 20
            assert config.max_overflow == 10
            assert config.pool_recycle == 3600
            assert config.echo is True
            assert config.use_null_pool is False

        finally:
            # Restore all env vars to their original values
            if original_url:
                os.environ["DATABASE_URL"] = original_url
            else:
                os.environ.pop("DATABASE_URL", None)
            if original_pool:
                os.environ["DATABASE_POOL_SIZE"] = original_pool
            else:
                os.environ.pop("DATABASE_POOL_SIZE", None)
            if original_max_overflow:
                os.environ["DATABASE_MAX_OVERFLOW"] = original_max_overflow
            else:
                os.environ.pop("DATABASE_MAX_OVERFLOW", None)
            if original_pool_recycle:
                os.environ["DATABASE_POOL_RECYCLE"] = original_pool_recycle
            else:
                os.environ.pop("DATABASE_POOL_RECYCLE", None)
            if original_echo:
                os.environ["DATABASE_ECHO"] = original_echo
            else:
                os.environ.pop("DATABASE_ECHO", None)
            if original_null_pool:
                os.environ["DATABASE_USE_NULL_POOL"] = original_null_pool
            else:
                os.environ.pop("DATABASE_USE_NULL_POOL", None)


class TestDatabaseModels:
    """Test SQLAlchemy model definitions."""

    def test_tenant_model(self):
        """Test Tenant model definition."""
        from app.models.tenant import Tenant
        from app.models import Base

        # Check table name
        assert Tenant.__tablename__ == "tenants"

        # Check columns exist
        assert hasattr(Tenant, "id")
        assert hasattr(Tenant, "name")
        assert hasattr(Tenant, "created_at")
        assert hasattr(Tenant, "updated_at")

        # Check relationship
        assert hasattr(Tenant, "documents")

    def test_document_model(self):
        """Test Document model definition."""
        from app.models.document import Document

        # Check table name
        assert Document.__tablename__ == "documents"

        # Check columns exist
        assert hasattr(Document, "id")
        assert hasattr(Document, "tenant_id")
        assert hasattr(Document, "doc_id")
        assert hasattr(Document, "title")
        assert hasattr(Document, "created_at")
        assert hasattr(Document, "updated_at")

        # Check relationships
        assert hasattr(Document, "tenant")
        assert hasattr(Document, "chunks")

    def test_document_chunk_model(self):
        """Test DocumentChunk model definition."""
        from app.models.document_chunk import DocumentChunk

        # Check table name
        assert DocumentChunk.__tablename__ == "document_chunks"

        # Check columns exist
        assert hasattr(DocumentChunk, "id")
        assert hasattr(DocumentChunk, "document_id")
        assert hasattr(DocumentChunk, "chunk_index")
        assert hasattr(DocumentChunk, "text")
        assert hasattr(DocumentChunk, "embedding")
        assert hasattr(DocumentChunk, "metadata")
        assert hasattr(DocumentChunk, "created_at")

        # Check relationship
        assert hasattr(DocumentChunk, "document")

    def test_base_metadata(self):
        """Test that Base metadata includes all tables."""
        from app.models import Base

        table_names = Base.metadata.tables.keys()
        assert "tenants" in table_names
        assert "documents" in table_names
        assert "document_chunks" in table_names


class TestDatabaseInitialization:
    """Test database initialization lifecycle."""

    def test_initialize_and_shutdown_database(self):
        """Test database initialize and shutdown."""
        from app.core.database import (
            initialize_database,
            shutdown_database,
            is_database_initialized,
            get_async_session_factory,
        )

        # Should not be initialized initially
        assert is_database_initialized() is False

        # Initialize
        initialize_database()
        assert is_database_initialized() is True

        # Should be able to get session factory
        factory = get_async_session_factory()
        assert factory is not None

        # Shutdown (async, but we can test the sync part)
        import asyncio
        asyncio.run(shutdown_database())
        assert is_database_initialized() is False

    def test_double_initialize_idempotent(self):
        """Test that double initialization is safe."""
        from app.core.database import (
            initialize_database,
            shutdown_database,
            is_database_initialized,
            get_async_session_factory,
        )

        initialize_database()
        first_factory = get_async_session_factory()

        # Second initialize should not create new engine
        initialize_database()
        second_factory = get_async_session_factory()

        assert first_factory is second_factory

        import asyncio
        asyncio.run(shutdown_database())


class TestDatabaseMigrationUtilities:
    """Test database migration utilities."""

    @patch("subprocess.run")
    def test_run_alembic_upgrade_success(self, mock_run):
        """Test successful alembic upgrade."""
        from app.core.migration import run_alembic_upgrade

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Upgrade successful"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = run_alembic_upgrade("head")

        assert result is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_alembic_upgrade_failure(self, mock_run):
        """Test failed alembic upgrade."""
        from app.core.migration import run_alembic_upgrade

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Migration failed"
        mock_run.return_value = mock_result

        result = run_alembic_upgrade("head")

        assert result is False

    @patch("subprocess.run")
    def test_get_current_revision(self, mock_run):
        """Test getting current revision."""
        from app.core.migration import get_current_revision

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "001 (head)"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        revision = get_current_revision()

        assert revision == "001 (head)"