"""Database configuration and connection management."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool


class DatabaseConfig:
    """Database configuration from environment variables."""

    def __init__(self) -> None:
        self._url: Optional[str] = None

    @property
    def url(self) -> str:
        """Get the database URL from environment or use default."""
        if self._url is None:
            self._url = os.environ.get("DATABASE_URL")
            if not self._url:
                # Default for local development
                self._url = "postgresql+asyncpg://postgres:postgres@localhost:5432/proposal_intelligence"
        return self._url

    @property
    def echo(self) -> bool:
        """Enable SQL logging."""
        return os.environ.get("DATABASE_ECHO", "false").lower() == "true"

    @property
    def pool_size(self) -> int:
        """Connection pool size."""
        return int(os.environ.get("DATABASE_POOL_SIZE", "10"))

    @property
    def max_overflow(self) -> int:
        """Maximum overflow connections."""
        return int(os.environ.get("DATABASE_MAX_OVERFLOW", "5"))

    @property
    def pool_recycle(self) -> int:
        """Pool recycle time in seconds."""
        return int(os.environ.get("DATABASE_POOL_RECYCLE", "1800"))

    @property
    def use_null_pool(self) -> bool:
        """Use NullPool for testing environments."""
        return os.environ.get("DATABASE_USE_NULL_POOL", "false").lower() == "true"


_database_config = DatabaseConfig()
_async_engine: Optional[AsyncEngine] = None
_async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_database_config() -> DatabaseConfig:
    """Get the database configuration singleton."""
    return _database_config


def create_async_engine_instance() -> AsyncEngine:
    """Create a new async engine instance."""
    config = get_database_config()
    pool_class = NullPool if config.use_null_pool else None

    kwargs = {
        "url": config.url,
        "echo": config.echo,
        "poolclass": pool_class,
        "future": True,
    }

    if not config.use_null_pool:
        kwargs["pool_size"] = config.pool_size
        kwargs["max_overflow"] = config.max_overflow
        kwargs["pool_recycle"] = config.pool_recycle

    return create_async_engine(**kwargs)


def initialize_database() -> None:
    """Initialize the async engine and session factory."""
    global _async_engine, _async_session_factory

    if _async_engine is not None:
        return

    _async_engine = create_async_engine_instance()
    _async_session_factory = async_sessionmaker(
        _async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def shutdown_database() -> None:
    """Shutdown the database engine."""
    global _async_engine, _async_session_factory

    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_factory = None


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get the async session factory."""
    if _async_session_factory is None:
        raise RuntimeError(
            "Database not initialized. Call initialize_database() at startup."
        )
    return _async_session_factory


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    factory = get_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def is_database_initialized() -> bool:
    """Check if database is initialized."""
    return _async_engine is not None