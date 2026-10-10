"""Factory for creating vector stores based on environment configuration."""

import os

from app.rag.store import InMemoryVectorStore, PersistentVectorStore, VectorStore


def get_vector_store() -> VectorStore:
    """Select and instantiate a vector store based on environment configuration.

    Supported stores:
    - memory: In-memory vector store (default)
    - persistent: File-based persistent vector store using JSONL

    Environment variables:
    - VECTOR_STORE_TYPE: "memory" or "persistent" (default: "memory")
    - VECTOR_STORE_PATH: Path for persistent store (default: ./data/vector_store.jsonl)

    Returns:
        A VectorStore instance.

    Raises:
        RuntimeError: If store type is unknown or persistent store initialization fails.
    """
    store_type = (
        os.environ.get("VECTOR_STORE_TYPE", "memory")
        .strip()
        .strip('"')
        .strip("'")
        .lower()
    )

    if store_type == "memory":
        return InMemoryVectorStore()

    if store_type == "persistent":
        path = os.environ.get("VECTOR_STORE_PATH")
        if not path:
            # Default to a local data directory
            path = os.path.join(os.getcwd(), "data", "vector_store.jsonl")
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return PersistentVectorStore(path)

    raise RuntimeError(
        f"Invalid VECTOR_STORE_TYPE: '{store_type}'. "
        "Must be 'memory' or 'persistent'."
    )