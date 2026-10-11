"""Simple tools for the tool-calling demonstration."""

from typing import Any

from app.ai.embeddings import EmbeddingProvider
from app.rag.retrieve import retrieve
from app.rag.store import VectorStore


def count_requirements(requirements: list[str]) -> int:
    """Count the number of requirements in a list.

    Args:
        requirements: A list of requirement strings.

    Returns:
        The number of items in the list.
    """
    if not isinstance(requirements, list):
        raise TypeError(
            f"requirements must be a list, got {type(requirements).__name__}"
        )
    return len(requirements)


def calculate_total(items: list[dict]) -> float:
    """Calculate the total cost of a list of line items.

    Each item must be a dict with 'description', 'quantity', and 'unit_price'.
    quantity and unit_price must be non-negative numbers.

    Args:
        items: A list of dicts with keys:
            - description (str): item description
            - quantity (int or float): number of units
            - unit_price (int or float): price per unit

    Returns:
        The sum of quantity * unit_price for all items.
    """
    if not isinstance(items, list):
        raise TypeError(
            f"items must be a list, got {type(items).__name__}"
        )

    total = 0.0
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(
                f"items[{i}] must be a dict, got {type(item).__name__}"
            )

        quantity = item.get("quantity")
        unit_price = item.get("unit_price")

        if not isinstance(quantity, (int, float)) or isinstance(quantity, bool):
            raise TypeError(
                f"items[{i}].quantity must be a number, got "
                f"{type(quantity).__name__}"
            )
        if not isinstance(unit_price, (int, float)) or isinstance(unit_price, bool):
            raise TypeError(
                f"items[{i}].unit_price must be a number, got "
                f"{type(unit_price).__name__}"
            )

        if quantity < 0:
            raise ValueError(
                f"items[{i}].quantity must be non-negative, got {quantity}"
            )
        if unit_price < 0:
            raise ValueError(
                f"items[{i}].unit_price must be non-negative, got {unit_price}"
            )

        total += float(quantity) * float(unit_price)

    return round(total, 2)


def create_search_company_knowledge_tool(
    embedder: EmbeddingProvider,
    store: VectorStore,
):
    """Create a search_company_knowledge tool handler with injected dependencies.

    Args:
        embedder: Embedding provider for query embedding.
        store: Vector store to search (shared instance).

    Returns:
        A handler function with signature (query: str, top_k: int = 5) -> list[dict].
    """

    def search_company_knowledge(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search the company knowledge base for relevant information.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return (default: 5).

        Returns:
            A list of search results, each containing:
            - text: The chunk text
            - score: Similarity score (0.0 to 1.0)
            - metadata: Document metadata including doc_id and chunk_index
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        results = retrieve(query=query, embedder=embedder, store=store, top_k=top_k)

        # Convert SearchResult objects to JSON-serializable dicts
        return [
            {
                "text": result.record.text,
                "score": result.score,
                "metadata": result.record.metadata,
            }
            for result in results
        ]

    return search_company_knowledge