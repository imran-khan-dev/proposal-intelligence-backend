"""Simple tools for the tool-calling demonstration."""


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