"""Tests for the builtin tools."""
import pytest

from app.agent.builtin_tools import calculate_total, count_requirements


class TestCountRequirements:
    def test_count_simple_list(self):
        result = count_requirements(requirements=["a", "b", "c"])
        assert result == 3

    def test_count_empty_list(self):
        result = count_requirements(requirements=[])
        assert result == 0

    def test_count_single_item(self):
        result = count_requirements(requirements=["only one"])
        assert result == 1

    def test_count_large_list(self):
        items = [f"item_{i}" for i in range(100)]
        result = count_requirements(requirements=items)
        assert result == 100

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError):
            count_requirements(requirements="not a list")

    def test_invalid_type_int_raises_type_error(self):
        with pytest.raises(TypeError):
            count_requirements(requirements=42)


class TestCalculateTotal:
    def test_single_item(self):
        items = [
            {"description": "Widget", "quantity": 3, "unit_price": 10.0}
        ]
        assert calculate_total(items) == 30.0

    def test_multiple_items(self):
        items = [
            {"description": "A", "quantity": 2, "unit_price": 5.0},
            {"description": "B", "quantity": 1, "unit_price": 10.0},
        ]
        assert calculate_total(items) == 20.0

    def test_empty_list(self):
        assert calculate_total(items=[]) == 0.0

    def test_integer_quantities_and_prices(self):
        items = [
            {"description": "X", "quantity": 4, "unit_price": 15}
        ]
        assert calculate_total(items) == 60.0

    def test_float_precision(self):
        items = [
            {"description": "Y", "quantity": 3, "unit_price": 3.33}
        ]
        result = calculate_total(items)
        assert abs(result - 9.99) < 0.01

    def test_zero_quantity(self):
        items = [
            {"description": "Free", "quantity": 0, "unit_price": 100.0}
        ]
        assert calculate_total(items) == 0.0

    def test_zero_price(self):
        items = [
            {"description": "Free item", "quantity": 5, "unit_price": 0.0}
        ]
        assert calculate_total(items) == 0.0

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError):
            calculate_total(items="not a list")

    def test_item_not_dict_raises_type_error(self):
        with pytest.raises(TypeError):
            calculate_total(items=["not a dict"])

    def test_missing_quantity_raises_type_error(self):
        items = [{"description": "X", "unit_price": 10.0}]
        with pytest.raises(TypeError):
            calculate_total(items=items)

    def test_missing_unit_price_raises_type_error(self):
        items = [{"description": "X", "quantity": 1}]
        with pytest.raises(TypeError):
            calculate_total(items=items)

    def test_negative_quantity_raises_value_error(self):
        items = [
            {"description": "X", "quantity": -1, "unit_price": 10.0}
        ]
        with pytest.raises(ValueError):
            calculate_total(items=items)

    def test_negative_price_raises_value_error(self):
        items = [
            {"description": "X", "quantity": 1, "unit_price": -5.0}
        ]
        with pytest.raises(ValueError):
            calculate_total(items=items)

    def test_bool_quantity_rejected(self):
        items = [
            {"description": "X", "quantity": True, "unit_price": 10.0}
        ]
        with pytest.raises(TypeError):
            calculate_total(items=items)