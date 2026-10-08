import numpy as np

from src.inventory import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_eoq
)


def test_safety_stock():
    result = calculate_safety_stock(
        demand_std=20
    )

    assert result > 0


def test_safety_stock_increases_with_demand_variability():
    low_variability = calculate_safety_stock(
        demand_std=10
    )

    high_variability = calculate_safety_stock(
        demand_std=30
    )

    assert high_variability > low_variability


def test_reorder_point():
    safety_stock = calculate_safety_stock(
        demand_std=20
    )

    result = calculate_reorder_point(
        average_demand=100,
        safety_stock=safety_stock
    )

    assert result > 700


def test_reorder_point_increases_with_demand():
    safety_stock = calculate_safety_stock(
        demand_std=20
    )

    low_demand = calculate_reorder_point(
        average_demand=50,
        safety_stock=safety_stock
    )

    high_demand = calculate_reorder_point(
        average_demand=100,
        safety_stock=safety_stock
    )

    assert high_demand > low_demand


def test_eoq():
    result = calculate_eoq(
        annual_demand=36500
    )

    assert result > 0


def test_zero_demand_eoq():
    result = calculate_eoq(
        annual_demand=0
    )

    assert result == 0


def test_eoq_increases_with_demand():
    low_demand = calculate_eoq(
        annual_demand=10000
    )

    high_demand = calculate_eoq(
        annual_demand=40000
    )

    assert high_demand > low_demand