"""Tests for CostModel abstraction.

Verifies that:
- CostModel named constructors produce the expected field values.
- Negative values are rejected.
- Portfolio factory methods accept cost_model and that it takes precedence over
  individual cost_per_unit / cost_bps parameters.
"""

from __future__ import annotations

import dataclasses
from datetime import date

import polars as pl
import pytest

from jquantstats import CostModel, Portfolio

# ── CostModel construction ────────────────────────────────────────────────────


def test_cost_model_per_unit():
    """CostModel.per_unit sets cost_per_unit and leaves cost_bps at zero."""
    cm = CostModel.per_unit(0.01)
    assert cm.cost_per_unit == pytest.approx(0.01)
    assert cm.cost_bps == pytest.approx(0.0)


def test_cost_model_turnover_bps():
    """CostModel.turnover_bps sets cost_bps and leaves cost_per_unit at zero."""
    cm = CostModel.turnover_bps(5.0)
    assert cm.cost_per_unit == pytest.approx(0.0)
    assert cm.cost_bps == pytest.approx(5.0)


def test_cost_model_zero():
    """CostModel.zero returns a model with both parameters at zero."""
    cm = CostModel.zero()
    assert cm.cost_per_unit == pytest.approx(0.0)
    assert cm.cost_bps == pytest.approx(0.0)


def test_cost_model_direct_construction():
    """CostModel rejects simultaneous non-zero cost_per_unit and cost_bps."""
    with pytest.raises(
        ValueError,
        match=(
            r"^Only one cost model may be active at a time: "
            r"got cost_per_unit=0\.02 and cost_bps=3\.0\. "
            r"Use CostModel\.per_unit\(\) or CostModel\.turnover_bps\(\) to make intent explicit\.$"
        ),
    ):
        CostModel(cost_per_unit=0.02, cost_bps=3.0)


def test_cost_model_rejects_fractional_bps_alongside_per_unit():
    """The both-models-active check triggers for any positive cost_bps, including values below 1."""
    with pytest.raises(ValueError, match="Only one cost model"):
        CostModel(cost_per_unit=0.02, cost_bps=0.5)


def test_cost_model_direct_construction_single_field():
    """CostModel can be constructed directly with a single non-zero field."""
    cm_unit = CostModel(cost_per_unit=0.02)
    assert cm_unit.cost_per_unit == pytest.approx(0.02)
    assert cm_unit.cost_bps == pytest.approx(0.0)

    cm_bps = CostModel(cost_bps=3.0)
    assert cm_bps.cost_per_unit == pytest.approx(0.0)
    assert cm_bps.cost_bps == pytest.approx(3.0)


def test_cost_model_negative_cost_per_unit_raises():
    """CostModel rejects negative cost_per_unit."""
    with pytest.raises(ValueError, match=r"^cost_per_unit must be non-negative, got -0\.01$"):
        CostModel(cost_per_unit=-0.01)


def test_cost_model_negative_cost_bps_raises():
    """CostModel rejects negative cost_bps."""
    with pytest.raises(ValueError, match=r"^cost_bps must be non-negative, got -1\.0$"):
        CostModel(cost_bps=-1.0)


def test_cost_model_frozen():
    """CostModel is immutable (frozen dataclass)."""
    cm = CostModel.per_unit(0.01)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cm.cost_per_unit = 0.02  # type: ignore[misc]


def test_cost_model_repr():
    """CostModel repr contains both field values."""
    cm = CostModel.per_unit(0.01)
    r = repr(cm)
    assert "0.01" in r
    assert "0.0" in r


# ── Portfolio integration ─────────────────────────────────────────────────────


@pytest.fixture
def simple_prices_and_positions():
    """Three-day single-asset price and position frames."""
    dates = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
    prices = pl.DataFrame({"date": dates, "A": [100.0, 110.0, 121.0]})
    positions = pl.DataFrame({"date": dates, "A": [1000.0, 1200.0, 900.0]})
    return prices, positions


def test_from_cash_position_with_cost_model_per_unit(simple_prices_and_positions):
    """from_cash_position honours cost_model=CostModel.per_unit(...)."""
    prices, positions = simple_prices_and_positions
    pf = Portfolio.from_cash_position(
        prices=prices, cash_position=positions, aum=1e5, cost_model=CostModel.per_unit(0.01)
    )
    assert pf.cost_per_unit == pytest.approx(0.01)
    assert pf.cost_bps == pytest.approx(0.0)


def test_from_cash_position_with_cost_model_turnover_bps(simple_prices_and_positions):
    """from_cash_position honours cost_model=CostModel.turnover_bps(...)."""
    prices, positions = simple_prices_and_positions
    pf = Portfolio.from_cash_position(
        prices=prices, cash_position=positions, aum=1e5, cost_model=CostModel.turnover_bps(5.0)
    )
    assert pf.cost_per_unit == pytest.approx(0.0)
    assert pf.cost_bps == pytest.approx(5.0)


def test_cost_model_takes_precedence_over_raw_params(simple_prices_and_positions):
    """cost_model takes precedence over raw cost_per_unit / cost_bps params."""
    prices, positions = simple_prices_and_positions
    pf = Portfolio.from_cash_position(
        prices=prices,
        cash_position=positions,
        aum=1e5,
        cost_per_unit=99.0,  # should be overridden
        cost_bps=99.0,  # should be overridden
        cost_model=CostModel.per_unit(0.01),
    )
    assert pf.cost_per_unit == pytest.approx(0.01)
    assert pf.cost_bps == pytest.approx(0.0)


def test_from_cash_position_backward_compat_raw_params(simple_prices_and_positions):
    """Raw cost_per_unit / cost_bps params still work without cost_model."""
    prices, positions = simple_prices_and_positions
    pf = Portfolio.from_cash_position(
        prices=prices, cash_position=positions, aum=1e5, cost_per_unit=0.005, cost_bps=2.0
    )
    assert pf.cost_per_unit == pytest.approx(0.005)
    assert pf.cost_bps == pytest.approx(2.0)


def test_cost_model_affects_position_delta_costs(simple_prices_and_positions):
    """Portfolio built with CostModel.per_unit produces identical position_delta_costs as raw param."""
    prices, positions = simple_prices_and_positions
    pf_model = Portfolio.from_cash_position(
        prices=prices, cash_position=positions, aum=1e5, cost_model=CostModel.per_unit(0.01)
    )
    pf_raw = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5, cost_per_unit=0.01)
    assert pf_model.position_delta_costs["cost"].to_list() == pf_raw.position_delta_costs["cost"].to_list()
