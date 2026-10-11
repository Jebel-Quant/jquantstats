"""Tests for `Portfolio.from_weights`: sparse target weights, price drift, realised costs.

The simulation is checked against `_reference_backtest`, a deliberately plain
row loop written independently of ``jquantstats._rebalance``: it keeps a cash
balance and a unit count and nothing else, as cvxgrp/simulator and
``qis.backtest_model_portfolio`` do.  Agreement with it to round-off is the
evidence that units, costs and NAV follow the intended recursion.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from jquantstats import Portfolio
from jquantstats.exceptions import (
    NegativeCostBpsError,
    PositionExprColumnError,
    RebalancePriceError,
    RowCountMismatchError,
    WeightsColumnError,
)

_AUM: float = 1e6
_ROWS: int = 120
_EVERY: int = 10
_ASSETS: list[str] = ["A", "B", "C"]


def _reference_backtest(
    prices: np.ndarray, weights: np.ndarray, aum: float, cost_bps: float, compounding: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the book one row at a time with an explicit cash balance.

    Args:
        prices: Price matrix, shape ``(t, n)``.
        weights: Target weights, NaN on every row that does not rebalance.
        aum: Starting cash.
        cost_bps: One-way cost in basis points of traded notional.
        compounding: Size trades off NAV (True) or off *aum* (False).

    Returns:
        Units, per-asset costs and NAV for every row.
    """
    n_rows, n_assets = prices.shape
    units, costs, nav = np.zeros((n_rows, n_assets)), np.zeros((n_rows, n_assets)), np.zeros(n_rows)
    held, cash = np.zeros(n_assets), aum
    for t in range(n_rows):
        if not np.isnan(weights[t]).all():
            capital = cash + held @ prices[t] if compounding else aum
            new = capital * np.nan_to_num(weights[t]) / prices[t]
            traded = (new - held) * prices[t]
            costs[t] = cost_bps / 1e4 * np.abs(traded)
            cash -= traded.sum() + costs[t].sum()
            held = new
        units[t] = held
        nav[t] = cash + held @ prices[t]
    return units, costs, nav


@pytest.fixture
def market() -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray]:
    """A dated three-asset market with targets on every tenth row, 10 % left in cash.

    Returns:
        Prices and sparse weights as frames, and the same two as arrays.
    """
    rng = np.random.default_rng(42)
    price_matrix = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, (_ROWS, 3)), axis=0))
    weight_matrix = np.full((_ROWS, 3), np.nan)
    for t in range(0, _ROWS, _EVERY):
        weight_matrix[t] = rng.dirichlet(np.ones(3)) * 0.9
    dates = pl.date_range(date(2020, 1, 1), date(2020, 1, 1) + timedelta(days=_ROWS - 1), eager=True)
    prices = pl.DataFrame({"date": dates, **{a: price_matrix[:, i] for i, a in enumerate("ABC")}})
    weights = pl.DataFrame({"date": dates, **{a: weight_matrix[:, i] for i, a in enumerate("ABC")}}).fill_nan(None)
    return prices, weights, price_matrix, weight_matrix


# ── Agreement with the reference loop ─────────────────────────────────────────


@pytest.mark.parametrize("compounding", [True, False])
@pytest.mark.parametrize("cost_bps", [0.0, 10.0])
def test_matches_reference_backtest(market, compounding: bool, cost_bps: float) -> None:
    """Units, realised costs and NAV agree with an independent row loop."""
    prices, weights, price_matrix, weight_matrix = market
    pf = Portfolio.from_weights(prices, weights, aum=_AUM, cost_bps=cost_bps, compounding=compounding)
    units, costs, nav = _reference_backtest(price_matrix, weight_matrix, _AUM, cost_bps, compounding)

    np.testing.assert_allclose(pf.units.select(_ASSETS).to_numpy(), units, rtol=1e-12)
    np.testing.assert_allclose(pf.realized_costs.select(_ASSETS).to_numpy(), costs, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(pf.nav_accumulated["NAV_accumulated"].to_numpy(), nav, rtol=1e-12)


def test_costs_lower_nav(market) -> None:
    """Paying costs ends below the cost-free book, by more than the sum of costs."""
    prices, weights, _, _ = market
    gross = Portfolio.from_weights(prices, weights, aum=_AUM)
    net = Portfolio.from_weights(prices, weights, aum=_AUM, cost_bps=25.0)
    paid = net.realized_costs.select(_ASSETS).to_numpy().sum()
    assert paid > 0.0
    assert net.nav_accumulated["NAV_accumulated"][-1] < gross.nav_accumulated["NAV_accumulated"][-1]
    # The portfolio's own cost_bps stays zero, so nothing is charged twice.
    assert net.cost_bps == 0.0


# ── Drift between rebalancings ────────────────────────────────────────────────


def test_units_are_held_between_rebalancings(market) -> None:
    """Units only change on rebalancing rows; weights drift in between."""
    prices, weights, _, weight_matrix = market
    pf = Portfolio.from_weights(prices, weights, aum=_AUM)
    trades = pf.trades_units.select(_ASSETS).to_numpy()
    rebalancing = ~np.isnan(weight_matrix).all(axis=1)
    np.testing.assert_allclose(trades[~rebalancing], 0.0, atol=1e-9)
    drifted = pf.weights.select(_ASSETS).to_numpy()
    np.testing.assert_allclose(drifted[rebalancing], weight_matrix[rebalancing], rtol=1e-12)
    assert not np.allclose(drifted[~rebalancing][:5], weight_matrix[0])


def test_cash_until_first_rebalancing() -> None:
    """Before the first non-null weight row the book is all cash and NAV is flat."""
    prices = pl.DataFrame({"A": [100.0, 120.0, 150.0, 180.0]})
    weights = pl.DataFrame({"A": [None, None, 1.0, None]})
    pf = Portfolio.from_weights(prices, weights, aum=1500.0)
    assert pf.units["A"].to_list() == [0.0, 0.0, 10.0, 10.0]
    assert pf.nav_accumulated["NAV_accumulated"].to_list() == [1500.0, 1500.0, 1500.0, 1800.0]


def test_null_and_missing_weights_on_a_rebalancing_row_mean_zero() -> None:
    """A null inside a rebalancing row, or an absent asset column, is a zero target."""
    prices = pl.DataFrame({"A": [100.0, 100.0], "B": [50.0, 50.0], "C": [10.0, 10.0]})
    weights = pl.DataFrame({"A": [0.5, 1.0], "B": [0.5, None]})
    pf = Portfolio.from_weights(prices, weights, aum=1000.0)
    assert pf.units.to_dict(as_series=False) == {"A": [5.0, 10.0], "B": [10.0, 0.0], "C": [0.0, 0.0]}


def test_expression_weights() -> None:
    """Weights given as an expression are evaluated against prices."""
    prices = pl.DataFrame({"A": [100.0, 200.0, 400.0]})
    expr = pl.when(pl.int_range(pl.len()) == 0).then(1.0).otherwise(None).alias("A")
    pf = Portfolio.from_weights(prices, expr, aum=1000.0)
    assert pf.units["A"].to_list() == [10.0, 10.0, 10.0]


def test_date_column_is_preserved(market) -> None:
    """The date axis carries through to positions and realised costs."""
    prices, weights, _, _ = market
    pf = Portfolio.from_weights(prices, weights, aum=_AUM)
    assert pf.cashposition.columns == ["date", "A", "B", "C"]
    assert pf.realized_costs.columns == ["date", "A", "B", "C"]


def test_price_hole_between_rebalancings_is_tolerated() -> None:
    """A missing price on a non-rebalancing row does not stop the simulation."""
    prices = pl.DataFrame({"A": [100.0, None, 120.0]})
    weights = pl.DataFrame({"A": [1.0, None, None]})
    pf = Portfolio.from_weights(prices, weights, aum=1000.0)
    assert pf.units["A"].to_list() == [10.0, None, 10.0]


# ── Compounding semantics ─────────────────────────────────────────────────────


def test_compounding_returns_reproduce_nav(market) -> None:
    """Compounding the returns rebuilds the accumulated NAV exactly."""
    prices, weights, _, _ = market
    pf = Portfolio.from_weights(prices, weights, aum=_AUM, cost_bps=10.0)
    nav = pf.nav_accumulated["NAV_accumulated"].to_numpy()
    np.testing.assert_allclose(_AUM * np.cumprod(1.0 + pf.returns["returns"].to_numpy()), nav, rtol=1e-12)
    np.testing.assert_allclose(pf.nav_compounded["NAV_compounded"].to_numpy(), nav, rtol=1e-12)


def test_fixed_notional_returns_are_measured_against_aum(market) -> None:
    """With compounding off, returns are profit over the constant aum."""
    prices, weights, _, weight_matrix = market
    pf = Portfolio.from_weights(prices, weights, aum=_AUM, compounding=False)
    np.testing.assert_allclose(pf.returns["returns"].to_numpy(), pf.profit["profit"].to_numpy() / _AUM)
    first = ~np.isnan(weight_matrix[_EVERY])
    sized = pf.cashposition.select(_ASSETS).to_numpy()[_EVERY]
    np.testing.assert_allclose(sized[first], _AUM * weight_matrix[_EVERY][first])


def test_turnover_and_investment_ratio_use_nav_when_compounding() -> None:
    """Turnover divides by the previous NAV, the investment ratio by the current NAV."""
    prices = pl.DataFrame({"A": [100.0, 200.0, 200.0]})
    weights = pl.DataFrame({"A": [0.5, None, 0.25]})
    pf = Portfolio.from_weights(prices, weights, aum=1000.0)
    # NAV: 1000 → 1500 → 1500. Units: 5 → 5 (drift, no trade) → 1.875, so 3.125 sold at 200.
    assert pf.turnover["turnover"].to_list() == [0.0, 0.0, 625.0 / 1500.0]
    assert pf.investment_ratio["investment_ratio"].to_list() == [0.5, 1000.0 / 1500.0, 0.25]


# ── Transforms ────────────────────────────────────────────────────────────────


def test_truncate_starts_from_the_nav_reached(market) -> None:
    """A truncated compounding book keeps the original returns and NAV level."""
    prices, weights, _, _ = market
    pf = Portfolio.from_weights(prices, weights, aum=_AUM, cost_bps=10.0)
    window = pf.truncate(start=_EVERY, end=3 * _EVERY)
    original = pf.returns.slice(_EVERY, 2 * _EVERY + 1)
    assert window.aum == pytest.approx(original["NAV_accumulated"][0])
    np.testing.assert_allclose(window.returns["returns"][1:].to_numpy(), original["returns"][1:].to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(
        window.nav_accumulated["NAV_accumulated"].to_numpy(), original["NAV_accumulated"].to_numpy(), rtol=1e-12
    )
    assert window.realized_costs.select(_ASSETS).row(0) == (0.0, 0.0, 0.0)
    assert window.compounding


def test_truncate_by_date_and_to_nothing(market) -> None:
    """A date window works the same way, and an empty window keeps the aum."""
    prices, weights, _, _ = market
    pf = Portfolio.from_weights(prices, weights, aum=_AUM)
    window = pf.truncate(start=date(2020, 1, 11))
    assert window.aum == pytest.approx(pf.nav_accumulated["NAV_accumulated"][_EVERY])
    empty = pf.truncate(start=date(2030, 1, 1))
    assert empty.prices.height == 0
    assert empty.aum == _AUM


def test_truncate_fixed_notional_keeps_aum(market) -> None:
    """A fixed-notional book truncates against its original aum."""
    prices, weights, _, _ = market
    pf = Portfolio.from_weights(prices, weights, aum=_AUM, compounding=False, cost_bps=5.0)
    assert pf.truncate(start=_EVERY).aum == _AUM


def test_position_transforms_drop_realized_costs(market) -> None:
    """lag, smoothing and attribution trade differently, so the costs do not carry over."""
    prices, weights, _, _ = market
    pf = Portfolio.from_weights(prices, weights, aum=_AUM, cost_bps=10.0)
    for derived in (pf.lag(1), pf.smoothed_holding(2), pf.tilt, pf.timing):
        assert derived.realized_costs is None
        assert derived.compounding


# ── Direct construction with realised costs ───────────────────────────────────


def test_realized_costs_may_omit_assets() -> None:
    """Only the assets present in realized_costs are charged, in profits and on truncate."""
    pf = Portfolio(
        prices=pl.DataFrame({"A": [100.0, 110.0, 121.0], "B": [10.0, 10.0, 10.0]}),
        cashposition=pl.DataFrame({"A": [1000.0, 1100.0, 1210.0], "B": [100.0, 100.0, 100.0]}),
        aum=10_000.0,
        realized_costs=pl.DataFrame({"A": [1.0, 2.0, 3.0]}),
    )
    assert pf.profits["A"].to_list() == pytest.approx([-1.0, 98.0, 107.0])
    assert pf.profits["B"].to_list() == [0.0, 0.0, 0.0]
    assert pf.truncate(start=1).realized_costs["A"].to_list() == [0.0, 3.0]


def test_realized_costs_row_count_is_checked() -> None:
    """A realized_costs frame of the wrong length is rejected and named."""
    with pytest.raises(RowCountMismatchError, match="realized_costs"):
        Portfolio(
            prices=pl.DataFrame({"A": [100.0, 110.0]}),
            cashposition=pl.DataFrame({"A": [0.0, 0.0]}),
            aum=1.0,
            realized_costs=pl.DataFrame({"A": [0.0]}),
        )


# ── Validation ────────────────────────────────────────────────────────────────


def test_rejects_weights_of_the_wrong_length() -> None:
    """Weights must align row for row with prices."""
    with pytest.raises(RowCountMismatchError, match="weights"):
        Portfolio.from_weights(pl.DataFrame({"A": [1.0, 2.0]}), pl.DataFrame({"A": [1.0]}), aum=1.0)


def test_rejects_unknown_weight_columns() -> None:
    """A weight with no price column cannot be traded."""
    with pytest.raises(WeightsColumnError, match="'Z'"):
        Portfolio.from_weights(pl.DataFrame({"A": [1.0]}), pl.DataFrame({"A": [1.0], "Z": [0.0]}), aum=1.0)


def test_rejects_expression_creating_columns() -> None:
    """An expression that aliases to a new column is caught like the other constructors."""
    with pytest.raises(PositionExprColumnError):
        Portfolio.from_weights(pl.DataFrame({"A": [1.0]}), pl.lit(1.0).alias("Z"), aum=1.0)


@pytest.mark.parametrize(
    ("prices", "weights"),
    [
        ({"A": [None, 100.0]}, {"A": [1.0, None]}),  # targeted without a price
        ({"A": [100.0, 0.0]}, {"A": [1.0, 0.0]}),  # held, sold at a zero price
    ],
)
def test_rejects_unpriced_trades(prices, weights) -> None:
    """A held or targeted asset needs a finite, positive price on a rebalancing row."""
    with pytest.raises(RebalancePriceError, match="'A'"):
        Portfolio.from_weights(pl.DataFrame(prices), pl.DataFrame(weights), aum=1000.0)


def test_unheld_untargeted_asset_may_lack_a_price() -> None:
    """An asset that is neither held nor targeted does not need a price."""
    prices = pl.DataFrame({"A": [100.0, 100.0], "B": [None, None]}, schema={"A": pl.Float64, "B": pl.Float64})
    pf = Portfolio.from_weights(prices, pl.DataFrame({"A": [1.0, 1.0], "B": [0.0, 0.0]}), aum=1000.0)
    assert pf.units["A"].to_list() == [10.0, 10.0]


@pytest.mark.parametrize(
    ("cost_bps", "error"),
    [(-1.0, NegativeCostBpsError), (math.nan, ValueError), (True, TypeError), ("5", TypeError)],
)
def test_rejects_bad_cost_bps(cost_bps, error) -> None:
    """cost_bps must be a finite, non-negative number."""
    with pytest.raises(error):
        Portfolio.from_weights(pl.DataFrame({"A": [1.0]}), pl.DataFrame({"A": [1.0]}), aum=1.0, cost_bps=cost_bps)
