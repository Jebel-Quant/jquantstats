"""Tests for Portfolio cost-related functionality.

Covers: cost_adjusted_returns, trading_cost_impact, position_delta_costs,
net_cost_nav, cost_per_unit field, from_position factory, and cost-parameter
forwarding through transforms.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
import polars.testing as pt
import pytest

from jquantstats import CostModel, Portfolio

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def cost_portfolio():
    """3-day, single-asset portfolio for position-delta cost tests.

    Position sequence:  0 → 1000 → 700
    |Δ position|:       0,  1000,   300
    cost_per_unit = 0.01
    Expected daily costs: [0, 1000, 300] * 0.01 = [0.0, 10.0, 3.0]
    """
    n = 3
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    return Portfolio(
        prices=pl.DataFrame({"date": dates, "A": pl.Series([100.0, 110.0, 121.0])}),
        cashposition=pl.DataFrame({"date": dates, "A": pl.Series([0.0, 1000.0, 700.0])}),
        aum=1e5,
        cost_per_unit=0.01,
    )


@pytest.fixture
def cost_pf():
    """3-day portfolio with cost_per_unit=0.05 for transform-forwarding tests."""
    prices = pl.DataFrame({"A": [100.0, 110.0, 105.0]})
    pos = pl.DataFrame({"A": [1000.0, 1200.0, 1000.0]})
    return Portfolio(prices=prices, cashposition=pos, aum=1e5, cost_per_unit=0.05)


@pytest.fixture
def cost_bps_pf():
    """3-day portfolio with cost_bps=5.0 for cost_bps-forwarding tests."""
    prices = pl.DataFrame({"A": [100.0, 110.0, 105.0]})
    pos = pl.DataFrame({"A": [1000.0, 1200.0, 1000.0]})
    return Portfolio(prices=prices, cashposition=pos, aum=1e5, cost_bps=5.0)


# ─── cost_adjusted_returns ───────────────────────────────────────────────────


def test_cost_adjusted_returns_zero_bps_equals_base_returns(turnover_portfolio):
    """cost_adjusted_returns(0) must equal the base returns exactly."""
    base = turnover_portfolio.returns
    adj = turnover_portfolio.cost_adjusted_returns(0.0)
    pt.assert_series_equal(adj["returns"], base["returns"])


def test_cost_adjusted_returns_positive_costs_reduce_returns(turnover_portfolio):
    """Positive trading costs must strictly reduce returns for a portfolio with non-zero turnover."""
    base = turnover_portfolio.returns["returns"]
    adj = turnover_portfolio.cost_adjusted_returns(5.0)["returns"]
    assert float(adj.sum()) < float(base.sum())


def test_cost_adjusted_returns_preserves_date_column(turnover_portfolio):
    """cost_adjusted_returns must preserve the date column when present."""
    adj = turnover_portfolio.cost_adjusted_returns(10.0)
    assert "date" in adj.columns


def test_cost_adjusted_returns_negative_bps_raises(turnover_portfolio):
    """cost_adjusted_returns must raise ValueError for negative cost_bps."""
    with pytest.raises(ValueError, match="cost_bps must be non-negative"):
        turnover_portfolio.cost_adjusted_returns(-1.0)


def test_cost_adjusted_returns_non_numeric_bps_raises(turnover_portfolio):
    """cost_adjusted_returns must raise TypeError for non-numeric cost_bps (incl. bool)."""
    with pytest.raises(TypeError, match="cost_bps must be a number"):
        turnover_portfolio.cost_adjusted_returns("5")
    with pytest.raises(TypeError, match="cost_bps must be a number"):
        turnover_portfolio.cost_adjusted_returns(True)


def test_cost_adjusted_returns_non_finite_bps_raises(turnover_portfolio):
    """cost_adjusted_returns must raise ValueError for NaN or infinite cost_bps."""
    with pytest.raises(ValueError, match="cost_bps must be finite"):
        turnover_portfolio.cost_adjusted_returns(float("inf"))
    with pytest.raises(ValueError, match="cost_bps must be finite"):
        turnover_portfolio.cost_adjusted_returns(float("nan"))


def test_cost_adjusted_returns_higher_bps_lower_returns(turnover_portfolio):
    """Higher basis points must lead to lower (or equal) total adjusted returns."""
    adj5 = float(turnover_portfolio.cost_adjusted_returns(5.0)["returns"].sum())
    adj10 = float(turnover_portfolio.cost_adjusted_returns(10.0)["returns"].sum())
    assert adj10 <= adj5


# ─── trading_cost_impact ──────────────────────────────────────────────────────


def test_trading_cost_impact_columns(turnover_portfolio):
    """trading_cost_impact must return a DataFrame with 'cost_bps' and 'sharpe' columns."""
    impact = turnover_portfolio.trading_cost_impact()
    assert "cost_bps" in impact.columns
    assert "sharpe" in impact.columns


def test_trading_cost_impact_default_range(turnover_portfolio):
    """trading_cost_impact default runs from 0 to 20 inclusive (21 rows)."""
    impact = turnover_portfolio.trading_cost_impact()
    assert impact.height == 21
    assert int(impact["cost_bps"][0]) == 0
    assert int(impact["cost_bps"][-1]) == 20


def test_trading_cost_impact_custom_max_bps(turnover_portfolio):
    """trading_cost_impact(max_bps=5) must produce exactly 6 rows (0..5)."""
    impact = turnover_portfolio.trading_cost_impact(max_bps=5)
    assert impact.height == 6
    assert list(impact["cost_bps"].to_list()) == [0, 1, 2, 3, 4, 5]


def test_trading_cost_impact_sharpe_decreases_with_cost():
    """Sharpe ratio must be non-increasing as trading costs increase."""
    n = 252
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates, "A": pl.Series([100.0 * (1.001**i) for i in range(n)])})
    positions = pl.DataFrame({"date": dates, "A": pl.Series([1000.0 + 100.0 * (i % 2) for i in range(n)])})
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5)
    impact = pf.trading_cost_impact(max_bps=10)
    sharpe = [s for s in impact["sharpe"].to_list() if s == s]  # drop NaN
    for i in range(len(sharpe) - 1):
        assert sharpe[i] >= sharpe[i + 1] - 1e-9, f"Sharpe increased at bps={i}: {sharpe[i]} → {sharpe[i + 1]}"


def test_trading_cost_impact_invalid_max_bps_raises(turnover_portfolio):
    """trading_cost_impact must raise ValueError for max_bps=0 or non-integer."""
    with pytest.raises(ValueError, match="max_bps must be a positive integer"):
        turnover_portfolio.trading_cost_impact(max_bps=0)
    with pytest.raises(ValueError, match="max_bps must be a positive integer"):
        turnover_portfolio.trading_cost_impact(max_bps=-1)  # type: ignore[arg-type]


def test_trading_cost_impact_vectorised_equivalence():
    """Regression: vectorised implementation must match per-iteration formula exactly."""
    n = 252
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates, "A": pl.Series([100.0 * (1.001**i) for i in range(n)])})
    positions = pl.DataFrame({"date": dates, "A": pl.Series([1000.0 + 50.0 * (i % 3) for i in range(n)])})
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5)

    max_bps = 10
    impact = pf.trading_cost_impact(max_bps=max_bps)

    base_rets = pf.returns["returns"].to_numpy()
    turnover_arr = pf.turnover["turnover"].to_numpy()
    periods = pf.data._periods_per_year
    _eps = np.finfo(np.float64).eps

    for i, bps in enumerate(range(0, max_bps + 1)):
        adj = base_rets - turnover_arr * (bps / 10_000.0)
        mean_val = float(np.mean(adj))
        std_val = float(np.std(adj, ddof=1))
        if std_val <= _eps * max(abs(mean_val), _eps) * 10:
            expected = float("nan")
        else:
            expected = mean_val / std_val * math.sqrt(periods)

        actual = impact["sharpe"][i]
        if math.isnan(expected):
            assert actual is None or math.isnan(float(actual)), f"Expected NaN at bps={bps}, got {actual}"
        else:
            assert math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-10), (
                f"Mismatch at bps={bps}: got {float(actual)}, expected {expected}"
            )


# ─── Plots.trading_cost_impact_plot ──────────────────────────────────────────


def test_trading_cost_impact_plot_returns_figure(turnover_portfolio):
    """trading_cost_impact_plot must return a Plotly Figure."""
    fig = turnover_portfolio.plots.trading_cost_impact_plot()
    assert isinstance(fig, go.Figure)


def test_trading_cost_impact_plot_has_one_trace(turnover_portfolio):
    """trading_cost_impact_plot figure must have exactly one scatter trace."""
    fig = turnover_portfolio.plots.trading_cost_impact_plot()
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(scatter_traces) == 1


def test_trading_cost_impact_plot_x_axis_length(turnover_portfolio):
    """trading_cost_impact_plot trace x-values must span 0..20 (21 points)."""
    fig = turnover_portfolio.plots.trading_cost_impact_plot()
    assert len(fig.data[0].x) == 21


def test_trading_cost_impact_plot_title_contains_bps(turnover_portfolio):
    """trading_cost_impact_plot figure title must mention 'bps'."""
    fig = turnover_portfolio.plots.trading_cost_impact_plot()
    assert "bps" in fig.layout.title.text.lower()


# ─── position_delta_costs ────────────────────────────────────────────────────


def test_position_delta_costs_columns(cost_portfolio):
    """position_delta_costs must have 'date' and 'cost' columns."""
    df = cost_portfolio.position_delta_costs
    assert "date" in df.columns
    assert "cost" in df.columns
    assert df.height == cost_portfolio.prices.height


def test_position_delta_costs_first_row_is_zero(cost_portfolio):
    """First row of position_delta_costs must be 0.0 (no prior position)."""
    assert float(cost_portfolio.position_delta_costs["cost"][0]) == pytest.approx(0.0, abs=1e-12)


def test_position_delta_costs_analytical_values(cost_portfolio):
    """position_delta_costs must match known analytical values.

    Position changes: 0, 1000, 300  → costs: 0*0.01, 1000*0.01, 300*0.01.
    """
    costs = cost_portfolio.position_delta_costs["cost"].to_list()
    assert costs[0] == pytest.approx(0.0, abs=1e-12)
    assert costs[1] == pytest.approx(10.0, rel=1e-9)
    assert costs[2] == pytest.approx(3.0, rel=1e-9)


def test_position_delta_costs_zero_cost_per_unit_all_zeros():
    """When cost_per_unit=0.0 all costs must be zero regardless of positions."""
    n = 5
    start = date(2021, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    pf = Portfolio(
        prices=pl.DataFrame({"date": dates, "A": pl.Series([100.0] * n)}),
        cashposition=pl.DataFrame({"date": dates, "A": pl.Series([float(i) * 500 for i in range(n)])}),
        aum=1e4,
        cost_per_unit=0.0,
    )
    costs = pf.position_delta_costs["cost"].to_list()
    assert all(c == pytest.approx(0.0, abs=1e-12) for c in costs)


def test_position_delta_costs_two_assets():
    """position_delta_costs must sum |Δx| across all assets correctly.

    Asset A: 0 → 500 → 200,  |Δ| = 0, 500, 300
    Asset B: 0 → 200 → 400,  |Δ| = 0, 200, 200
    cost_per_unit = 0.005
    Expected costs: 0, (500+200)*0.005=3.5, (300+200)*0.005=2.5
    """
    n = 3
    start = date(2020, 3, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    pf = Portfolio(
        prices=pl.DataFrame({"date": dates, "A": [100.0] * n, "B": [200.0] * n}),
        cashposition=pl.DataFrame({"date": dates, "A": [0.0, 500.0, 200.0], "B": [0.0, 200.0, 400.0]}),
        aum=1e5,
        cost_per_unit=0.005,
    )
    costs = pf.position_delta_costs["cost"].to_list()
    assert costs[0] == pytest.approx(0.0, abs=1e-12)
    assert costs[1] == pytest.approx(3.5, rel=1e-9)
    assert costs[2] == pytest.approx(2.5, rel=1e-9)


def test_position_delta_costs_without_date_column():
    """position_delta_costs must work on date-free portfolios."""
    pf = Portfolio(
        prices=pl.DataFrame({"A": [100.0, 110.0, 121.0]}),
        cashposition=pl.DataFrame({"A": [0.0, 1000.0, 700.0]}),
        aum=1e4,
        cost_per_unit=0.001,
    )
    df = pf.position_delta_costs
    assert "date" not in df.columns
    assert "cost" in df.columns
    assert df.height == 3
    costs = df["cost"].to_list()
    assert costs[0] == pytest.approx(0.0)
    assert costs[1] == pytest.approx(1.0, rel=1e-9)
    assert costs[2] == pytest.approx(0.3, rel=1e-9)


def test_position_delta_costs_nan_warmup_rows_produce_zero_cost():
    """NaN positions (e.g. EWMA warmup rows) must not propagate NaN into cost.

    Positions: [NaN, NaN, 1000, 1200, 900]
    After fill_null+fill_nan: [0, 0, 0, 200, 300]
    cost (* 0.01): [0, 0, 0, 2, 3]
    """
    n = 5
    start = date(2021, 6, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    positions = [float("nan"), float("nan"), 1000.0, 1200.0, 900.0]
    pf = Portfolio(
        prices=pl.DataFrame({"date": dates, "A": [100.0] * n}),
        cashposition=pl.DataFrame({"date": dates, "A": pl.Series(positions, dtype=pl.Float64)}),
        aum=1e5,
        cost_per_unit=0.01,
    )
    costs = pf.position_delta_costs["cost"].to_list()
    assert all(c == c for c in costs), "NaN found in cost column"  # NaN != NaN
    assert costs[0] == pytest.approx(0.0, abs=1e-12)
    assert costs[1] == pytest.approx(0.0, abs=1e-12)
    assert costs[2] == pytest.approx(0.0, abs=1e-12)  # 1000 - NaN → NaN → 0
    assert costs[3] == pytest.approx(2.0, rel=1e-9)  # |1200 - 1000| * 0.01
    assert costs[4] == pytest.approx(3.0, rel=1e-9)  # |900 - 1200| * 0.01


# ─── net_cost_nav ─────────────────────────────────────────────────────────────


def test_net_cost_nav_columns(cost_portfolio):
    """net_cost_nav must return a frame with 'date', 'profit', 'cost', and 'NAV_accumulated_net'."""
    df = cost_portfolio.net_cost_nav
    for col in ("date", "profit", "cost", "NAV_accumulated_net"):
        assert col in df.columns, f"Missing column: {col}"
    assert df.height == cost_portfolio.prices.height


def test_net_cost_nav_zero_cost_equals_nav_accumulated():
    """With cost_per_unit=0, net_cost_nav must equal the gross nav_accumulated."""
    n = 5
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    pf = Portfolio(
        prices=pl.DataFrame({"date": dates, "A": pl.Series([100.0 * (1.01**i) for i in range(n)])}),
        cashposition=pl.DataFrame({"date": dates, "A": pl.Series([1000.0] * n)}),
        aum=1e5,
        cost_per_unit=0.0,
    )
    net_nav = pf.net_cost_nav["NAV_accumulated_net"].to_list()
    gross_nav = pf.nav_accumulated["NAV_accumulated"].to_list()
    for net, gross in zip(net_nav, gross_nav, strict=False):
        assert net == pytest.approx(gross, rel=1e-9)


def test_net_cost_nav_positive_cost_lower_than_gross(cost_portfolio):
    """With positive cost_per_unit, net NAV must be <= gross NAV at every point."""
    net_nav = cost_portfolio.net_cost_nav["NAV_accumulated_net"].to_list()
    gross_nav = cost_portfolio.nav_accumulated["NAV_accumulated"].to_list()
    for net, gross in zip(net_nav, gross_nav, strict=False):
        assert net <= gross + 1e-9, f"net {net} > gross {gross}"


def test_net_cost_nav_analytical_values(cost_portfolio):
    """net_cost_nav must match exact hand-computed values.

    Setup (cost_portfolio fixture):
      AUM = 1e5, Positions: A = [0, 1000, 700], cost_per_unit=0.01
      Prices: A = [100, 110, 121]
      Daily costs: 0, 10.0, 3.0
      Net profit: 0-0=0, 0-10=-10, 100-3=97
      NAV: 1e5, 1e5-10=99990, 99990+97=100087
    """
    df = cost_portfolio.net_cost_nav
    nav = df["NAV_accumulated_net"].to_list()
    assert nav[0] == pytest.approx(1e5, rel=1e-9)
    assert nav[1] == pytest.approx(1e5 - 10.0, rel=1e-9)
    assert nav[2] == pytest.approx(1e5 - 10.0 + 97.0, rel=1e-9)


# ─── Portfolio.cost_per_unit field ────────────────────────────────────────────


def test_portfolio_cost_per_unit_default_is_zero():
    """Portfolio constructed without cost_per_unit must default to 0.0."""
    pf = Portfolio(
        prices=pl.DataFrame({"date": [date(2020, 1, 1)], "A": [100.0]}),
        cashposition=pl.DataFrame({"date": [date(2020, 1, 1)], "A": [1000.0]}),
        aum=1e5,
    )
    assert pf.cost_per_unit == 0.0


def test_portfolio_from_cash_position_passes_cost_per_unit():
    """from_cash_position must forward cost_per_unit to the Portfolio."""
    prices = pl.DataFrame({"date": [date(2020, 1, 1)], "A": [100.0]})
    pos = pl.DataFrame({"date": [date(2020, 1, 1)], "A": [500.0]})
    pf = Portfolio.from_cash_position(prices=prices, cash_position=pos, aum=1e5, cost_per_unit=0.002)
    assert pf.cost_per_unit == pytest.approx(0.002)


# ─── Portfolio.from_position ──────────────────────────────────────────────────


def test_from_position_cash_equals_units_times_price():
    """from_position must set cashposition = position * prices for each asset."""
    prices = pl.DataFrame({"A": [100.0, 110.0, 105.0], "B": [50.0, 55.0, 52.0]})
    pos = pl.DataFrame({"A": [10.0, 10.0, 10.0], "B": [20.0, 20.0, 20.0]})
    pf = Portfolio.from_position(prices=prices, position=pos, aum=1e6)
    assert pf.cashposition["A"].to_list() == pytest.approx([1000.0, 1100.0, 1050.0])
    assert pf.cashposition["B"].to_list() == pytest.approx([1000.0, 1100.0, 1040.0])


def test_from_position_forwards_cost_model():
    """from_position must forward cost_model parameters to from_cash_position."""
    prices = pl.DataFrame({"A": [100.0, 110.0]})
    pos = pl.DataFrame({"A": [5.0, 5.0]})
    cm = CostModel(cost_per_unit=0.01, cost_bps=0.0)
    pf = Portfolio.from_position(prices=prices, position=pos, aum=1e5, cost_model=cm)
    assert pf.cost_per_unit == pytest.approx(0.01)


def test_from_position_accepts_expr():
    """from_position evaluates a pl.Expr against prices in place of a DataFrame."""
    prices = pl.DataFrame({"A": [100.0, 110.0, 105.0], "B": [50.0, 55.0, 52.0]})
    # Expression sets both A and B to 10 units; cashposition = 10 * price
    pf_expr = Portfolio.from_position(prices=prices, position=pl.all() * 0.0 + 10.0, aum=1e6)
    assert pf_expr.cashposition["A"].to_list() == pytest.approx([1000.0, 1100.0, 1050.0])
    assert pf_expr.cashposition["B"].to_list() == pytest.approx([500.0, 550.0, 520.0])


def test_net_cost_nav_integer_indexed(int_portfolio):
    """net_cost_nav on an integer-indexed portfolio uses hstack (no 'date' join)."""
    result = int_portfolio.net_cost_nav
    assert "NAV_accumulated_net" in result.columns
    assert "profit" in result.columns
    assert "cost" in result.columns
    assert "date" not in result.columns
    assert result.height == int_portfolio.prices.height


# ─── cost_per_unit forwarding through transforms ──────────────────────────────


def test_truncate_forwards_cost_per_unit(cost_pf):
    """Truncate must preserve cost_per_unit."""
    assert cost_pf.truncate(start=0, end=1).cost_per_unit == pytest.approx(0.05)


def test_lag_forwards_cost_per_unit(cost_pf):
    """Lag must preserve cost_per_unit."""
    assert cost_pf.lag(1).cost_per_unit == pytest.approx(0.05)


def test_smoothed_holding_forwards_cost_per_unit(cost_pf):
    """smoothed_holding must preserve cost_per_unit."""
    assert cost_pf.smoothed_holding(1).cost_per_unit == pytest.approx(0.05)


def test_tilt_forwards_cost_per_unit(cost_pf):
    """Tilt must preserve cost_per_unit."""
    assert cost_pf.tilt.cost_per_unit == pytest.approx(0.05)


def test_timing_forwards_cost_per_unit(cost_pf):
    """Timing must preserve cost_per_unit."""
    assert cost_pf.timing.cost_per_unit == pytest.approx(0.05)


# ─── cost_bps construction-time parameter ────────────────────────────────────


def test_cost_bps_forwarded_through_transforms(cost_bps_pf):
    """All transforms must preserve the construction-time cost_bps value."""
    assert cost_bps_pf.truncate(start=0, end=1).cost_bps == pytest.approx(5.0)
    assert cost_bps_pf.lag(1).cost_bps == pytest.approx(5.0)
    assert cost_bps_pf.smoothed_holding(1).cost_bps == pytest.approx(5.0)
    assert cost_bps_pf.tilt.cost_bps == pytest.approx(5.0)
    assert cost_bps_pf.timing.cost_bps == pytest.approx(5.0)


def test_cost_adjusted_returns_defaults_to_construction_cost_bps(cost_bps_pf, turnover_portfolio):
    """cost_adjusted_returns() with no argument uses self.cost_bps."""
    adj_default = cost_bps_pf.cost_adjusted_returns()
    adj_explicit = cost_bps_pf.cost_adjusted_returns(5.0)
    assert adj_default["returns"].to_list() == pytest.approx(adj_explicit["returns"].to_list())


def test_cost_adjusted_returns_explicit_overrides_construction_cost_bps(cost_bps_pf):
    """An explicit cost_bps argument overrides self.cost_bps."""
    adj_override = cost_bps_pf.cost_adjusted_returns(0.0)
    adj_base = cost_bps_pf.cost_adjusted_returns(5.0)
    assert float(adj_override["returns"].sum()) >= float(adj_base["returns"].sum())
