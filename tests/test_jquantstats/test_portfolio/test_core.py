"""Tests for Portfolio construction, validation, and core profit/NAV/Sharpe properties."""

from __future__ import annotations

from datetime import date

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from jquantstats import Portfolio

# ─── Core: profits, NAV, Sharpe ───────────────────────────────────────────────


def test_compute_daily_profits_portfolio_basic(portfolio):
    """Compute per-asset profits and preserve date column."""
    profits = portfolio.profits

    assert "date" in profits.columns

    expected = pl.DataFrame(
        {
            "date": portfolio.prices["date"],
            "A": pl.Series([0.0, 100.0, 100.0], dtype=pl.Float64),
            "B": pl.Series([0.0, 0.0, 50.0], dtype=pl.Float64),
        }
    )

    assert profits.columns == expected.columns
    for c in ["A", "B"]:
        assert np.allclose(profits[c].to_numpy(), expected[c].to_numpy(), rtol=1e-12, atol=1e-12)


def test_portfolio_profit_and_nav(portfolio):
    """Aggregate per-asset profits to portfolio profit and compute NAV."""
    profit_df = portfolio.profit
    assert profit_df.columns == ["date", "profit"]

    expected_profit = np.array([0.0, 100.0, 150.0])
    assert np.allclose(profit_df["profit"].to_numpy(), expected_profit)

    nav_df = portfolio.nav_accumulated
    assert nav_df.columns == ["date", "profit", "NAV_accumulated"]

    expected_nav = np.array([1e5, 1e5 + 100.0, 1e5 + 250.0])
    assert np.allclose(nav_df["NAV_accumulated"].to_numpy(), expected_nav)


def test_portfolio_sharpe_matches_manual(portfolio):
    """Sharpe returned by class matches manual computation."""
    out = portfolio.stats.sharpe()["returns"]
    assert np.isfinite(out)
    assert np.isclose(out, 20.845234695819794, rtol=1e-12, atol=1e-12)


def test_portfolio_plot_returns_figure(portfolio):
    """Plot method returns a Plotly Figure and is serializable."""
    fig = portfolio.plots.snapshot()
    assert isinstance(fig, go.Figure)
    _ = fig.to_dict()


# ─── __post_init__ validation ─────────────────────────────────────────────────


def test_portfolio_post_init_requires_polars_dataframes(prices, positions):
    """__post_init__ should assert inputs are Polars DataFrames."""
    with pytest.raises(TypeError, match=r"cashposition must be pl\.DataFrame, got dict"):
        Portfolio(prices=prices, cashposition={"date": [1, 2, 3]}, aum=1e5)

    with pytest.raises(TypeError, match=r"prices must be pl\.DataFrame, got list"):
        Portfolio(prices=[[1.0, 2.0, 3.0]], cashposition=positions, aum=1e5)


def test_portfolio_post_init_requires_same_number_of_rows(prices, positions):
    """__post_init__ should raise ValueError when row counts differ."""
    with pytest.raises(ValueError, match=r"cashposition and prices must have the same number of rows"):
        Portfolio(prices=prices.head(3), cashposition=positions.head(2), aum=1e5)


def test_portfolio_post_init_requires_positive_aum(prices, positions):
    """__post_init__ should raise ValueError when AUM is not strictly positive."""
    with pytest.raises(ValueError, match=r"aum must be strictly positive"):
        Portfolio(prices=prices, cashposition=positions, aum=0.0)

    with pytest.raises(ValueError, match=r"aum must be strictly positive"):
        Portfolio(prices=prices, cashposition=positions, aum=-1.0)


# ─── from_riskposition edge cases ─────────────────────────────────────────────


def test_from_riskposition_returns_portfolio_and_cashposition_shape():
    """from_riskposition should return a Portfolio with aligned cashposition columns/height."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 2, 10), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.linspace(100, 120, len(dates)), dtype=pl.Float64),
            "B": pl.Series(np.linspace(50, 60, len(dates)), dtype=pl.Float64),
        }
    )
    riskposition = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.sin(np.linspace(0, 3.14, len(dates))), dtype=pl.Float64),
            "B": pl.Series(np.cos(np.linspace(0, 3.14, len(dates))), dtype=pl.Float64),
        }
    )

    pf = Portfolio.from_risk_position(prices, riskposition, vola=8, aum=1e8)
    assert isinstance(pf, Portfolio)
    assert pf.cashposition.height == prices.height
    for c in ["A", "B"]:
        assert c in pf.cashposition.columns


def test_sharpe_zero_std_returns_nan():
    """Sharpe should return NaN when NAV differences have zero std (flat NAV)."""
    import math

    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 5), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates, "A": pl.Series([100.0] * len(dates), dtype=pl.Float64)})
    positions = pl.DataFrame({"date": dates, "A": pl.Series([0.0] * len(dates), dtype=pl.Float64)})

    pf = Portfolio(prices=prices, cashposition=positions, aum=1e5)
    result = pf.stats.sharpe()["returns"]
    assert math.isnan(result)


def test_compute_daily_profits_replaces_nonfinite_with_zero():
    """_compute_daily_profits_portfolio should replace non-finite profit values with 0.0."""
    prices = pl.DataFrame(
        {
            "date": pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 2), interval="1d", eager=True).cast(
                pl.Date
            ),
            "A": pl.Series([0.0, 1.0], dtype=pl.Float64),
        }
    )
    positions = pl.DataFrame({"date": prices["date"], "A": pl.Series([1.0, 1.0], dtype=pl.Float64)})

    pf = Portfolio(prices=prices, cashposition=positions, aum=1e5)
    profits = pf.profits
    assert np.allclose(profits["A"].to_numpy(), np.array([0.0, 0.0]))


def test_compute_daily_profits_no_numeric_columns():
    """When there are no numeric columns, function should return only non-numeric columns unchanged."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 2), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates})
    positions = pl.DataFrame({"date": dates})
    pf = Portfolio(prices=prices, cashposition=positions, aum=1e5)
    profits = pf.profits
    assert profits.columns == ["date"]
    assert profits.height == 2


def test_profit_raises_when_no_numeric_asset_columns():
    """Portfolio.profit should raise ValueError if there are no numeric asset columns."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 2), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates})
    positions = pl.DataFrame({"date": dates})
    pf = Portfolio(prices=prices, cashposition=positions, aum=1e5)

    with pytest.raises(ValueError, match="no numeric asset columns"):
        _ = pf.profit


def test_from_cash_position_accepts_expr(prices):
    """from_cash_position evaluates a pl.Expr against prices in place of a DataFrame."""
    expr = pl.all().exclude("date") * 10.0
    pf_expr = Portfolio.from_cash_position(prices=prices, cash_position=expr, aum=1e5)
    pf_df = Portfolio.from_cash_position(prices=prices, cash_position=prices.with_columns(expr), aum=1e5)

    for col in ["A", "B"]:
        assert np.allclose(pf_expr.profits[col].to_numpy(), pf_df.profits[col].to_numpy())


def test_from_cash_position_expr_with_new_column_raises(prices):
    """from_cash_position rejects an Expr that creates a column absent from prices."""
    from jquantstats.exceptions import PositionExprColumnError

    expr = (pl.col("A") * 10.0).alias("A_scaled")
    with pytest.raises(PositionExprColumnError, match="cash_position expression created new column"):
        Portfolio.from_cash_position(prices=prices, cash_position=expr, aum=1e5)


def test_from_cash_position_expr_error_names_extra_columns(prices):
    """The PositionExprColumnError carries the parameter name and offending columns."""
    from jquantstats.exceptions import PositionExprColumnError

    expr = (pl.col("A") * 10.0).alias("A_scaled")
    with pytest.raises(PositionExprColumnError) as exc_info:
        Portfolio.from_cash_position(prices=prices, cash_position=expr, aum=1e5)
    assert exc_info.value.param == "cash_position"
    assert exc_info.value.extra == ["A_scaled"]


def test_from_position_expr_with_new_column_raises(prices):
    """from_position rejects an Expr that creates a column absent from prices."""
    from jquantstats.exceptions import PositionExprColumnError

    expr = (pl.col("A") * 2.0).alias("units")
    with pytest.raises(PositionExprColumnError, match="position expression created new column"):
        Portfolio.from_position(prices=prices, position=expr, aum=1e5)


def test_from_risk_position_expr_with_new_column_raises(prices):
    """from_risk_position rejects an Expr that creates a column absent from prices."""
    from jquantstats.exceptions import PositionExprColumnError

    expr = (pl.col("A") * 0.5).alias("risk_units")
    with pytest.raises(PositionExprColumnError, match="risk_position expression created new column"):
        Portfolio.from_risk_position(prices=prices, risk_position=expr, aum=1e5)
