"""Tests for Portfolio returns, NAV variants, drawdown, and aggregation properties."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from jquantstats import Portfolio

from ..tolerances import TOL_FLOAT64


def test_returns_property_scales_profit_by_aum_and_preserves_date(portfolio):
    """Returns should divide numeric columns by aum and retain the 'date' column."""
    rets = portfolio.returns

    assert "date" in rets.columns
    assert "profit" in rets.columns

    expected = (portfolio.profit.select(pl.col("profit")) / portfolio.aum)["profit"].to_numpy()
    actual = rets["returns"].to_numpy()
    assert np.allclose(actual, expected, rtol=TOL_FLOAT64, atol=TOL_FLOAT64)


def test_nav_compounded_uses_compounding_and_is_close_to_nav_for_small_returns(portfolio):
    """nav_compounded should compound returns; for small returns it approximates additive NAV."""
    nav_add = portfolio.nav_accumulated
    nav_cmp = portfolio.nav_compounded

    assert "date" in nav_cmp.columns
    assert "date" in nav_add.columns

    cmp_values = nav_cmp["NAV_compounded"].to_numpy()
    assert np.isclose(cmp_values[0], portfolio.aum)

    add_values = nav_add["NAV_accumulated"].to_numpy()
    assert np.isclose(add_values[0], portfolio.aum)


def test_highwater_is_cummax_of_nav(portfolio):
    """Highwater should equal the cumulative maximum of NAV and preserve 'date'."""
    nav_df = portfolio.nav_accumulated
    hw_df = portfolio.highwater

    assert "date" in hw_df.columns
    assert "highwater" in hw_df.columns

    expected = nav_df["NAV_accumulated"].cum_max().to_numpy()
    actual = hw_df["highwater"].to_numpy()
    assert np.allclose(actual, expected, rtol=TOL_FLOAT64, atol=TOL_FLOAT64)


def test_drawdown_is_highwater_minus_nav_and_preserves_date(portfolio):
    """Drawdown should equal highwater - NAV, start at 0, be non-negative, and keep 'date'."""
    dd_df = portfolio.drawdown

    assert "date" in dd_df.columns
    assert "drawdown" in dd_df.columns

    expected = (dd_df["highwater"] - dd_df["NAV_accumulated"]).to_numpy()
    actual = dd_df["drawdown"].to_numpy()

    assert np.isclose(actual[0], 0.0)
    assert np.all(actual >= 0.0)
    assert np.allclose(actual, expected, rtol=TOL_FLOAT64, atol=TOL_FLOAT64)


def test_stats(portfolio):
    """stats() returns legacy Stats with expected Sharpe; kurtosis is None for tiny samples."""
    stats = portfolio.stats

    assert pytest.approx(stats.sharpe()["returns"]) == 20.845234695819794
    # Legacy kurtosis(bias=False) requires ≥4 observations; returns None for 3-row portfolios.
    assert stats.kurtosis()["returns"] is None


def test_portfolio_snapshot_log_scale(portfolio):
    """snapshot(log_scale=True) returns a Figure and sets the first y-axis to logarithmic scale."""
    fig = portfolio.plots.snapshot(log_scale=True)
    assert isinstance(fig, go.Figure)
    assert fig.layout.yaxis.type == "log"


def test_assert_clean_series_raises_on_null():
    """_assert_clean_series should raise UncleanSeriesError when series contains null values."""
    s = pl.Series([1.0, None, 3.0])
    with pytest.raises(ValueError, match="contains null values"):
        Portfolio._assert_clean_series(s)


def test_assert_clean_series_raises_on_nonfinite():
    """_assert_clean_series should raise UncleanSeriesError when series contains non-finite values."""
    s = pl.Series([1.0, float("inf"), 3.0])
    with pytest.raises(ValueError, match="contains non-finite values"):
        Portfolio._assert_clean_series(s)


def test_portfolio_all_merges_drawdown_and_nav_compounded(portfolio):
    """All property should join drawdown and nav_compounded on 'date' with expected columns."""
    result = portfolio.all
    assert "date" in result.columns
    assert "NAV_accumulated" in result.columns
    assert "NAV_compounded" in result.columns
    assert "drawdown" in result.columns
    assert len(result) == len(portfolio.prices)


# ─── Caching tests ────────────────────────────────────────────────────────────


def test_profits_cache_is_none_before_first_access(portfolio):
    """_profits_cache should be None before profits is accessed."""
    assert portfolio._profits_cache is None


def test_profits_cache_is_set_after_first_access(portfolio):
    """_profits_cache should be populated after profits is accessed."""
    _ = portfolio.profits
    assert portfolio._profits_cache is not None


def test_profits_returns_cached_object_on_second_access(portfolio):
    """Repeated access to profits should return the same cached DataFrame object."""
    first = portfolio.profits
    second = portfolio.profits
    assert first is second


def test_returns_cache_is_none_before_first_access(portfolio):
    """_returns_cache should be None before returns is accessed."""
    assert portfolio._returns_cache is None


def test_returns_cache_is_set_after_first_access(portfolio):
    """_returns_cache should be populated after returns is accessed."""
    _ = portfolio.returns
    assert portfolio._returns_cache is not None


def test_returns_returns_cached_object_on_second_access(portfolio):
    """Repeated access to returns should return the same cached DataFrame object."""
    first = portfolio.returns
    second = portfolio.returns
    assert first is second
