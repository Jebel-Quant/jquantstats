"""Tests for Portfolio positional transforms: lag, smoothed_holding, tilt, timing, truncate, monthly."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import polars.testing as pt
import pytest

from jquantstats import Portfolio
from jquantstats.exceptions import IntegerIndexBoundError, MissingDateColumnError

# ─── Lag ─────────────────────────────────────────────────────────────────────


def test_lag_positive_shifts_weights_down_and_preserves_date(portfolio):
    """lag(+1) should shift numeric columns down by one and preserve 'date'."""
    pf_lag1 = portfolio.lag(1)
    assert isinstance(pf_lag1, Portfolio)
    assert pf_lag1.aum == portfolio.aum
    assert pf_lag1.cashposition.columns[0] == "date"

    for c in portfolio.assets:
        s0 = portfolio.cashposition[c]
        s1 = pf_lag1.cashposition[c]
        assert s1.null_count() == 1
        assert np.allclose(s1.drop_nulls().to_numpy(), s0[:-1].to_numpy(), rtol=0, atol=0)

    _ = pf_lag1.profit


def test_lag_negative_leads_weights_and_last_becomes_null(portfolio):
    """lag(-1) should lead numeric columns; last element becomes null."""
    pf_lead1 = portfolio.lag(-1)
    for c in portfolio.assets:
        s0 = portfolio.cashposition[c]
        s1 = pf_lead1.cashposition[c]
        assert s1.null_count() == 1
        assert np.allclose(s1.head(len(s1) - 1).to_numpy(), s0[1:].to_numpy(), rtol=0, atol=0)

    _ = pf_lead1.profit


def test_lag_zero_returns_same_portfolio_object_or_equal_data(portfolio):
    """lag(0) should be a no-op: same object or equal data content and AUM preserved."""
    pf0 = portfolio.lag(0)
    assert pf0.aum == portfolio.aum
    pt.assert_frame_equal(pf0.cashposition, portfolio.cashposition)
    pt.assert_frame_equal(pf0.prices, portfolio.prices)


def test_lag_raises_typeerror_for_non_int(portfolio):
    """Passing a non-integer value to lag() should raise TypeError."""
    with pytest.raises(TypeError):
        _ = portfolio.lag(1.5)  # type: ignore[arg-type]


# ─── Smoothed holding ─────────────────────────────────────────────────────────


def test_smoothed_holding_zero_returns_self_and_preserves_state(portfolio):
    """Calling smoothed_holding(0) should return the same Portfolio instance and keep data intact."""
    pf_zero = portfolio.smoothed_holding(0)

    assert pf_zero is portfolio
    assert pf_zero.aum == portfolio.aum
    pt.assert_frame_equal(pf_zero.prices, portfolio.prices)
    pt.assert_frame_equal(pf_zero.cashposition, portfolio.cashposition)

    nav_before = portfolio.nav_accumulated
    _ = portfolio.profit
    nav_after = pf_zero.nav_accumulated
    pt.assert_frame_equal(nav_after, nav_before)


# ─── Tilt / timing ────────────────────────────────────────────────────────────


def test_timing_prices_are_difference_and_portfolio_computable(portfolio):
    """timing.prices must equal original prices - tilt.prices, date preserved."""
    tilt = portfolio.tilt
    assert isinstance(tilt, Portfolio)
    assert tilt.aum == portfolio.aum
    pt.assert_frame_equal(tilt.prices, portfolio.prices)

    timing = portfolio.timing
    assert isinstance(timing, Portfolio)
    assert timing.aum == portfolio.aum
    pt.assert_frame_equal(timing.prices, portfolio.prices)

    pt.assert_frame_equal(
        portfolio.cashposition.select(portfolio.assets),
        timing.cashposition.select(portfolio.assets) + tilt.cashposition.select(portfolio.assets),
    )
    print(portfolio.tilt_timing_decomp)


# ─── Truncate ─────────────────────────────────────────────────────────────────


def test_truncate_by_start_end_inclusive_preserves_aum_and_dates(truncate_portfolio):
    """Truncating with both start and end returns new Portfolio, preserves AUM, and filters dates inclusively."""
    start = date(2020, 1, 2)
    end = date(2020, 1, 4)

    pf_t = truncate_portfolio.truncate(start=start, end=end)

    assert isinstance(pf_t, Portfolio)
    assert pf_t.aum == truncate_portfolio.aum
    assert pf_t.prices.height == 3
    assert pf_t.cashposition.height == 3
    assert pf_t.prices["date"].min() == start
    assert pf_t.prices["date"].max() == end

    nav = pf_t.nav_accumulated
    assert "NAV_accumulated" in nav.columns
    assert nav.height == 3


def test_truncate_with_only_start_or_end_open_bounds(truncate_portfolio):
    """Truncating with only a start or only an end applies open bounds and remains computable."""
    pf_s = truncate_portfolio.truncate(start=date(2020, 1, 4))
    assert pf_s.prices["date"].min() == date(2020, 1, 4)
    assert pf_s.prices.height == 3  # days 4,5,6

    pf_e = truncate_portfolio.truncate(end=date(2020, 1, 3))
    assert pf_e.prices["date"].max() == date(2020, 1, 3)
    assert pf_e.prices.height == 3  # days 1,2,3

    _ = pf_s.profit
    _ = pf_e.profit


# ─── Validation ───────────────────────────────────────────────────────────────


def test_portfolio_smoothed_holding_negative_raises_value_error(portfolio):
    """Portfolio.smoothed_holding should raise ValueError when n < 0."""
    with pytest.raises(ValueError, match="non-negative integer"):
        _ = portfolio.smoothed_holding(-1)


def test_portfolio_smoothed_holding_type_error_on_non_int(portfolio):
    """Portfolio.smoothed_holding should raise TypeError when n is not an int."""
    with pytest.raises(TypeError):
        _ = portfolio.smoothed_holding(1.5)  # type: ignore[arg-type]


# ─── Monthly ──────────────────────────────────────────────────────────────────


def test_monthly_structure_and_end_of_month_dates(monthly_portfolio):
    """Monthly should include date (month-end), returns, and calendar columns including month_name."""
    monthly = monthly_portfolio.monthly

    assert monthly.columns == ["date", "returns", "NAV_accumulated", "profit", "year", "month", "month_name"]
    assert monthly["date"].dtype == pl.Date
    assert list(monthly["year"]) == [2020, 2020, 2020]
    assert list(monthly["month"]) == [1, 2, 3]
    assert list(monthly["month_name"]) == ["Jan", "Feb", "Mar"]
    assert monthly["returns"].is_finite().all()


# ─── Integer-indexed (no date column) portfolios ─────────────────────────────


def test_truncate_integer_indexed_both_bounds(int_portfolio):
    """truncate(start, end) on integer-indexed portfolio slices rows inclusively."""
    pf_t = int_portfolio.truncate(start=1, end=3)
    assert isinstance(pf_t, Portfolio)
    assert pf_t.prices.height == 3
    assert pf_t.cashposition.height == 3
    assert pf_t.aum == int_portfolio.aum
    _ = pf_t.profit


def test_truncate_integer_indexed_start_only(int_portfolio):
    """truncate(start=n) on integer-indexed portfolio returns rows from n onward."""
    pf_s = int_portfolio.truncate(start=2)
    assert pf_s.prices.height == 4  # rows 2,3,4,5
    _ = pf_s.profit


def test_truncate_integer_indexed_end_only(int_portfolio):
    """truncate(end=n) on integer-indexed portfolio returns rows up to n inclusive."""
    pf_e = int_portfolio.truncate(end=2)
    assert pf_e.prices.height == 3  # rows 0,1,2
    _ = pf_e.profit


def test_truncate_integer_indexed_no_bounds_returns_full(int_portfolio):
    """truncate() with no bounds on integer-indexed portfolio returns all rows."""
    pf_all = int_portfolio.truncate()
    assert pf_all.prices.height == int_portfolio.prices.height


def test_truncate_integer_indexed_raises_on_non_int_start(int_portfolio):
    """Truncate with non-integer start on integer-indexed portfolio raises IntegerIndexBoundError."""
    with pytest.raises(IntegerIndexBoundError, match="start must be an integer"):
        int_portfolio.truncate(start="2020-01-01")


def test_truncate_integer_indexed_raises_on_non_int_end(int_portfolio):
    """Truncate with non-integer end on integer-indexed portfolio raises IntegerIndexBoundError."""
    with pytest.raises(IntegerIndexBoundError, match="end must be an integer"):
        int_portfolio.truncate(end=3.5)


def test_monthly_raises_without_date_column(int_portfolio):
    """Portfolio.monthly raises MissingDateColumnError when no 'date' column is present."""
    with pytest.raises(MissingDateColumnError, match="missing the required 'date' column"):
        _ = int_portfolio.monthly


def test_all_works_without_date_column(int_portfolio):
    """Portfolio.all returns a DataFrame with expected columns for integer-indexed data."""
    result = int_portfolio.all
    assert "NAV_accumulated" in result.columns
    assert "NAV_compounded" in result.columns
    assert "drawdown" in result.columns
    assert "date" not in result.columns
    assert result.height == int_portfolio.prices.height


def test_stats_works_without_date_column(int_portfolio):
    """Portfolio.stats returns a Stats object for integer-indexed portfolios."""
    stats = int_portfolio.stats
    sharpe = stats.sharpe()["returns"]
    assert np.isfinite(sharpe)


def test_tilt_timing_decomp_works_without_date_column(int_portfolio):
    """tilt_timing_decomp returns portfolio/tilt/timing columns for integer-indexed data."""
    decomp = int_portfolio.tilt_timing_decomp
    assert "portfolio" in decomp.columns
    assert "tilt" in decomp.columns
    assert "timing" in decomp.columns
    assert "date" not in decomp.columns
    assert decomp.height == int_portfolio.prices.height
    # Numerical check: portfolio NAV ≈ tilt NAV + timing NAV - aum (decomposition identity)
    expected_portfolio = decomp["tilt"].to_numpy() + decomp["timing"].to_numpy() - int_portfolio.aum
    assert np.allclose(decomp["portfolio"].to_numpy(), expected_portfolio, rtol=1e-10, atol=1e-6)


# ─── Tilt caching tests ───────────────────────────────────────────────────────


def test_tilt_cache_is_none_before_first_access(portfolio):
    """_tilt_cache should be None before tilt is accessed."""
    assert portfolio._tilt_cache is None


def test_tilt_cache_is_set_after_first_access(portfolio):
    """_tilt_cache should be populated after tilt is accessed."""
    _ = portfolio.tilt
    assert portfolio._tilt_cache is not None


def test_tilt_returns_cached_object_on_second_access(portfolio):
    """Repeated access to tilt should return the same cached Portfolio object."""
    first = portfolio.tilt
    second = portfolio.tilt
    assert first is second
