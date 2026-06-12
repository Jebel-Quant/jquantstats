"""Tests for Portfolio turnover calculations (daily, weekly, summary)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from jquantstats import Portfolio

from ..tolerances import TOL_FLOAT64, TOL_PINNED


def test_turnover_columns_and_length(turnover_portfolio):
    """Turnover DataFrame should have 'date' and 'turnover' columns with same row count as portfolio."""
    to = turnover_portfolio.turnover
    assert "date" in to.columns
    assert "turnover" in to.columns
    assert to.height == turnover_portfolio.prices.height


def test_turnover_first_row_is_zero(turnover_portfolio):
    """The first row of daily turnover must be 0.0 (no prior position)."""
    to = turnover_portfolio.turnover
    assert float(to["turnover"][0]) == pytest.approx(0.0, abs=TOL_FLOAT64)


def test_turnover_subsequent_rows_correct(turnover_portfolio):
    """Daily turnover from row 1 onward: (|ΔA| + |ΔB|) / AUM = 0.015."""
    to = turnover_portfolio.turnover
    expected = 0.015  # (100 + 50) / 10_000
    assert np.allclose(to["turnover"][1:].to_numpy(), expected, rtol=TOL_FLOAT64, atol=TOL_FLOAT64)


def test_turnover_nonnegative(turnover_portfolio):
    """All turnover values must be non-negative."""
    to = turnover_portfolio.turnover
    assert (to["turnover"] >= 0.0).all()


def test_turnover_without_date_column(int_portfolio):
    """Turnover on an integer-indexed portfolio has only a 'turnover' column."""
    to = int_portfolio.turnover
    assert "date" not in to.columns
    assert "turnover" in to.columns
    assert to.height == int_portfolio.prices.height
    assert float(to["turnover"][0]) == pytest.approx(0.0, abs=TOL_FLOAT64)


def test_turnover_constant_positions_are_zero():
    """Constant cash positions → zero turnover on every row after the first."""
    n = 5
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    pf = Portfolio.from_cash_position(
        prices=pl.DataFrame({"date": dates, "A": pl.Series([100.0] * n)}),
        cash_position=pl.DataFrame({"date": dates, "A": pl.Series([500.0] * n, dtype=pl.Float64)}),
        aum=1e4,
    )
    to = pf.turnover
    assert np.allclose(to["turnover"].to_numpy(), 0.0, atol=TOL_FLOAT64)


def test_turnover_weekly_columns_when_date_present(turnover_portfolio):
    """turnover_weekly with a date column should have 'date' and 'turnover' columns."""
    wto = turnover_portfolio.turnover_weekly
    assert "date" in wto.columns
    assert "turnover" in wto.columns


def test_turnover_weekly_sums_daily(turnover_portfolio):
    """Weekly turnover must equal the sum of daily turnovers within the same week."""
    daily = turnover_portfolio.turnover
    weekly = turnover_portfolio.turnover_weekly
    assert np.isclose(float(daily["turnover"].sum()), float(weekly["turnover"].sum()), rtol=TOL_PINNED)


def test_turnover_weekly_without_date_uses_rolling(int_portfolio):
    """Without a date column, turnover_weekly uses a rolling 5-period sum."""
    wto = int_portfolio.turnover_weekly
    assert "date" not in wto.columns
    assert "turnover" in wto.columns
    # First 4 rows should be null (rolling window with min_samples=5)
    assert wto["turnover"][:4].null_count() == 4
    # Row index 4 should be non-null
    assert wto["turnover"][4] is not None


def test_turnover_summary_structure(turnover_portfolio):
    """turnover_summary should return a DataFrame with metric and value columns."""
    summary = turnover_portfolio.turnover_summary()
    assert list(summary["metric"]) == ["mean_daily_turnover", "mean_weekly_turnover", "turnover_std"]
    assert "value" in summary.columns
    assert summary["value"].is_finite().all()


def test_turnover_summary_mean_daily_matches_manual(turnover_portfolio):
    """mean_daily_turnover in summary should match the manual mean of the turnover series."""
    summary = turnover_portfolio.turnover_summary()
    manual_mean = float(turnover_portfolio.turnover["turnover"].mean())
    row = summary.filter(pl.col("metric") == "mean_daily_turnover")
    assert float(row["value"][0]) == pytest.approx(manual_mean, rel=TOL_PINNED)


def test_turnover_summary_std_matches_manual(turnover_portfolio):
    """turnover_std in summary should match the manual std of the turnover series."""
    summary = turnover_portfolio.turnover_summary()
    manual_std = float(turnover_portfolio.turnover["turnover"].std())
    row = summary.filter(pl.col("metric") == "turnover_std")
    assert float(row["value"][0]) == pytest.approx(manual_std, rel=TOL_PINNED)


def test_turnover_summary_mean_weekly_positive(turnover_portfolio):
    """mean_weekly_turnover must be positive for a portfolio with non-zero trading."""
    summary = turnover_portfolio.turnover_summary()
    row = summary.filter(pl.col("metric") == "mean_weekly_turnover")
    assert float(row["value"][0]) > 0.0


def test_turnover_summary_without_date_column(int_portfolio):
    """turnover_summary should work for integer-indexed portfolios."""
    summary = int_portfolio.turnover_summary()
    assert list(summary["metric"]) == ["mean_daily_turnover", "mean_weekly_turnover", "turnover_std"]


# ─── Caching tests ────────────────────────────────────────────────────────────


def test_turnover_cache_is_none_before_first_access(turnover_portfolio):
    """_turnover_cache should be None before turnover is accessed."""
    assert turnover_portfolio._turnover_cache is None


def test_turnover_cache_is_set_after_first_access(turnover_portfolio):
    """_turnover_cache should be populated after turnover is accessed."""
    _ = turnover_portfolio.turnover
    assert turnover_portfolio._turnover_cache is not None


def test_turnover_returns_cached_object_on_second_access(turnover_portfolio):
    """Repeated access to turnover should return the same cached DataFrame object."""
    first = turnover_portfolio.turnover
    second = turnover_portfolio.turnover
    assert first is second
