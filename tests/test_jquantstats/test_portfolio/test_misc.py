"""Tests for Portfolio data bridge, caching, repr/describe, and computation properties."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jquantstats import Portfolio
from jquantstats.data import Data
from jquantstats.exceptions import MissingReturnsColumnError

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def prices_single():
    """Three-day single-asset price frame (A: 100 → 110 → 121)."""
    return pl.DataFrame(
        {
            "date": pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 3), interval="1d", eager=True).cast(
                pl.Date
            ),
            "A": pl.Series([100.0, 110.0, 121.0], dtype=pl.Float64),
        }
    )


@pytest.fixture
def positions_single(prices_single):
    """Three-day cash-position frame aligned with the prices_single fixture."""
    return pl.DataFrame(
        {
            "date": prices_single["date"],
            "A": pl.Series([1000.0, 1000.0, 1000.0], dtype=pl.Float64),
        }
    )


@pytest.fixture
def portfolio_single(prices_single, positions_single):
    """Portfolio instance built from the prices_single and positions_single fixtures."""
    return Portfolio(prices=prices_single, cashposition=positions_single, aum=1e5)


# ─── Portfolio.data bridge property ──────────────────────────────────────────


def test_portfolio_data_property_returns_data_object(portfolio):
    """portfolio.data returns a legacy Data object with a 'returns' column and date index."""
    d = portfolio.data
    assert isinstance(d, Data)
    assert "returns" in d.returns.columns
    assert d.returns.height == portfolio.prices.height
    assert d.index.height == portfolio.prices.height


def test_portfolio_data_property_integer_indexed(int_portfolio):
    """portfolio.data on an integer-indexed portfolio creates a synthetic integer index."""
    d = int_portfolio.data
    assert isinstance(d, Data)
    assert "returns" in d.returns.columns
    assert "date" not in d.returns.columns
    assert d.index.columns == ["index"]
    assert d.index.height == int_portfolio.prices.height


def test_integer_indexed_stats_uses_252_periods_per_year(int_portfolio):
    """Integer-indexed portfolio.stats must use 252 periods/year, not ~31.5 million."""
    assert int_portfolio.data._periods_per_year == pytest.approx(252.0)
    sharpe = int_portfolio.stats.sharpe()
    assert "returns" in sharpe
    assert abs(sharpe["returns"]) < 1000  # sanity: not ~5600x inflated


# ─── Portfolio.as_data over derived returns frames ───────────────────────────


def test_as_data_defaults_to_the_portfolio_returns(portfolio):
    """as_data() with no argument must match the data property's series."""
    assert portfolio.as_data().returns["returns"].to_list() == portfolio.data.returns["returns"].to_list()


def test_as_data_narrows_a_derived_frame_to_the_return_series(portfolio):
    """A cost-adjusted frame carries profit and NAV columns; only 'returns' may survive."""
    adjusted = portfolio.cost_adjusted_returns(cost_bps=5.0)
    assert {"profit", "NAV_accumulated"}.issubset(adjusted.columns)

    d = portfolio.as_data(adjusted)
    assert d.returns.columns == ["returns"]
    assert d.index.columns == ["date"]
    assert d.returns["returns"].to_list() == adjusted["returns"].to_list()
    # The narrowing is the point: profit and NAV must not show up as assets.
    assert set(d.stats.sharpe()) == {"returns"}


def test_as_data_accepts_a_bare_two_column_frame(portfolio):
    """A hand-built date+returns frame needs no portfolio-specific columns."""
    bare = portfolio.returns.select(["date", "returns"])
    assert portfolio.as_data(bare).returns["returns"].to_list() == bare["returns"].to_list()


def test_as_data_normalises_a_differently_named_date_column(portfolio):
    """A caller's 'Date' column must become the index, not be dropped for a row counter."""
    renamed = portfolio.returns.select(["date", "returns"]).rename({"date": "Date"})
    d = portfolio.as_data(renamed)
    assert d.index.columns == ["date"]
    assert d.index["date"].to_list() == portfolio.returns["date"].to_list()


def test_as_data_falls_back_to_a_row_index_without_dates(portfolio):
    """A frame with no temporal column gets the synthetic integer index."""
    d = portfolio.as_data(portfolio.returns.select("returns"))
    assert d.index.columns == ["index"]
    assert d.index["index"].to_list() == list(range(portfolio.returns.height))


def test_as_data_rejects_a_frame_without_a_returns_column(portfolio):
    """A frame carrying no return series must raise rather than be analysed anyway."""
    with pytest.raises(MissingReturnsColumnError, match="no 'returns' column"):
        portfolio.as_data(portfolio.nav_accumulated)


def test_as_data_error_lists_the_available_columns(portfolio):
    """The error names the columns present, so the caller can see what they passed."""
    with pytest.raises(MissingReturnsColumnError) as excinfo:
        portfolio.as_data(portfolio.prices)
    assert excinfo.value.available == portfolio.prices.columns
    assert "'A'" in str(excinfo.value)


def test_as_data_is_not_cached(portfolio):
    """as_data must build a fresh Data per call; only the no-arg property is cached."""
    frame = portfolio.cost_adjusted_returns(cost_bps=5.0)
    assert portfolio.as_data(frame) is not portfolio.as_data(frame)


def test_cost_and_fee_ladder_through_as_data(portfolio):
    """The documented ladder — costs then fee — must reduce the Sharpe monotonically."""
    gross = portfolio.as_data().stats.sharpe()["returns"]
    net_costs = portfolio.cost_adjusted_returns(cost_bps=25.0)
    net_all = portfolio.deduct_management_fee(annual_fee=0.02, base=net_costs)

    assert portfolio.as_data(net_costs).stats.sharpe()["returns"] < gross
    assert portfolio.as_data(net_all).stats.sharpe()["returns"] < portfolio.as_data(net_costs).stats.sharpe()["returns"]


# ─── _data_bridge caching ─────────────────────────────────────────────────────


def test_data_property_returns_same_object(portfolio):
    """pf.data must return the identical Data object on repeated calls."""
    assert portfolio.data is portfolio.data


def test_data_cached_after_factory():
    """Portfolio built via from_cash_position must cache the Data bridge."""
    prices = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [100.0, 110.0]})
    pos = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [1000.0, 1000.0]})
    pf = Portfolio.from_cash_position(prices=prices, cash_position=pos, aum=1e5)
    assert pf.data is pf.data


def test_stats_plots_report_cached():
    """stats, plots, and report must return the same object on repeated access."""
    prices = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [100.0, 110.0]})
    pos = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [1000.0, 1000.0]})
    pf = Portfolio.from_cash_position(prices=prices, cash_position=pos, aum=1e5)
    assert pf.stats is pf.stats
    assert pf.plots is pf.plots
    assert pf.report is pf.report


# ─── repr and describe ────────────────────────────────────────────────────────


def test_repr(portfolio):
    """Tests that Portfolio.__repr__ returns an informative string."""
    r = repr(portfolio)
    assert r.startswith("Portfolio(assets=")
    assert "rows=" in r
    assert "start=" in r
    assert "end=" in r
    for asset in portfolio.assets:
        assert asset in r


def test_describe(portfolio):
    """Tests that Portfolio.describe() returns a tidy summary DataFrame."""
    df = portfolio.describe()
    assert "asset" in df.columns
    assert "start" in df.columns
    assert "end" in df.columns
    assert "rows" in df.columns
    assert len(df) == len(portfolio.assets)
    for asset in portfolio.assets:
        assert asset in df["asset"].to_list()


def test_repr_integer_indexed(int_portfolio):
    """Portfolio.__repr__ omits start/end for integer-indexed (no date) portfolios."""
    r = repr(int_portfolio)
    assert r.startswith("Portfolio(assets=")
    assert "rows=" in r
    assert "start=" not in r
    assert "end=" not in r


# ─── Construction factories ───────────────────────────────────────────────────


def test_from_cash_position_returns_portfolio(prices_single, positions_single):
    """Portfolio.from_cash_position returns a Portfolio instance."""
    pf = Portfolio.from_cash_position(prices=prices_single, cash_position=positions_single, aum=2e5)
    assert isinstance(pf, Portfolio)
    assert pf.aum == 2e5


def test_from_risk_position_returns_portfolio(prices_single, positions_single):
    """Portfolio.from_risk_position returns a Portfolio instance."""
    pf = Portfolio.from_risk_position(prices=prices_single, risk_position=positions_single, vola=2, aum=1e5)
    assert isinstance(pf, Portfolio)
    assert pf.assets == ["A"]


# ─── Computation properties ───────────────────────────────────────────────────


def test_portfolio_assets(portfolio_single):
    """Portfolio.assets lists numeric column names from prices."""
    assert portfolio_single.assets == ["A"]


def test_portfolio_profits_columns(portfolio_single):
    """Portfolio.profits contains the asset column."""
    assert "A" in portfolio_single.profits.columns


def test_portfolio_profit_columns(portfolio_single):
    """Portfolio.profit contains a 'profit' column."""
    assert "profit" in portfolio_single.profit.columns


def test_portfolio_nav_accumulated(portfolio_single):
    """Portfolio.nav_accumulated contains a 'NAV_accumulated' column."""
    assert "NAV_accumulated" in portfolio_single.nav_accumulated.columns


def test_portfolio_returns(portfolio_single):
    """Portfolio.returns contains a 'returns' column."""
    assert "returns" in portfolio_single.returns.columns


def test_portfolio_nav_compounded(portfolio_single):
    """Portfolio.nav_compounded contains a 'NAV_compounded' column."""
    assert "NAV_compounded" in portfolio_single.nav_compounded.columns


def test_portfolio_highwater(portfolio_single):
    """Portfolio.highwater contains a 'highwater' column."""
    assert "highwater" in portfolio_single.highwater.columns


def test_portfolio_drawdown(portfolio_single):
    """Portfolio.drawdown contains both 'drawdown' and 'drawdown_pct' columns."""
    assert "drawdown" in portfolio_single.drawdown.columns
    assert "drawdown_pct" in portfolio_single.drawdown.columns


def test_portfolio_all_columns(portfolio_single):
    """Portfolio.all merges NAV, drawdown, and compounded NAV columns."""
    df = portfolio_single.all
    assert "NAV_accumulated" in df.columns
    assert "NAV_compounded" in df.columns
    assert "drawdown" in df.columns
