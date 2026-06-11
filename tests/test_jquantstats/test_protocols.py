"""Tests for internal Protocol/interface definitions.

Verifies that concrete implementations satisfy each @runtime_checkable protocol
and that the protocol modules are importable.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jquantstats import Portfolio
from jquantstats._plots._protocol import DataLike as PlotsDataLike
from jquantstats._plots._protocol import PortfolioLike as PlotsPortfolioLike
from jquantstats._protocol import DataLike as RootDataLike
from jquantstats._protocol import DataLike as StatsDataLike  # stats mixins now use the root protocol
from jquantstats._protocol import StatsLike as RootStatsLike
from jquantstats._reports._protocol import DataLike as ReportsDataLike
from jquantstats._reports._protocol import PortfolioLike as ReportsPortfolioLike
from jquantstats._reports._protocol import StatsLike as ReportsStatsLike
from jquantstats._utils._protocol import DataLike as UtilsDataLike


@pytest.fixture
def simple_portfolio():
    """Minimal two-asset Portfolio for protocol checks."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 5), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates, "A": pl.Series([100.0, 101.0, 102.0, 103.0, 104.0], dtype=pl.Float64)})
    positions = pl.DataFrame({"date": dates, "A": pl.Series([1000.0] * 5, dtype=pl.Float64)})
    return Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5)


def test_data_satisfies_stats_data_like(data):
    """Data satisfies the StatsDataLike protocol."""
    assert isinstance(data, StatsDataLike)


def test_data_satisfies_plots_data_like(data):
    """Data satisfies the PlotsDataLike protocol."""
    assert isinstance(data, PlotsDataLike)


def test_data_satisfies_reports_data_like(data):
    """Data satisfies the ReportsDataLike protocol."""
    assert isinstance(data, ReportsDataLike)


def test_data_like_protocol_is_shared_across_subpackages():
    """Plots/Reports/Utils DataLike names point to the root DataLike protocol."""
    assert PlotsDataLike is RootDataLike
    assert ReportsDataLike is RootDataLike
    assert UtilsDataLike is RootDataLike


def test_stats_like_protocol_is_shared():
    """Reports StatsLike points to the root StatsLike protocol."""
    assert ReportsStatsLike is RootStatsLike


def test_portfolio_satisfies_plots_portfolio_like(simple_portfolio):
    """Portfolio satisfies the PlotsPortfolioLike protocol."""
    assert isinstance(simple_portfolio, PlotsPortfolioLike)


def test_portfolio_satisfies_reports_portfolio_like(simple_portfolio):
    """Portfolio satisfies the ReportsPortfolioLike protocol."""
    assert isinstance(simple_portfolio, ReportsPortfolioLike)


def test_reports_stats_like_protocol_is_minimal():
    """Reports StatsLike only defines methods used by Report."""
    public_members = {name for name in ReportsStatsLike.__dict__ if not name.startswith("_")}
    assert public_members == {"summary"}


def test_non_data_does_not_satisfy_stats_protocol():
    """A plain object does not satisfy StatsDataLike."""
    assert not isinstance("not-a-data", StatsDataLike)


def test_non_data_does_not_satisfy_plots_protocol():
    """A plain object does not satisfy PlotsDataLike."""
    assert not isinstance(42, PlotsDataLike)
