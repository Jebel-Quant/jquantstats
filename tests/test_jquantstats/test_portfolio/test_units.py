"""Tests for the share-count view: units, trades, weights.

The numbers here are chosen so every expectation is exact in binary floating
point — prices are powers-of-ten multiples and positions are round — so a
failure means the arithmetic changed, not that a tolerance drifted.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jquantstats import Portfolio

_AUM: float = 10_000.0


@pytest.fixture
def units_portfolio():
    """Three-day, one-asset portfolio with a buy, a hold and a partial sell.

    Prices 100 → 110 → 105 against cash positions 1000 → 1100 → 525, i.e. a
    constant 10 units for two days and then a sale down to 5.
    """
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 3), interval="1d", eager=True).cast(pl.Date)
    return Portfolio(
        prices=pl.DataFrame({"date": dates, "A": pl.Series([100.0, 110.0, 105.0])}),
        cashposition=pl.DataFrame({"date": dates, "A": pl.Series([1000.0, 1100.0, 525.0])}),
        aum=_AUM,
    )


def test_units_divides_cash_by_price(units_portfolio) -> None:
    """``units`` is cashposition / prices, per asset."""
    assert units_portfolio.units["A"].to_list() == [10.0, 10.0, 5.0]


def test_units_keeps_the_date_column(units_portfolio) -> None:
    """The date axis rides along, so the frame lines up with cashposition."""
    assert units_portfolio.units.columns == ["date", "A"]


def test_units_on_a_date_free_portfolio(int_portfolio) -> None:
    """An integer-indexed portfolio yields asset columns only."""
    assert int_portfolio.units.columns == ["A", "B"]
    assert int_portfolio.units.height == int_portfolio.prices.height


def test_units_maps_a_zero_price_to_null() -> None:
    """A zero price yields null rather than an infinity."""
    pf = Portfolio(
        prices=pl.DataFrame({"A": pl.Series([100.0, 0.0])}),
        cashposition=pl.DataFrame({"A": pl.Series([1000.0, 500.0])}),
        aum=_AUM,
    )
    assert pf.units["A"].to_list() == [10.0, None]


def test_equity_is_the_cash_position(units_portfolio) -> None:
    """``equity`` is an alias, not a copy with different numbers."""
    assert units_portfolio.equity.equals(units_portfolio.cashposition)


def test_trades_units_seeds_row_zero_with_the_opening_position(units_portfolio) -> None:
    """Row 0 is the opening trade; later rows are differences of the unit count."""
    assert units_portfolio.trades_units["A"].to_list() == [10.0, 0.0, -5.0]


def test_trades_units_keeps_the_date_column(units_portfolio) -> None:
    """The date axis rides along here too."""
    assert units_portfolio.trades_units.columns == ["date", "A"]


def test_trades_currency_prices_the_unit_trades(units_portfolio) -> None:
    """Currency trades are unit trades at that row's price."""
    assert units_portfolio.trades_currency["A"].to_list() == [1000.0, 0.0, -525.0]


def test_trades_ignore_price_moves_on_an_untraded_book() -> None:
    """A price move with no trade reports zero, not phantom turnover.

    This is the reason trades are differenced in units rather than in cash: the
    cash position doubles here purely because the price doubles.
    """
    pf = Portfolio(
        prices=pl.DataFrame({"A": pl.Series([100.0, 200.0])}),
        cashposition=pl.DataFrame({"A": pl.Series([1000.0, 2000.0])}),
        aum=_AUM,
    )
    assert pf.trades_units["A"].to_list() == [10.0, 0.0]
    assert pf.trades_currency["A"].to_list() == [1000.0, 0.0]


def test_weights_divide_the_cash_position_by_nav() -> None:
    """Weights are the fraction of accumulated NAV held in each asset."""
    pf = Portfolio(
        prices=pl.DataFrame({"A": pl.Series([100.0, 100.0])}),
        cashposition=pl.DataFrame({"A": pl.Series([100.0, 100.0])}),
        aum=1000.0,
    )
    assert pf.weights["A"].to_list() == [0.1, 0.1]


def test_weights_map_a_zero_nav_to_null() -> None:
    """A NAV that reaches exactly zero yields null rather than an infinity.

    The asset halves, and the position is sized so the loss consumes the whole
    of AUM on the second row.
    """
    pf = Portfolio(
        prices=pl.DataFrame({"A": pl.Series([100.0, 50.0])}),
        cashposition=pl.DataFrame({"A": pl.Series([2000.0, 1000.0])}),
        aum=1000.0,
    )
    assert pf.nav_accumulated["NAV_accumulated"].to_list() == [1000.0, 0.0]
    assert pf.weights["A"].to_list() == [2.0, None]


def test_weights_keep_the_date_column(units_portfolio) -> None:
    """The date axis rides along for weights as well."""
    assert units_portfolio.weights.columns == ["date", "A"]


def test_units_survive_an_empty_portfolio() -> None:
    """A zero-row portfolio yields zero-row frames rather than an index error."""
    pf = Portfolio(
        prices=pl.DataFrame({"A": pl.Series([], dtype=pl.Float64)}),
        cashposition=pl.DataFrame({"A": pl.Series([], dtype=pl.Float64)}),
        aum=_AUM,
    )
    assert pf.units.height == 0
    assert pf.trades_units.height == 0
    assert pf.trades_currency.height == 0


def test_from_position_round_trips_through_units() -> None:
    """Units handed to ``from_position`` come back out of ``units`` unchanged.

    ``from_position`` converts units to cash on the way in; ``units`` converts
    back on the way out. The two must be inverses, which is what lets a
    unit-based simulator hand its positions over and read them back.
    """
    prices = pl.DataFrame({"A": pl.Series([100.0, 110.0, 105.0])})
    positions = pl.DataFrame({"A": pl.Series([10.0, 10.0, 5.0])})
    pf = Portfolio.from_position(prices=prices, position=positions, aum=_AUM)
    assert pf.units["A"].to_list() == [10.0, 10.0, 5.0]


def test_multi_asset_units_and_trades(portfolio) -> None:
    """The shared two-asset fixture resolves per asset, not just for one column."""
    units = portfolio.units
    assert units.columns == ["date", "A", "B"]
    # B is bought on day 2: 0 units, then 500/180, then 500/198.
    assert units["B"][0] == 0.0
    assert portfolio.trades_units["B"][0] == 0.0
    assert portfolio.trades_units["B"][1] > 0.0
