"""Units, trades and weights mixin for Portfolio.

`Portfolio` stores *cash* positions, because every analytic downstream of it —
NAV, returns, turnover, cost — is denominated in currency. The share-count view
is the other half of the same picture, and a simulator that decides *how many
units to hold* needs it back: how many units are held, how many changed hands
between two rows, and what fraction of NAV each position represents.

Everything here is **derived**, never stored. ``units`` is ``cashposition /
prices``, so the whole surface is available on a portfolio built through any
constructor — `from_cash_position`, `from_position`, or `from_risk_position` —
rather than only on one that happened to be handed units to begin with.

Trades are deliberately computed in *units*, not as a difference of cash
positions. A cash position moves when the price moves, with no trade taking
place at all, so ``cashposition.diff()`` would report phantom turnover on a
buy-and-hold book. The difference is taken on the share count and only then
converted back to currency at the traded price.
"""

from __future__ import annotations

import polars as pl

from ._portfolio_base import _PortfolioMembers


class PortfolioUnitsMixin(_PortfolioMembers):
    """Mixin providing the share-count view of a Portfolio: units, trades, weights."""

    def _numeric_frame(self, values: pl.DataFrame) -> pl.DataFrame:
        """Return *values* carrying the date column, if the portfolio has one.

        The mixin's properties all build a frame of per-asset numbers and then
        need the same date column re-attached. Centralising it keeps every
        property returning a frame shaped like ``cashposition``.

        Args:
            values: Frame of per-asset columns, without a date column.

        Returns:
            *values* with the portfolio's ``'date'`` column prepended when one
            exists, otherwise *values* unchanged.
        """
        if "date" not in self.prices.columns:
            return values
        return values.insert_column(0, self.prices["date"])

    @property
    def units(self) -> pl.DataFrame:
        """Number of units held per asset over time.

        Derived as ``cashposition / prices``. A zero price yields a null rather
        than an infinity, so a delisted or not-yet-listed asset does not poison
        the trade and weight frames built on top of this one.

        Returns:
            pl.DataFrame: Units per asset, with the ``'date'`` column when the
            portfolio has one.

        Examples:
            >>> import polars as pl
            >>> from jquantstats.portfolio import Portfolio
            >>> prices = pl.DataFrame({"A": [100.0, 110.0, 105.0]})
            >>> pos = pl.DataFrame({"A": [1000.0, 1100.0, 1050.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e6)
            >>> pf.units["A"].to_list()
            [10.0, 10.0, 10.0]
        """
        values = pl.select(
            pl.when(self.prices[asset] != 0.0)
            .then(self.cashposition[asset] / self.prices[asset])
            .otherwise(None)
            .alias(asset)
            for asset in self.assets
        )
        return self._numeric_frame(values)

    @property
    def equity(self) -> pl.DataFrame:
        """Cash value of each position over time.

        An alias for ``cashposition``, kept because "equity" is the term the
        simulator vocabulary uses for the same quantity. Not to be confused with
        the equity asset class.

        Returns:
            pl.DataFrame: The portfolio's cash positions, unchanged.
        """
        return self.cashposition

    @property
    def trades_units(self) -> pl.DataFrame:
        """Units bought (positive) or sold (negative) at each step.

        The first row is the opening position: there is no prior row to
        difference against, and treating it as zero would hide the initial
        trade that established the book.

        Returns:
            pl.DataFrame: Unit trades per asset, with the ``'date'`` column when
            the portfolio has one.

        Examples:
            >>> import polars as pl
            >>> from jquantstats.portfolio import Portfolio
            >>> prices = pl.DataFrame({"A": [100.0, 100.0, 100.0]})
            >>> pos = pl.DataFrame({"A": [1000.0, 1500.0, 500.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e6)
            >>> pf.trades_units["A"].to_list()
            [10.0, 5.0, -10.0]
        """
        units = self.units

        def _trades(asset: str) -> pl.Series:
            """Differences of *asset*'s unit series, seeded with the opening position."""
            held = units[asset].fill_null(0.0)
            # `diff` leaves exactly one null, at row 0; filling it with the
            # opening position records the trade that established the book.
            return held.diff().fill_null(held[0]) if held.len() else held

        values = pl.select(_trades(asset).alias(asset) for asset in self.assets)
        return self._numeric_frame(values)

    @property
    def trades_currency(self) -> pl.DataFrame:
        """Cash value of the trades at each step, priced at the traded row.

        Computed as ``trades_units * prices`` rather than as a difference of
        cash positions, so a price move on an untraded book reports zero rather
        than phantom turnover.

        Returns:
            pl.DataFrame: Currency trades per asset, with the ``'date'`` column
            when the portfolio has one. Positive values are buys (cash out),
            negative values are sells (cash in).

        Examples:
            >>> import polars as pl
            >>> from jquantstats.portfolio import Portfolio
            >>> prices = pl.DataFrame({"A": [100.0, 200.0]})
            >>> pos = pl.DataFrame({"A": [1000.0, 2000.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e6)
            >>> pf.trades_currency["A"].to_list()
            [1000.0, 0.0]
        """
        trades = self.trades_units
        values = pl.select((trades[asset] * self.prices[asset]).alias(asset) for asset in self.assets)
        return self._numeric_frame(values)

    @property
    def weights(self) -> pl.DataFrame:
        """Fraction of NAV held in each asset over time.

        Each cash position divided by the accumulated NAV of the same row. For a
        fully invested, unlevered book the weights sum to 1.0; short positions
        are negative.

        Returns:
            pl.DataFrame: Weights per asset, with the ``'date'`` column when the
            portfolio has one.

        Examples:
            >>> import polars as pl
            >>> from jquantstats.portfolio import Portfolio
            >>> prices = pl.DataFrame({"A": [100.0, 100.0]})
            >>> pos = pl.DataFrame({"A": [100.0, 100.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1000.0)
            >>> pf.weights["A"].to_list()
            [0.1, 0.1]
        """
        nav = self.nav_accumulated["NAV_accumulated"]
        values = pl.select(
            pl.when(nav != 0.0).then(self.cashposition[asset] / nav).otherwise(None).alias(asset)
            for asset in self.assets
        )
        return self._numeric_frame(values)
