"""NAV & returns chain mixin for Portfolio."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from .exceptions import MissingDateColumnError, NoAssetColumnsError


class PortfolioNavMixin:
    """Mixin providing NAV & returns chain properties for Portfolio."""

    if TYPE_CHECKING:
        prices: pl.DataFrame
        cashposition: pl.DataFrame
        aum: float
        _profits_cache: pl.DataFrame | None
        _returns_cache: pl.DataFrame | None

        @staticmethod
        def _assert_clean_series(series: pl.Series, name: str = "") -> None:
            """Raise ValueError if *series* contains nulls or non-finite values."""
            ...

    @property
    def profits(self) -> pl.DataFrame:
        """Compute per-asset daily cash profits, preserving non-numeric columns.

        Returns:
            pl.DataFrame: Per-asset daily profit series along with any
            non-numeric columns (e.g., ``'date'``).

        The result is cached after first access so repeated calls are O(1).

        Note:
            Caching is not thread-safe.  Concurrent access from multiple
            threads may trigger redundant computation, but will never produce
            incorrect results because each thread stores the same deterministic
            value.

        Examples:
            >>> from jquantstats.portfolio import Portfolio
            >>> import polars as pl
            >>> prices = pl.DataFrame({"A": [100.0, 110.0, 105.0]})
            >>> pos = pl.DataFrame({"A": [1000.0, 1000.0, 1000.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e6)
            >>> pf.profits.columns
            ['A']
        """
        cache = getattr(self, "_profits_cache", None)
        if cache is not None:
            return cache

        assets = [c for c in self.prices.columns if self.prices[c].dtype.is_numeric()]

        result = self.prices.with_columns(
            (self.prices[asset].pct_change().fill_null(0.0) * self.cashposition[asset].shift(n=1).fill_null(0.0)).alias(
                asset
            )
            for asset in assets
        )

        if assets:
            result = result.with_columns(
                pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(0.0).fill_null(0.0).alias(c) for c in assets
            )

        # Direct write is safe: Portfolio is a frozen, slotted dataclass that
        # declares every cache field, so object.__setattr__ cannot fail here.
        object.__setattr__(self, "_profits_cache", result)
        return result

    @property
    def profit(self) -> pl.DataFrame:
        """Return total daily portfolio profit including the ``'date'`` column.

        Aggregates per-asset profits into a single ``'profit'`` column and
        validates that no day's total profit is NaN/null.

        Raises:
            NoAssetColumnsError: If the profits frame has no numeric asset columns.
        """
        df_profits = self.profits
        assets = [c for c in df_profits.columns if df_profits[c].dtype.is_numeric()]

        if not assets:
            raise NoAssetColumnsError("profits")

        non_assets = [c for c in df_profits.columns if c not in set(assets)]

        portfolio_daily_profit = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in assets]).alias("profit")
        result = df_profits.select([*non_assets, portfolio_daily_profit])

        self._assert_clean_series(series=result["profit"])
        return result

    @property
    def nav_accumulated(self) -> pl.DataFrame:
        """Compute cumulative additive NAV of the portfolio, preserving ``'date'``."""
        return self.profit.with_columns((pl.col("profit").cum_sum() + self.aum).alias("NAV_accumulated"))

    @property
    def returns(self) -> pl.DataFrame:
        """Return daily returns as profit scaled by AUM, preserving ``'date'``.

        The returned DataFrame contains the original ``'date'`` column with the
        ``'profit'`` column scaled by AUM (i.e., per-period returns), and also
        an additional convenience column named ``'returns'`` with the same
        values for downstream consumers.

        The result is cached after first access so repeated calls are O(1).

        Note:
            Caching is not thread-safe.  Concurrent access from multiple
            threads may trigger redundant computation, but will never produce
            incorrect results because each thread stores the same deterministic
            value.
        """
        cache = getattr(self, "_returns_cache", None)
        if cache is not None:
            return cache
        result = self.nav_accumulated.with_columns(
            (pl.col("profit") / self.aum).alias("returns"),
        )
        # Direct write is safe: see the comment on the profits cache above.
        object.__setattr__(self, "_returns_cache", result)
        return result

    @property
    def monthly(self) -> pl.DataFrame:
        """Return monthly compounded returns and calendar columns.

        Aggregates daily returns (profit/AUM) by calendar month and computes
        the compounded monthly return: prod(1 + r_d) - 1. The resulting frame
        includes:

        - ``date``: month-end label as a Polars Date (end of the grouping window)
        - ``returns``: compounded monthly return
        - ``NAV_accumulated``: last NAV within the month
        - ``profit``: summed profit within the month
        - ``year``: integer year (e.g., 2020)
        - ``month``: integer month number (1-12)
        - ``month_name``: abbreviated month name (e.g., ``"Jan"``, ``"Feb"``)

        Raises:
            MissingDateColumnError: If the portfolio data has no ``'date'``
                column.
        """
        if "date" not in self.prices.columns:
            raise MissingDateColumnError("monthly")
        daily = self.returns.select(["date", "returns", "profit", "NAV_accumulated"])
        monthly = (
            daily.group_by_dynamic(
                "date",
                every="1mo",
                period="1mo",
                label="left",
                closed="right",
            )
            .agg(
                [
                    pl.col("profit").sum().alias("profit"),
                    pl.col("NAV_accumulated").last().alias("NAV_accumulated"),
                    (pl.col("returns") + 1.0).product().alias("gross"),
                ]
            )
            .with_columns((pl.col("gross") - 1.0).alias("returns"))
            .select(["date", "returns", "NAV_accumulated", "profit"])
            .with_columns(
                [
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month"),
                    pl.col("date").dt.strftime("%b").alias("month_name"),
                ]
            )
            .sort("date")
        )
        return monthly

    @property
    def nav_compounded(self) -> pl.DataFrame:
        """Compute compounded NAV from returns (profit/AUM), preserving ``'date'``."""
        return self.returns.with_columns(((pl.col("returns") + 1.0).cum_prod() * self.aum).alias("NAV_compounded"))

    @property
    def highwater(self) -> pl.DataFrame:
        """Return the cumulative maximum of NAV as the high-water mark series.

        The resulting DataFrame preserves the ``'date'`` column and adds a
        ``'highwater'`` column computed as the cumulative maximum of
        ``'NAV_accumulated'``.
        """
        return self.returns.with_columns(pl.col("NAV_accumulated").cum_max().alias("highwater"))

    @property
    def drawdown(self) -> pl.DataFrame:
        """Return drawdown as the distance from high-water mark to current NAV.

        Computes ``'drawdown'`` = ``'highwater'`` - ``'NAV_accumulated'`` and
        preserves the ``'date'`` column alongside the intermediate columns.
        """
        return self.highwater.with_columns(
            (pl.col("highwater") - pl.col("NAV_accumulated")).alias("drawdown"),
            ((pl.col("highwater") - pl.col("NAV_accumulated")) / pl.col("highwater")).alias("drawdown_pct"),
        )

    @property
    def all(self) -> pl.DataFrame:
        """Return a merged view of drawdown and compounded NAV.

        When a ``'date'`` column is present the two frames are joined on that
        column to ensure temporal alignment.  When the data is integer-indexed
        (no ``'date'`` column) the frames are stacked horizontally — they are
        guaranteed to have identical row counts because both are derived from
        the same source portfolio.
        """
        left = self.drawdown
        if "date" in left.columns:
            right = self.nav_compounded.select(["date", "NAV_compounded"])
            return left.join(right, on="date", how="inner")
        else:
            right = self.nav_compounded.select(["NAV_compounded"])
            return left.hstack(right)
