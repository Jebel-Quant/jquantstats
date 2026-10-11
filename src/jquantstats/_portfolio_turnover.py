"""Turnover analytics mixin for Portfolio."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ._cache import cached_in_slot
from ._portfolio_base import _PortfolioMembers


class PortfolioTurnoverMixin(_PortfolioMembers):
    """Mixin providing turnover analytics for Portfolio."""

    if TYPE_CHECKING:
        _turnover_cache: pl.DataFrame | None

    @property
    @cached_in_slot("_turnover_cache")
    def turnover(self) -> pl.DataFrame:
        """Daily one-way portfolio turnover as a fraction of AUM.

        Computes the traded notional summed across all assets for each period,
        normalised by AUM — or, for a ``compounding`` portfolio, by the
        previous period's NAV, the same capital `returns` uses, so
        `cost_adjusted_returns` deducts costs on a consistent base.  The trade
        is measured in **units** and priced at the traded row —
        ``|Δunits| * price`` — rather than as a difference of cash positions:
        a cash position moves whenever the price moves, so ``|Δcashposition|``
        would report turnover on a book that trades nothing.

        The first row is always zero because there is no prior position to
        form a difference against.  A missing or zero price carries the units
        held across it, so a gap in the data is not mistaken for a sale and a
        rebuy.

        The result is cached after first access so repeated calls are O(1).

        Note:
            Caching is not thread-safe.  Concurrent access from multiple
            threads may trigger redundant computation, but will never produce
            incorrect results because each thread stores the same deterministic
            value.

        Returns:
            pl.DataFrame: Frame with an optional ``'date'`` column and a
            ``'turnover'`` column (dimensionless fraction of AUM).

        Examples:
            >>> from jquantstats.portfolio import Portfolio
            >>> import polars as pl
            >>> from datetime import date
            >>> _d = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
            >>> prices = pl.DataFrame({"date": _d, "A": [100.0, 110.0, 121.0]})
            >>> pos = pl.DataFrame({"date": prices["date"], "A": [1000.0, 1650.0, 1210.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)
            >>> pf.units["A"].to_list()  # buy 5 units, then sell 5
            [10.0, 15.0, 10.0]
            >>> [round(t, 6) for t in pf.turnover["turnover"].to_list()]  # 5 * 110 and 5 * 121, over aum
            [0.0, 0.0055, 0.00605]
        """
        units = self.units

        def _traded_notional(asset: str) -> pl.Series:
            """``|Δunits| * price`` for *asset*, holding units across a missing price."""
            held = units[asset].forward_fill().fill_null(0.0)
            return (held.diff().abs() * self.prices[asset]).fill_null(0.0).fill_nan(0.0)

        traded = pl.select(_traded_notional(asset) for asset in self.assets)
        daily = traded.select((pl.sum_horizontal(pl.all()) / self._return_base).alias("turnover"))
        if "date" in self.prices.columns:
            return daily.insert_column(0, self.prices["date"])
        return daily

    @property
    def turnover_weekly(self) -> pl.DataFrame:
        """Weekly aggregated one-way portfolio turnover as a fraction of AUM.

        When a ``'date'`` column is present, sums the daily turnover within
        each calendar week (Monday-based ``group_by_dynamic``).  Without a
        date column, a rolling 5-period sum with ``min_samples=5`` is returned
        (the first four rows will be ``null``).

        Returns:
            pl.DataFrame: Frame with an optional ``'date'`` column (week
            start) and a ``'turnover'`` column (fraction of AUM, summed over
            the week).
        """
        daily = self.turnover
        if "date" not in daily.columns or not daily["date"].dtype.is_temporal():
            return daily.with_columns(pl.col("turnover").rolling_sum(window_size=5, min_samples=5))
        return daily.group_by_dynamic("date", every="1w").agg(pl.col("turnover").sum()).sort("date")

    def turnover_summary(self) -> pl.DataFrame:
        """Return a summary DataFrame of turnover statistics.

        Computes three metrics from the daily turnover series:

        - ``mean_daily_turnover``: mean of daily one-way turnover (fraction
          of AUM).
        - ``mean_weekly_turnover``: mean of weekly-aggregated turnover
          (fraction of AUM).
        - ``turnover_std``: standard deviation of daily turnover (fraction of
          AUM); complements the mean to detect regime switches.

        Returns:
            pl.DataFrame: One row per metric with columns ``'metric'`` and
            ``'value'``.

        Examples:
            >>> from jquantstats.portfolio import Portfolio
            >>> import polars as pl
            >>> from datetime import date, timedelta
            >>> import numpy as np
            >>> start = date(2020, 1, 1)
            >>> dates = pl.date_range(start=start, end=start + timedelta(days=9), interval="1d", eager=True)
            >>> prices = pl.DataFrame({"date": dates, "A": pl.Series(np.ones(10) * 100.0)})
            >>> pos = pl.DataFrame({"date": dates, "A": pl.Series([float(i) * 100 for i in range(10)])})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e4)
            >>> summary = pf.turnover_summary()
            >>> list(summary["metric"])
            ['mean_daily_turnover', 'mean_weekly_turnover', 'turnover_std']
        """
        daily_col = self.turnover["turnover"]
        _mean = daily_col.mean()
        mean_daily = float(_mean) if isinstance(_mean, (int, float)) else 0.0
        _std = daily_col.std()
        std_daily = float(_std) if isinstance(_std, (int, float)) else 0.0
        weekly_col = self.turnover_weekly["turnover"].drop_nulls()
        _weekly_mean = weekly_col.mean()
        mean_weekly = (
            float(_weekly_mean) if weekly_col.len() > 0 and isinstance(_weekly_mean, (int, float)) else float("nan")
        )
        return pl.DataFrame(
            {
                "metric": ["mean_daily_turnover", "mean_weekly_turnover", "turnover_std"],
                "value": [mean_daily, mean_weekly, std_daily],
            }
        )
