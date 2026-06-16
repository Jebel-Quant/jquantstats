"""Cost analysis mixin for Portfolio."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

from .exceptions import InvalidMaxBpsError, NegativeCostBpsError

if TYPE_CHECKING:
    from .data import Data


class PortfolioCostMixin:
    """Mixin providing cost analysis methods for Portfolio."""

    if TYPE_CHECKING:
        cashposition: pl.DataFrame
        aum: float
        cost_per_unit: float
        cost_bps: float

        @property
        def data(self) -> Data:
            """Defined on Portfolio."""

        @property
        def returns(self) -> pl.DataFrame:
            """Defined on PortfolioNavMixin."""

        @property
        def turnover(self) -> pl.DataFrame:
            """Defined on PortfolioTurnoverMixin."""

        @property
        def profit(self) -> pl.DataFrame:
            """Defined on PortfolioNavMixin."""

    @property
    def position_delta_costs(self) -> pl.DataFrame:
        """Daily trading cost using the position-delta model.

        Computes the per-period cost as::

            cost_t = sum_i( |x_{i,t} - x_{i,t-1}| ) * cost_per_unit

        where ``x_{i,t}`` is the cash position in asset *i* at time *t* and
        ``cost_per_unit`` is the one-way cost per unit of traded notional.
        The first row is always zero because there is no prior position to
        form a difference against.

        Returns:
            pl.DataFrame: Frame with an optional ``'date'`` column and a
            ``'cost'`` column (absolute cash cost per period).

        Examples:
            >>> from jquantstats.portfolio import Portfolio
            >>> import polars as pl
            >>> from datetime import date
            >>> _d = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
            >>> prices = pl.DataFrame({"date": _d, "A": [100.0, 110.0, 121.0]})
            >>> pos = pl.DataFrame({"date": _d, "A": [1000.0, 1200.0, 900.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e5, cost_per_unit=0.01)
            >>> pf.position_delta_costs["cost"].to_list()
            [0.0, 2.0, 3.0]
        """
        assets = [c for c in self.cashposition.columns if c != "date" and self.cashposition[c].dtype.is_numeric()]
        abs_position_changes = pl.sum_horizontal(pl.col(c).diff().abs().fill_null(0.0).fill_nan(0.0) for c in assets)
        daily_cost = (abs_position_changes * self.cost_per_unit).alias("cost")
        cols: list[str | pl.Expr] = []
        if "date" in self.cashposition.columns:
            cols.append("date")
        cols.append(daily_cost)
        return self.cashposition.select(cols)

    @property
    def net_cost_nav(self) -> pl.DataFrame:
        """Net-of-cost cumulative additive NAV using the position-delta cost model.

        Deducts `position_delta_costs` from daily portfolio profit and
        computes the running cumulative sum offset by AUM.  The result
        represents the realised NAV path a strategy would achieve after paying
        ``cost_per_unit`` on every unit of position change.

        When ``cost_per_unit`` is zero the result equals `nav_accumulated`.

        Returns:
            pl.DataFrame: Frame with an optional ``'date'`` column,
            ``'profit'``, ``'cost'``, and ``'NAV_accumulated_net'`` columns.

        Examples:
            >>> from jquantstats.portfolio import Portfolio
            >>> import polars as pl
            >>> from datetime import date
            >>> _d = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
            >>> prices = pl.DataFrame({"date": _d, "A": [100.0, 110.0, 121.0]})
            >>> pos = pl.DataFrame({"date": _d, "A": [1000.0, 1200.0, 900.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e5, cost_per_unit=0.0)
            >>> net = pf.net_cost_nav
            >>> list(net.columns)
            ['date', 'profit', 'cost', 'NAV_accumulated_net']
        """
        profit_df = self.profit
        cost_df = self.position_delta_costs
        if "date" in profit_df.columns:
            df = profit_df.join(cost_df, on="date", how="left")
        else:
            df = profit_df.hstack(cost_df.select(["cost"]))
        return df.with_columns(((pl.col("profit") - pl.col("cost")).cum_sum() + self.aum).alias("NAV_accumulated_net"))

    def cost_adjusted_returns(self, cost_bps: float | None = None) -> pl.DataFrame:
        """Return daily portfolio returns net of estimated one-way trading costs.

        Trading costs are modelled as a linear function of daily one-way
        turnover: for every unit of AUM traded, the strategy incurs
        ``cost_bps`` basis points (i.e. ``cost_bps / 10_000`` fractional
        cost). The daily cost deduction is therefore::

            daily_cost = turnover * (cost_bps / 10_000)

        where ``turnover`` is the fraction-of-AUM one-way turnover already
        computed by `turnover`.  The deduction is applied to the
        ``returns`` column of `returns`, leaving all other columns
        (including ``date``) untouched.

        Args:
            cost_bps: One-way trading cost in basis points per unit of AUM
                traded.  Must be non-negative.  Defaults to ``self.cost_bps``
                set at construction time.

        Returns:
            pl.DataFrame: Same schema as `returns` but with the
            ``returns`` column reduced by the per-period trading cost.

        Raises:
            TypeError: If ``cost_bps`` is not a number.
            ValueError: If ``cost_bps`` is not finite (NaN or infinity).
            NegativeCostBpsError: If ``cost_bps`` is negative.

        Examples:
            >>> from jquantstats.portfolio import Portfolio
            >>> import polars as pl
            >>> from datetime import date
            >>> _d = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
            >>> prices = pl.DataFrame({"date": _d, "A": [100.0, 110.0, 121.0]})
            >>> pos = pl.DataFrame({"date": _d, "A": [1000.0, 1200.0, 900.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)
            >>> adj = pf.cost_adjusted_returns(0.0)
            >>> float(adj["returns"][1]) == float(pf.returns["returns"][1])
            True
        """
        effective_bps = cost_bps if cost_bps is not None else self.cost_bps
        if isinstance(effective_bps, bool) or not isinstance(effective_bps, int | float):
            raise TypeError(f"cost_bps must be a number, got {type(effective_bps).__name__}")  # noqa: TRY003
        effective_bps = float(effective_bps)
        if not math.isfinite(effective_bps):
            raise ValueError(f"cost_bps must be finite, got {effective_bps}")  # noqa: TRY003
        if effective_bps < 0:
            raise NegativeCostBpsError(effective_bps)
        base = self.returns
        daily_cost = self.turnover["turnover"] * (effective_bps / 10_000.0)
        return base.with_columns((pl.col("returns") - daily_cost).alias("returns"))

    def trading_cost_impact(self, max_bps: int = 20) -> pl.DataFrame:
        """Estimate the impact of trading costs on the Sharpe ratio.

        Computes the annualised Sharpe ratio of cost-adjusted returns for
        each integer cost level from 0 up to and including ``max_bps`` basis
        points (1 bp = 0.01 %).  The result lets you quickly assess at what
        cost level the strategy's edge is eroded.

        Args:
            max_bps: Maximum one-way trading cost to evaluate, in basis
                points.  Defaults to 20 (i.e., evaluates 0, 1, 2, …, 20
                bps).  Must be a positive integer.

        Returns:
            pl.DataFrame: Frame with columns ``'cost_bps'`` (Int64) and
            ``'sharpe'`` (Float64), one row per cost level from 0 to
            ``max_bps`` inclusive.

        Raises:
            InvalidMaxBpsError: If ``max_bps`` is not a positive integer.

        Examples:
            >>> from jquantstats.portfolio import Portfolio
            >>> import polars as pl
            >>> from datetime import date, timedelta
            >>> import numpy as np
            >>> start = date(2020, 1, 1)
            >>> dates = pl.date_range(
            ...     start=start, end=start + timedelta(days=99), interval="1d", eager=True
            ... )
            >>> rng = np.random.default_rng(0)
            >>> prices = pl.DataFrame({
            ...     "date": dates,
            ...     "A": pl.Series(np.cumprod(1 + rng.normal(0.001, 0.01, 100)) * 100),
            ... })
            >>> pos = pl.DataFrame({"date": dates, "A": pl.Series(np.ones(100) * 1000.0)})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)
            >>> impact = pf.trading_cost_impact(max_bps=5)
            >>> list(impact["cost_bps"])
            [0, 1, 2, 3, 4, 5]
        """
        if not isinstance(max_bps, int) or max_bps < 1:
            raise InvalidMaxBpsError(max_bps)
        import numpy as np

        from ._stats._core import _std_is_negligible

        periods = self.data._periods_per_year  # one Data object, outside the loop
        sqrt_periods = float(np.sqrt(periods))
        cost_levels = list(range(max_bps + 1))

        # Extract base returns and turnover once — O(1) allocations regardless of max_bps
        base_rets = self.returns["returns"]
        turnover_s = self.turnover["turnover"]

        # Build all cost-adjusted return columns in one vectorised DataFrame construction,
        # then compute means and stds in a single aggregate pass (no per-iteration allocation).
        sweep = pl.DataFrame({str(bps): base_rets - turnover_s * (bps / 10_000.0) for bps in cost_levels})
        means_row = sweep.mean().row(0)
        stds_row = sweep.std(ddof=1).row(0)

        sharpe_values: list[float] = []
        for mean_raw, std_raw in zip(means_row, stds_row, strict=False):
            mean_val = 0.0 if mean_raw is None else float(mean_raw)
            if _std_is_negligible(std_raw, mean_val):
                sharpe_values.append(float("nan"))
            else:
                sharpe_values.append(mean_val / float(std_raw) * sqrt_periods)
        return pl.DataFrame({"cost_bps": pl.Series(cost_levels, dtype=pl.Int64), "sharpe": pl.Series(sharpe_values)})
