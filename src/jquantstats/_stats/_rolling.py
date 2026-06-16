"""Rolling-window statistical metrics for financial returns data."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ._core import _to_float
from ._internals import _annualization_factor
from ._performance import _RiskStatsMixin

if TYPE_CHECKING:
    from ..data import Data

# ── Rolling statistics mixin ─────────────────────────────────────────────────


class _RollingStatsMixin:
    """Mixin class providing rolling-window financial statistics methods.

    Separates rolling-window computations from the core point-in-time metrics
    in `_core`.  The concrete `Stats` class inherits from both.
    """

    _data: Data
    all: pl.DataFrame

    if TYPE_CHECKING:
        from .._protocol import DataLike

        data: DataLike

    def implied_volatility(self, periods: int = 252, annualize: bool = True) -> pl.DataFrame | dict[str, float]:
        """Calculate implied volatility using log returns.

        Uses log returns (ln(1 + r)) instead of simple returns for mathematical
        correctness with continuous compounding.

        When ``annualize=True`` (default), returns a rolling DataFrame of
        annualised log-return volatility: ``rolling_std(periods) * sqrt(periods)``.
        When ``annualize=False``, returns a scalar standard deviation per asset.

        Args:
            periods (int): Rolling window size and annualisation factor. Defaults to 252.
            annualize (bool): Whether to annualize and return a rolling series.
                Defaults to True.

        Returns:
            pl.DataFrame: Rolling annualised implied volatility (one column per
                asset) when ``annualize=True``.
            dict[str, float]: Scalar log-return std per asset when
                ``annualize=False``.

        """
        if annualize:
            scale = _annualization_factor(periods)
            return self.all.select(
                [pl.col(name) for name in self._data.date_col]
                + [
                    ((1.0 + pl.col(col)).log(math.e).rolling_std(window_size=periods) * scale).alias(col)
                    for col, _ in self._data.items()
                ]
            )
        return {
            col: _to_float((1.0 + series.cast(pl.Float64)).log(math.e).cast(pl.Float64).std())
            for col, series in self._data.items()
        }

    @staticmethod
    def _pct_rank_series(s: pl.Series) -> float:
        """Percentile rank of the last element among all elements (pandas average method).

        Args:
            s (pl.Series): Window of price values.

        Returns:
            float: Rank of s[-1] in [0, 100].

        """
        arr = s.to_numpy()
        current = arr[-1]
        n = len(arr)
        below = float(np.sum(arr < current))
        equal = float(np.sum(arr == current))
        return (below + (equal + 1) / 2) / n * 100.0

    def pct_rank(self, window: int = 60) -> pl.DataFrame:
        """Calculate the rolling percentile rank of prices within a window.

        Converts returns to a cumulative price series, then for each period
        returns the percentile rank (0-100) of the current price within the
        trailing ``window`` prices.  Matches ``qs.stats.pct_rank`` (pandas
        ``rank(pct=True)`` with ``method='average'``).

        Args:
            window (int): Rolling window size. Defaults to 60.

        Returns:
            pl.DataFrame: Date column(s) plus one percentile-rank column per asset.

        Raises:
            ValueError: If window is not a positive integer.

        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer")  # noqa: TRY003

        cols: list[pl.Expr | pl.Series] = [pl.col(name) for name in self._data.date_col]
        for col, series in self._data.items():
            prices = _RiskStatsMixin.prices(series)
            ranked = prices.rolling_map(
                function=self._pct_rank_series,
                window_size=window,
            ).alias(col)
            cols.append(ranked)

        return self.all.select(cols)

    def rolling_sortino(
        self,
        rolling_period: int = 126,
        periods_per_year: int | float | None = None,
    ) -> pl.DataFrame:
        """Calculate the rolling Sortino ratio.

        Args:
            rolling_period: Rolling window size. Defaults to 126.
            periods_per_year: Periods per year for annualisation.

        Returns:
            pl.DataFrame: Date column(s) plus one annualised rolling Sortino
            column per asset.

        Raises:
            ValueError: If rolling_period is not a positive integer.

        """
        if not isinstance(rolling_period, int) or rolling_period <= 0:
            raise ValueError("rolling_period must be a positive integer")  # noqa: TRY003
        ppy = periods_per_year or self._data._periods_per_year
        scale = _annualization_factor(ppy)
        exprs: list[pl.Expr] = []
        for col, _ in self._data.items():
            mean_ret = pl.col(col).rolling_mean(window_size=rolling_period)
            negative_squared = pl.when(pl.col(col) < 0).then(pl.col(col) ** 2).otherwise(0.0)
            downside = negative_squared.rolling_mean(window_size=rolling_period)
            exprs.append(((mean_ret / downside.sqrt()) * scale).alias(col))
        return self.all.select([pl.col(name) for name in self._data.date_col] + exprs)

    def rolling_sharpe(
        self,
        rolling_period: int = 126,
        periods_per_year: int | float | None = None,
    ) -> pl.DataFrame:
        """Calculate the rolling Sharpe ratio.

        Args:
            rolling_period: Rolling window size. Defaults to 126.
            periods_per_year: Periods per year for annualisation.

        Returns:
            pl.DataFrame: Date column(s) plus one annualised rolling Sharpe
            column per asset.

        Raises:
            ValueError: If rolling_period is not a positive integer.

        """
        actual_window = rolling_period
        actual_periods = periods_per_year or self._data._periods_per_year
        if not isinstance(actual_window, int) or actual_window <= 0:
            raise ValueError("rolling_period must be a positive integer")  # noqa: TRY003
        scale = _annualization_factor(actual_periods)
        return self.all.select(
            [pl.col(name) for name in self._data.date_col]
            + [
                (
                    pl.col(col).rolling_mean(window_size=actual_window)
                    / pl.col(col).rolling_std(window_size=actual_window)
                    * scale
                ).alias(col)
                for col, _ in self._data.items()
            ]
        )

    def rolling_greeks(
        self,
        rolling_period: int = 126,
        periods_per_year: int | float | None = None,
        benchmark: str | None = None,
    ) -> pl.DataFrame:
        """Rolling alpha and beta versus the benchmark.

        Computes rolling alpha (annualised) and beta for each asset against the
        benchmark using a trailing window.  Beta is estimated via the standard
        OLS formula: ``cov(asset, bench) / var(bench)``.  Alpha is the
        per-period intercept annualised by multiplying by *periods_per_year*.

        Args:
            rolling_period (int): Trailing window size. Defaults to 126.
            periods_per_year (int | float, optional): Periods per year used to
                annualise alpha. Defaults to the value inferred from the data.
            benchmark (str, optional): Benchmark column name. Defaults to the
                first benchmark column.

        Returns:
            pl.DataFrame: Date column(s) followed by ``{asset}_alpha`` and
                ``{asset}_beta`` columns for every asset.

        Raises:
            AttributeError: If no benchmark data is attached.
            ValueError: If *rolling_period* is not a positive integer.
        """
        if self._data.benchmark is None:
            raise AttributeError("No benchmark data available")  # noqa: TRY003
        if not isinstance(rolling_period, int) or rolling_period <= 0:
            raise ValueError("rolling_period must be a positive integer")  # noqa: TRY003

        ppy = periods_per_year or self._data._periods_per_year
        all_df = self.all
        bench_col = benchmark or self._data.benchmark.columns[0]

        w = rolling_period
        exprs: list[pl.Expr] = []
        for col, _ in self._data.items():
            mean_x = pl.col(col).rolling_mean(window_size=w)
            mean_y = pl.col(bench_col).rolling_mean(window_size=w)
            mean_xy = (pl.col(col) * pl.col(bench_col)).rolling_mean(window_size=w)
            mean_y2 = (pl.col(bench_col) ** 2).rolling_mean(window_size=w)

            bench_var = mean_y2 - mean_y**2
            bench_cov = mean_xy - mean_x * mean_y

            # beta = cov(asset, bench) / var(bench); NaN when var(bench) = 0
            beta_expr = (bench_cov / bench_var).alias(f"{col}_beta")
            # alpha (per period) = mean(asset) - beta * mean(bench), annualised
            alpha_expr = ((mean_x - (bench_cov / bench_var) * mean_y) * ppy).alias(f"{col}_alpha")

            exprs.extend([beta_expr, alpha_expr])

        return all_df.select([pl.col(name) for name in self._data.date_col] + exprs)

    def rolling_volatility(
        self,
        rolling_period: int = 126,
        periods_per_year: int | float | None = None,
        annualize: bool = True,
    ) -> pl.DataFrame:
        """Calculate the rolling volatility of returns.

        Args:
            rolling_period: Rolling window size. Defaults to 126.
            periods_per_year: Periods per year for annualisation.
            annualize: Multiply by ``sqrt(periods_per_year)`` when True (default).

        Returns:
            pl.DataFrame: Date column(s) plus one rolling volatility column
            per asset.

        Raises:
            ValueError: If rolling_period is not a positive integer.
            TypeError: If periods_per_year is not numeric.

        """
        actual_window = rolling_period
        actual_periods = periods_per_year or self._data._periods_per_year
        if not isinstance(actual_window, int) or actual_window <= 0:
            raise ValueError("rolling_period must be a positive integer")  # noqa: TRY003
        if not isinstance(actual_periods, int | float):
            raise TypeError
        factor = _annualization_factor(actual_periods) if annualize else 1.0
        return self.all.select(
            [pl.col(name) for name in self._data.date_col]
            + [
                (pl.col(col).rolling_std(window_size=actual_window) * factor).alias(col)
                for col, _ in self._data.items()
            ]
        )
