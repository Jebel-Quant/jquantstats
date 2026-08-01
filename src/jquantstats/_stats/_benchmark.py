"""Benchmark-relative and factor metrics.

Split out of :mod:`jquantstats._stats._performance`, which had grown to 714 lines
across four unrelated concerns. This module owns the metrics that are only
meaningful *against a benchmark* — R-squared, information ratio, the CAPM Greeks
(alpha/beta) and the Treynor ratio — each of which raises
:class:`~jquantstats.exceptions.NoBenchmarkError` when none is configured.

The risk-adjusted-ratio family stays in ``_performance``; concentration metrics
live in ``_concentration``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl

from ..exceptions import NoBenchmarkError
from ._core import _mean, columnwise_stat
from ._internals import _comp_return

if TYPE_CHECKING:
    from ..data import Data

# ── Benchmark & factor mixin ─────────────────────────────────────────────────


class _BenchmarkStatsMixin:
    """Mixin providing benchmark-relative and factor analytics.

    Covers R-squared, information ratio, the CAPM Greeks (alpha/beta) and the
    Treynor ratio. Every metric here requires ``self._data.benchmark`` and raises
    :class:`~jquantstats.exceptions.NoBenchmarkError` when it is absent — that
    shared precondition is what makes these a coherent unit.
    """

    _data: Data
    all: pl.DataFrame

    @columnwise_stat
    def r_squared(self, series: pl.Series, benchmark: str | None = None) -> float:
        """Measure the straight line fit of the equity curve.

        Args:
            series (pl.Series): The series to calculate R-squared for.
            benchmark (str, optional): The benchmark column name. Defaults to None.

        Returns:
            float: The R-squared value.

        Raises:
            AttributeError: If no benchmark data is available.

        """
        if self._data.benchmark is None:
            raise NoBenchmarkError

        benchmark_col = benchmark or self._data.benchmark.columns[0]

        # Evaluate both series and benchmark as Series
        all_data = self.all
        dframe = all_data.select([series, pl.col(benchmark_col).alias("benchmark")]).drop_nulls()

        matrix = dframe.to_numpy()
        # Get actual Series

        strategy_np = matrix[:, 0]
        benchmark_np = matrix[:, 1]

        corr_matrix = np.corrcoef(strategy_np, benchmark_np)
        r = corr_matrix[0, 1]
        return float(r**2)

    @columnwise_stat
    def information_ratio(
        self,
        series: pl.Series,
        periods_per_year: int | float | None = None,
        benchmark: str | None = None,
        annualise: bool = False,
    ) -> float:
        """Calculate the information ratio.

        This is essentially the risk return ratio of the net profits.

        Args:
            series (pl.Series): The series to calculate information ratio for.
            periods_per_year (int, optional): Number of periods per year. Defaults to 252.
            benchmark (str, optional): The benchmark column name. Defaults to None.
            annualise (bool, optional): Whether to annualise the ratio by multiplying by
                ``sqrt(periods_per_year)``. Defaults to ``True``.  Set to ``False`` to
                obtain the raw (non-annualised) information ratio, which matches the value
                returned by ``qs.stats.information_ratio``.

        Returns:
            float: The information ratio value.

        """
        if self._data.benchmark is None:
            raise NoBenchmarkError

        ppy = periods_per_year or self._data._periods_per_year

        benchmark_col = benchmark or self._data.benchmark.columns[0]
        all_series = self.all
        valid_pairs = pl.DataFrame({"strategy": series, "benchmark": all_series[benchmark_col]}).drop_nulls()
        active = valid_pairs["strategy"] - valid_pairs["benchmark"]

        mean_f = _mean(active)
        std_val = cast(float, active.std())

        try:
            std_f = std_val if std_val is not None else 1.0
            ir = mean_f / std_f
            return float(ir * (ppy**0.5) if annualise else ir)
        except ZeroDivisionError:
            return 0.0

    @columnwise_stat
    def greeks(
        self, series: pl.Series, periods_per_year: int | float | None = None, benchmark: str | None = None
    ) -> dict[str, float]:
        """Calculate alpha and beta of the portfolio.

        Args:
            series (pl.Series): The series to calculate greeks for.
            periods_per_year (int, optional): Number of periods per year. Defaults to 252.
            benchmark (str, optional): The benchmark column name. Defaults to None.

        Returns:
            dict[str, float]: Dictionary containing alpha and beta values.


        Returns NaN when:
            Both alpha and beta are ``float("nan")`` when the benchmark variance
            is zero.
        """
        ppy = periods_per_year or self._data._periods_per_year

        benchmark_data = cast(pl.DataFrame, self._data.benchmark)
        benchmark_col = benchmark or benchmark_data.columns[0]

        # Evaluate both series and benchmark as Series
        all_data = self.all
        dframe = all_data.select([series, pl.col(benchmark_col).alias("benchmark")]).drop_nulls()
        matrix = dframe.to_numpy()

        # Get actual Series
        strategy_np = matrix[:, 0]
        benchmark_np = matrix[:, 1]

        # 2x2 covariance matrix: [[var_strategy, cov], [cov, var_benchmark]]
        cov_matrix = np.cov(strategy_np, benchmark_np)

        cov = cov_matrix[0, 1]
        var_benchmark = cov_matrix[1, 1]

        beta = float(cov / var_benchmark) if var_benchmark != 0 else float("nan")
        alpha = float(np.mean(strategy_np) - beta * np.mean(benchmark_np))

        return {"alpha": float(alpha * ppy), "beta": beta}

    @columnwise_stat
    def treynor_ratio(
        self,
        series: pl.Series,
        periods: int | float | None = None,
        benchmark: str | None = None,
    ) -> float:
        """Treynor ratio: annualised excess return divided by beta.

        Measures return per unit of systematic (market) risk. Unlike the Sharpe
        ratio, which divides by total volatility, the Treynor ratio divides by
        beta — making it most meaningful for well-diversified portfolios.

        Args:
            series (pl.Series): The returns series for one asset.
            periods (int | float, optional): Periods per year for CAGR
                annualisation. Defaults to the value inferred from the data.
            benchmark (str, optional): Benchmark column name. Defaults to the
                first benchmark column.

        Returns:
            float: Treynor ratio, or ``nan`` when beta is zero or the benchmark
                is unavailable.

        Raises:
            AttributeError: If no benchmark data is attached.

        Returns NaN when:
            ``float("nan")`` when the benchmark variance or beta is zero, the
            series is empty, or the compounded NAV is non-positive.
        """
        if self._data.benchmark is None:
            raise NoBenchmarkError

        ppy = periods or self._data._periods_per_year

        benchmark_data = self._data.benchmark
        benchmark_col = benchmark or benchmark_data.columns[0]

        all_data = self.all
        dframe = all_data.select([series, pl.col(benchmark_col).alias("_bench")]).drop_nulls()
        matrix = dframe.to_numpy()
        strategy_np = matrix[:, 0]
        benchmark_np = matrix[:, 1]

        cov_matrix = np.cov(strategy_np, benchmark_np)
        var_benchmark = cov_matrix[1, 1]
        if var_benchmark == 0:
            return float("nan")
        beta = float(cov_matrix[0, 1] / var_benchmark)
        if beta == 0:
            return float("nan")

        n = len(series)
        if n == 0:
            return float("nan")  # pragma: no cover
        nav_final = 1.0 + _comp_return(series)
        if nav_final <= 0:
            return float("nan")
        cagr = float(nav_final ** (ppy / n) - 1.0)
        return cagr / beta
