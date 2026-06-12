"""Performance and risk-adjusted return metrics for financial data."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from scipy.stats import norm

from ._core import _mean, _std_is_negligible, _to_float, columnwise_stat, to_frame
from ._internals import _annualization_factor, _comp_return, _downside_deviation, _nav_series

if TYPE_CHECKING:
    from ..data import Data

# ── Risk statistics mixin ────────────────────────────────────────────────────


class _RiskStatsMixin:
    """Mixin providing performance, drawdown, and benchmark/factor metrics.

    Covers: Sharpe ratio, Sortino ratio, adjusted Sortino, drawdown series,
    max drawdown, prices, R-squared, information ratio, and Greeks (alpha/beta).

    Cross-mixin dependencies:
        - _BasicStatsMixin: geometric_mean, autocorr_penalty

    **Concentration metrics (intentionally public, optional use)**

    ``hhi_positive`` and ``hhi_negative`` implement the
    Herfindahl-Hirschman Index applied to the signed distribution of returns.
    They measure *temporal* concentration of gains and losses respectively —
    a value near 0 means returns are spread evenly across periods; a value
    near 1 means a single period dominates.  These metrics are not included
    in ``summary()`` by default because they are supplemental diagnostics
    rather than standard risk-adjusted-return measures, but they are fully
    supported as part of the public ``Stats`` API.
    """

    _data: Data
    all: pl.DataFrame

    if TYPE_CHECKING:
        from .._protocol import DataLike

        data: DataLike
        all: pl.DataFrame | None

        def autocorr_penalty(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def geometric_mean(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

    # ── Sharpe & Sortino ──────────────────────────────────────────────────────

    @columnwise_stat
    def sharpe(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calculate the Sharpe ratio of asset returns.

        Args:
            series (pl.Series): The series to calculate Sharpe ratio for.
            periods (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            float: The Sharpe ratio value.


        Returns NaN when:
            ``float("nan")`` when the standard deviation is missing (fewer than two
            observations) or numerically negligible.
        """
        periods = periods or self._data._periods_per_year

        std_val = cast(float | None, series.std(ddof=1))
        mean_val = series.mean()
        mean_f = cast(float, mean_val) if mean_val is not None else 0.0

        if _std_is_negligible(std_val, mean_f):
            return float("nan")

        res = mean_f / cast(float, std_val)
        factor = periods or 1
        return float(res * _annualization_factor(factor))

    @columnwise_stat
    def sharpe_variance(self, series: pl.Series, periods: int | float | None = None) -> float:
        r"""Calculate the asymptotic variance of the Sharpe Ratio.

        .. math::
            \text{Var}(SR) = \frac{1 + \frac{S \cdot SR}{2} + \frac{(K - 3) \cdot SR^2}{4}}{T}

        where:
            - \(S\) is the skewness of returns
            - \(K\) is the kurtosis of returns
            - \(SR\) is the Sharpe ratio (unannualized)
            - \(T\) is the number of observations

        Args:
            series (pl.Series): The series to calculate Sharpe ratio variance for.
            periods (int | float, optional): Number of periods per year. Defaults to data periods.

        Returns:
            float: The asymptotic variance of the Sharpe ratio.
            If number of periods per year is provided or inferred from the data, the result is annualized.


        Returns NaN when:
            ``float("nan")`` when the standard deviation is zero/missing or
            skewness/kurtosis cannot be computed.
        """
        t = series.count()
        mean_val = _mean(series)
        std_val = cast(float, series.std(ddof=1))
        if std_val is None or std_val == 0:
            return float("nan")  # indeterminate: zero or missing standard deviation
        sr = mean_val / std_val

        skew_val = series.skew(bias=False)
        kurt_val = series.kurtosis(bias=False)

        if skew_val is None or kurt_val is None:
            return float("nan")  # indeterminate: missing moments
        # Base variance calculation using unannualized Sharpe ratio
        # Formula: (1 + skew*SR/2 + (kurt-3)*SR²/4) / T
        base_variance = (1 + (float(skew_val) * sr) / 2 + ((float(kurt_val) - 3) / 4) * sr**2) / t
        # Annualize by scaling with the number of periods
        periods = periods or self._data._periods_per_year
        factor = periods or 1
        return float(base_variance * _annualization_factor(factor, sqrt=False))

    @columnwise_stat
    def probabilistic_sharpe_ratio(self, series: pl.Series) -> float:
        r"""Calculate the probabilistic sharpe ratio (PSR).

        Args:
            series (pl.Series): The series to calculate probabilistic Sharpe ratio for.

        Returns:
            float: Probabilistic Sharpe Ratio.

        Note:
            PSR is the probability that the observed Sharpe ratio is greater than a
            given benchmark Sharpe ratio.


        Returns NaN when:
            ``float("nan")`` when the standard deviation is zero/missing, moments
            are missing, or the estimated Sharpe variance is non-positive.
        """
        t = series.count()

        # Calculate observed unannualized Sharpe ratio
        mean_val = _mean(series)
        std_val = cast(float, series.std(ddof=1))
        if std_val is None or std_val == 0:
            return float("nan")  # indeterminate: zero or missing standard deviation
        # Unannualized observed Sharpe ratio
        observed_sr = mean_val / std_val

        skew_val = series.skew(bias=False)
        kurt_val = series.kurtosis(bias=False)

        if skew_val is None or kurt_val is None:
            return float("nan")  # indeterminate: missing moments

        benchmark_sr = 0.0
        # Calculate variance using unannualized benchmark Sharpe ratio
        var_bench_sr = (1 + (float(skew_val) * benchmark_sr) / 2 + ((float(kurt_val) - 3) / 4) * benchmark_sr**2) / t

        if var_bench_sr <= 0:
            return float("nan")  # pragma: no cover  # indeterminate: non-positive variance
        return float(norm.cdf((observed_sr - benchmark_sr) / np.sqrt(var_bench_sr)))

    # ── Concentration metrics (HHI) ───────────────────────────────────────────

    @columnwise_stat
    def hhi_positive(self, series: pl.Series) -> float:
        r"""Calculate the Herfindahl-Hirschman Index (HHI) for positive returns.

        This quantifies how concentrated the positive returns are in a series.

        .. math::
            w^{\plus} = \frac{r_{t}^{\plus}}{\sum{r_{t}^{\plus}}} \\
            HHI^{\plus} = \frac{N_{\plus} \sum{(w^{\plus})^2} - 1}{N_{\plus} - 1}

        where:
            - \(r_{t}^{\plus}\) are the positive returns
            - \(N_{\plus}\) is the number of positive returns
            - \(w^{\plus}\) are the weights of positive returns

        Args:
            series (pl.Series): The series to calculate HHI for.

        Returns:
            float: The HHI value for positive returns. Returns NaN if fewer than 3
                positive returns are present.

        Note:
            Values range from 0 (perfectly diversified gains) to 1 (all gains
            concentrated in a single period).
        """
        positive_returns = series.filter(series > 0).drop_nans()
        if positive_returns.len() <= 2:
            return float("nan")  # indeterminate: fewer than 3 positive returns
        weight = positive_returns / positive_returns.sum()
        return float((weight.len() * (weight**2).sum() - 1) / (weight.len() - 1))

    @columnwise_stat
    def hhi_negative(self, series: pl.Series) -> float:
        r"""Calculate the Herfindahl-Hirschman Index (HHI) for negative returns.

        This quantifies how concentrated the negative returns are in a series.

        .. math::
            w^{\minus} = \frac{r_{t}^{\minus}}{\sum{r_{t}^{\minus}}} \\
            HHI^{\minus} = \frac{N_{\minus} \sum{(w^{\minus})^2} - 1}{N_{\minus} - 1}

        where:
            - \(r_{t}^{\minus}\) are the negative returns
            - \(N_{\minus}\) is the number of negative returns
            - \(w^{\minus}\) are the weights of negative returns

        Args:
            series (pl.Series): The returns series to calculate HHI for.

        Returns:
            float: The HHI value for negative returns. Returns NaN if fewer than 3
                negative returns are present.

        Note:
            Values range from 0 (perfectly diversified losses) to 1 (all losses
            concentrated in a single period).
        """
        negative_returns = series.filter(series < 0).drop_nans()
        if negative_returns.len() <= 2:
            return float("nan")  # indeterminate: fewer than 3 negative returns
        weight = negative_returns / negative_returns.sum()
        return float((weight.len() * (weight**2).sum() - 1) / (weight.len() - 1))

    @columnwise_stat
    def sortino(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calculate the Sortino ratio.

        The Sortino ratio is the mean return divided by downside deviation.
        Based on Red Rock Capital's Sortino ratio paper.

        Args:
            series (pl.Series): The series to calculate Sortino ratio for.
            periods (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            float: The Sortino ratio value.


        Returns NaN when:
            ``float("nan")`` when both the mean return and the downside deviation
            are zero.
        """
        periods = periods or self._data._periods_per_year
        downside_deviation = _downside_deviation(series)
        mean_f = _mean(series)
        if downside_deviation == 0.0:
            if mean_f > 0:
                return float("inf")
            elif mean_f < 0:  # pragma: no cover  # unreachable: no negatives ⟹ mean ≥ 0
                return float("-inf")
            else:
                return float("nan")  # indeterminate: zero mean and zero downside deviation
        ratio = mean_f / downside_deviation
        return float(ratio * _annualization_factor(periods))

    @columnwise_stat
    def omega(
        self,
        series: pl.Series,
        rf: float = 0.0,
        required_return: float = 0.0,
        periods: int | float | None = None,
    ) -> float:
        """Calculate the Omega ratio.

        The Omega ratio is the probability-weighted ratio of gains to losses
        relative to a threshold return.  It is computed as the sum of returns
        above the threshold divided by the absolute sum of returns below it.

        Args:
            series (pl.Series): The series to calculate Omega ratio for.
            rf (float): Annualised risk-free rate. Defaults to 0.0.
            required_return (float): Annualised minimum acceptable return
                threshold. Defaults to 0.0.
            periods (int | float | None): Number of periods per year. Defaults
                to the value inferred from the data.

        Returns:
            float: The Omega ratio, or NaN when the denominator is zero or
                when ``required_return <= -1``.

        Note:
            See https://en.wikipedia.org/wiki/Omega_ratio for details.

        """
        if required_return <= -1:
            return float("nan")

        periods = periods or self._data._periods_per_year

        # Subtract per-period risk-free rate from returns when rf is non-zero.
        if rf != 0.0:
            rf_per_period = float((1.0 + rf) ** (1.0 / periods) - 1.0)
            series = series - rf_per_period

        # Convert annualised required return to a per-period threshold.
        return_threshold = float((1.0 + required_return) ** (1.0 / periods) - 1.0)

        returns_less_thresh = series - return_threshold

        numer = float(returns_less_thresh.filter(returns_less_thresh > 0.0).sum())
        denom = float(-returns_less_thresh.filter(returns_less_thresh < 0.0).sum())

        if denom <= 0.0:
            return float("nan")
        return numer / denom

    # ── Cumulative returns ────────────────────────────────────────────────────

    @to_frame
    def compsum(self, series: pl.Series) -> pl.Series:
        """Calculate the rolling compounded (cumulative) returns.

        Computed as cumprod(1 + r) - 1 for each period.

        Args:
            series (pl.Series): The series to calculate cumulative returns for.

        Returns:
            pl.Series: Cumulative compounded returns per period.

        """
        return (1.0 + series).cum_prod() - 1.0

    # ── Drawdown ──────────────────────────────────────────────────────────────

    @to_frame
    def drawdown(self, series: pl.Series) -> pl.Series:
        """Calculate the drawdown series for returns.

        Args:
            series (pl.Series): The series to calculate drawdown for.

        Returns:
            pl.Series: The drawdown series.

        """
        equity = self.prices(series)
        d = (equity / equity.cum_max()) - 1
        return -d

    @staticmethod
    def prices(series: pl.Series) -> pl.Series:
        """Convert returns series to price series.

        Args:
            series (pl.Series): The returns series to convert.

        Returns:
            pl.Series: The price series.

        """
        return _nav_series(series)

    @staticmethod
    def max_drawdown_single_series(series: pl.Series) -> float:
        """Compute the maximum drawdown for a single returns series.

        Args:
            series: A Polars Series of returns values.

        Returns:
            float: The maximum drawdown as a positive fraction (e.g. 0.2 for 20%).
        """
        price = _RiskStatsMixin.prices(series)
        peak = price.cum_max()
        drawdown = price / peak - 1
        dd_min = cast(float, drawdown.min())
        return dd_min if dd_min is not None else 0.0

    @columnwise_stat
    def max_drawdown(self, series: pl.Series) -> float:
        """Calculate the maximum drawdown for each column.

        Args:
            series (pl.Series): The series to calculate maximum drawdown for.

        Returns:
            float: The maximum drawdown value.

        """
        return _RiskStatsMixin.max_drawdown_single_series(series)

    def drawdown_details(self) -> dict[str, pl.DataFrame]:
        """Return detailed statistics for each individual drawdown period.

        For each contiguous underwater episode, records the start date, valley
        (worst point), recovery date, total duration, maximum drawdown, and
        recovery duration.

        Returns:
            dict[str, pl.DataFrame]: Per-asset DataFrames with columns
                ``start``, ``valley``, ``end``, ``duration``, ``max_drawdown``,
                ``recovery_duration``.

        Note:
            ``end`` and ``recovery_duration`` are ``null`` for drawdown periods
            that have not yet recovered by the last observation.
            ``max_drawdown`` is a negative fraction (e.g. ``-0.2`` for 20%).
        """
        all_df = cast(pl.DataFrame, self.all)
        date_col_name = self._data.date_col[0] if self._data.date_col else None
        has_date = date_col_name is not None and all_df[date_col_name].dtype.is_temporal()

        result: dict[str, pl.DataFrame] = {}
        for col, series in self._data.items():
            nav = _nav_series(series)
            hwm = nav.cum_max()
            in_dd = nav < hwm
            dd_pct = nav / hwm - 1  # negative or zero

            if has_date and date_col_name is not None:
                dates = all_df[date_col_name]
            else:
                dates = pl.Series(list(range(len(series))), dtype=pl.Int64)

            date_dtype = dates.dtype

            frame = (
                pl.DataFrame({"date": dates, "nav": nav, "dd_pct": dd_pct, "in_dd": in_dd})
                .with_row_index("row_idx")
                .with_columns(pl.col("in_dd").rle_id().cast(pl.Int64).alias("run_id"))
            )

            dd_frame = frame.filter(pl.col("in_dd"))

            # A monotonic NAV has no underwater rows, so drawdown_details should return an empty typed frame.
            if dd_frame.is_empty():
                result[col] = pl.DataFrame(
                    {
                        "start": pl.Series([], dtype=date_dtype),
                        "valley": pl.Series([], dtype=date_dtype),
                        "end": pl.Series([], dtype=date_dtype),
                        "duration": pl.Series([], dtype=pl.Int64),
                        "max_drawdown": pl.Series([], dtype=pl.Float64),
                        "recovery_duration": pl.Series([], dtype=pl.Int64),
                    }
                )
                continue

            # Per-period stats: start, last_dd_date, valley, max drawdown
            dd_periods = (
                dd_frame.group_by("run_id")
                .agg(
                    [
                        pl.col("date").first().alias("start"),
                        pl.col("date").last().alias("last_dd_date"),
                        pl.col("date").sort_by("nav").first().alias("valley"),
                        pl.col("dd_pct").min().alias("max_drawdown"),
                    ]
                )
                .sort("start")
            )

            # First date of each non-drawdown run → recovery date for the preceding drawdown run
            non_dd_starts = (
                frame.filter(~pl.col("in_dd"))
                .group_by("run_id")
                .agg(pl.col("date").first().alias("end"))
                .with_columns((pl.col("run_id") - 1).alias("run_id"))
            )

            dd_periods = dd_periods.join(non_dd_starts.select(["run_id", "end"]), on="run_id", how="left")

            # Compute durations
            if has_date:
                dd_periods = dd_periods.with_columns(
                    [
                        pl.when(pl.col("end").is_not_null())
                        .then((pl.col("end") - pl.col("start")).dt.total_days())
                        .otherwise((pl.col("last_dd_date") - pl.col("start")).dt.total_days() + 1)
                        .cast(pl.Int64)
                        .alias("duration"),
                        pl.when(pl.col("end").is_not_null())
                        .then((pl.col("end") - pl.col("valley")).dt.total_days().cast(pl.Int64))
                        .otherwise(pl.lit(None, dtype=pl.Int64))
                        .alias("recovery_duration"),
                    ]
                )
            else:
                dd_periods = dd_periods.with_columns(
                    [
                        pl.when(pl.col("end").is_not_null())
                        .then((pl.col("end") - pl.col("start")).cast(pl.Int64))
                        .otherwise((pl.col("last_dd_date") - pl.col("start") + 1).cast(pl.Int64))
                        .alias("duration"),
                        pl.when(pl.col("end").is_not_null())
                        .then((pl.col("end") - pl.col("valley")).cast(pl.Int64))
                        .otherwise(pl.lit(None, dtype=pl.Int64))
                        .alias("recovery_duration"),
                    ]
                )

            result[col] = dd_periods.select(["start", "valley", "end", "duration", "max_drawdown", "recovery_duration"])

        return result

    @staticmethod
    def _probabilistic_ratio_from_base(base: float, series: pl.Series) -> float:
        """Compute the probabilistic ratio given an observed unannualized base ratio.

        Uses the formula: norm.cdf(base / sigma), where
        sigma = sqrt((1 + 0.5·base² - skew·base + (kurt-3)/4·base²) / (n-1)).

        Args:
            base (float): Unannualized observed ratio (e.g. Sortino).
            series (pl.Series): The original returns series (for moments and n).

        Returns:
            float: Probabilistic ratio in [0, 1].


        Returns NaN when:
            ``float("nan")`` when moments are missing, there are fewer than two
            observations, or the estimated variance is non-positive.
        """
        n = series.count()
        skew_val = series.skew(bias=False)
        kurt_val = series.kurtosis(bias=False)
        if skew_val is None or kurt_val is None or n <= 1:
            return float("nan")  # indeterminate: missing moments or insufficient data
        variance = (1 + 0.5 * base**2 - float(skew_val) * base + ((float(kurt_val) - 3) / 4) * base**2) / (n - 1)
        if variance <= 0:
            return float("nan")  # indeterminate: non-positive variance
        return float(norm.cdf(base / np.sqrt(variance)))

    @columnwise_stat
    def probabilistic_sortino_ratio(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calculate the Probabilistic Sortino Ratio.

        The probability that the observed Sortino ratio is greater than zero,
        accounting for estimation uncertainty via skewness and kurtosis.

        Args:
            series (pl.Series): The series to calculate the ratio for.
            periods (int | float, optional): Accepted for API compatibility; has no effect
                since the base ratio is un-annualized.

        Returns:
            float: Probabilistic Sortino ratio in [0, 1].


        Returns NaN when:
            ``float("nan")`` when the downside deviation is zero, moments are
            missing, or the estimated variance is non-positive.
        """
        downside_deviation = _downside_deviation(series)
        mean_f = _mean(series)
        if downside_deviation == 0.0:
            return float("nan")  # indeterminate: zero downside deviation
        base = float(mean_f / downside_deviation)
        return self._probabilistic_ratio_from_base(base, series)

    @columnwise_stat
    def probabilistic_adjusted_sortino_ratio(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calculate the Probabilistic Adjusted Sortino Ratio.

        The probability that the observed adjusted Sortino ratio (divided by sqrt(2)
        for Sharpe comparability) is greater than zero, accounting for estimation
        uncertainty via skewness and kurtosis.

        Args:
            series (pl.Series): The series to calculate the ratio for.
            periods (int | float, optional): Accepted for API compatibility; has no effect
                since the base ratio is un-annualized.

        Returns:
            float: Probabilistic adjusted Sortino ratio in [0, 1].


        Returns NaN when:
            ``float("nan")`` when the downside deviation is zero, moments are
            missing, or the estimated variance is non-positive.
        """
        downside_deviation = _downside_deviation(series)
        mean_f = _mean(series)
        if downside_deviation == 0.0:
            return float("nan")  # indeterminate: zero downside deviation
        base = float(mean_f / downside_deviation) / np.sqrt(2)
        return self._probabilistic_ratio_from_base(base, series)

    def probabilistic_ratio(
        self,
        base: str | Callable[[pl.Series], float] = "sharpe",
    ) -> dict[str, float]:
        r"""Generic probabilistic ratio for any base metric.

        Computes the probability that the observed ratio is greater than zero,
        accounting for estimation uncertainty via skewness and kurtosis using
        the Lopez de Prado (2018) framework.

        Args:
            base: Base ratio to use. Either:

                - A string: ``'sharpe'``, ``'sortino'``, ``'adjusted_sortino'``.
                - A callable ``(series: pl.Series) -> float`` returning the
                  **unannualized** ratio for a single series.

        Returns:
            dict[str, float]: Probabilistic ratio in ``[0, 1]`` per asset.

        Raises:
            ValueError: If *base* is an unrecognised string.


        Returns NaN when:
            Entries are ``float("nan")`` when the base ratio is undefined (zero
            standard deviation / zero downside deviation), moments are missing, or
            the estimated variance is non-positive.
        """

        def _sharpe_base(s: pl.Series) -> float:
            """Return the per-period Sharpe ratio (mean / std, ddof=1) of *s*."""
            mean_val = _mean(s)
            std_val = cast(float, s.std(ddof=1))
            if not std_val or std_val == 0:
                return float("nan")
            return mean_val / std_val

        def _sortino_base(s: pl.Series) -> float:
            """Return the per-period Sortino ratio (mean / downside_dev) of *s*."""
            downside_sum = _to_float((s.filter(s < 0) ** 2).sum())
            downside_dev = float(np.sqrt(downside_sum / s.count()))
            if downside_dev == 0.0:
                return float("nan")
            return _mean(s) / downside_dev

        _builtin: dict[str, Callable[[pl.Series], float]] = {
            "sharpe": _sharpe_base,
            "sortino": _sortino_base,
            "adjusted_sortino": lambda s: _sortino_base(s) / float(np.sqrt(2)),
        }

        if isinstance(base, str):
            if base not in _builtin:
                raise ValueError(f"base must be one of {list(_builtin)}, got {base!r}")  # noqa: TRY003
            base_fn = _builtin[base]
        else:
            base_fn = base

        result: dict[str, float] = {}
        for col, series in self._data.items():
            base_val = base_fn(series)
            if np.isnan(base_val):
                result[col] = float("nan")
            else:
                result[col] = _RiskStatsMixin._probabilistic_ratio_from_base(base_val, series)
        return result

    def smart_sharpe(self, periods: int | float | None = None) -> dict[str, float]:
        """Calculate the Smart Sharpe ratio (Sharpe with autocorrelation penalty).

        Divides the Sharpe ratio by the autocorrelation penalty to account for
        return autocorrelation that can artificially inflate risk-adjusted metrics.

        Args:
            periods (int | float, optional): Number of periods per year. Defaults to periods_per_year.

        Returns:
            dict[str, float]: Dictionary mapping asset names to Smart Sharpe ratios.

        """
        sharpe_data = self.sharpe(periods=periods)
        penalty_data = self.autocorr_penalty()
        return {k: sharpe_data[k] / penalty_data[k] for k in sharpe_data}

    def smart_sortino(self, periods: int | float | None = None) -> dict[str, float]:
        """Calculate the Smart Sortino ratio (Sortino with autocorrelation penalty).

        Divides the Sortino ratio by the autocorrelation penalty to account for
        return autocorrelation that can artificially inflate risk-adjusted metrics.

        Args:
            periods (int | float, optional): Number of periods per year. Defaults to periods_per_year.

        Returns:
            dict[str, float]: Dictionary mapping asset names to Smart Sortino ratios.

        """
        sortino_data = self.sortino(periods=periods)
        penalty_data = self.autocorr_penalty()
        return {k: sortino_data[k] / penalty_data[k] for k in sortino_data}

    def adjusted_sortino(self, periods: int | float | None = None) -> dict[str, float]:
        """Calculate Jack Schwager's adjusted Sortino ratio.

        This adjustment allows for direct comparison to Sharpe ratio.
        See: https://archive.is/wip/2rwFW.

        Args:
            periods (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            dict[str, float]: Dictionary mapping asset names to adjusted Sortino ratios.

        """
        sortino_data = self.sortino(periods=periods)
        return {k: v / np.sqrt(2) for k, v in sortino_data.items()}

    # ── Benchmark & factor ────────────────────────────────────────────────────

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
            raise AttributeError("No benchmark data available")  # noqa: TRY003

        benchmark_col = benchmark or self._data.benchmark.columns[0]

        # Evaluate both series and benchmark as Series
        all_data = cast(pl.DataFrame, self.all)
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
            raise AttributeError("No benchmark data available")  # noqa: TRY003

        ppy = periods_per_year or self._data._periods_per_year

        benchmark_col = benchmark or self._data.benchmark.columns[0]
        all_series = cast(pl.DataFrame, self.all)
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
        all_data = cast(pl.DataFrame, self.all)
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
            raise AttributeError("No benchmark data available")  # noqa: TRY003

        ppy = periods or self._data._periods_per_year

        benchmark_data = self._data.benchmark
        benchmark_col = benchmark or benchmark_data.columns[0]

        all_data = cast(pl.DataFrame, self.all)
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
