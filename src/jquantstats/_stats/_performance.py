"""Risk-adjusted return ratios for financial data.

This module owns the ratio family: Sharpe and Sortino, Omega, and the
probabilistic / smart / adjusted variants derived from them. Two concerns that
previously shared this module were split out, taking it from 714 lines to ~460:

- concentration (HHI) → :mod:`jquantstats._stats._concentration`
- benchmark-relative and factor metrics → :mod:`jquantstats._stats._benchmark`

What remains is deliberately kept together: the ``probabilistic_*``, ``smart_*``
and ``adjusted_*`` methods are all defined in terms of the base ``sharpe`` and
``sortino`` ratios above them (via ``_probabilistic_ratio_from_base``), so
splitting them further would separate derived metrics from the ones they derive
from.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from scipy.stats import norm

from ._core import _mean, _std_is_negligible, _to_float, columnwise_stat
from ._internals import _annualization_factor, _downside_deviation

if TYPE_CHECKING:
    from ..data import Data

# ── Risk statistics mixin ────────────────────────────────────────────────────


class _RiskStatsMixin:
    """Mixin providing risk-adjusted return ratios.

    Covers: Sharpe ratio, Sortino ratio, Omega, adjusted Sortino, and the
    probabilistic / smart variants of each.

    Cross-mixin dependencies:
        - _BasicStatsMixin: geometric_mean, autocorr_penalty

    Sibling mixins split out of this one, both composed into the same ``Stats``
    class so the public API is unchanged:
        - _ConcentrationStatsMixin: hhi_positive, hhi_negative
        - _BenchmarkStatsMixin: r_squared, information_ratio, greeks, treynor_ratio
    """

    _data: Data
    all: pl.DataFrame

    if TYPE_CHECKING:
        from .._protocol import DataLike

        data: DataLike

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
