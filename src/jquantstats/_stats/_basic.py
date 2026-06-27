"""Composite ratios, streak, outlier and autocorrelation statistics.

`_BasicStatsMixin` extends `_BasicCoreMixin` with the higher-level
metrics that combine the core statistics (e.g. CPC index, common-sense
ratio, Kelly criterion) together with streak, outlier and
autocorrelation analytics. It remains the public mixin consumed by
`Stats`.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import polars as pl

from ._basic_core import _BasicCoreMixin
from ._core import _mean, columnwise_stat


class _BasicStatsMixin(_BasicCoreMixin):
    """Mixin providing basic return/risk and win/loss financial statistics.

    Extends `_BasicCoreMixin` with composite ratios (CPC, common-sense,
    Kelly), streak and outlier analytics, and autocorrelation metrics.
    It is the public mixin consumed by `Stats`.
    """

    @columnwise_stat
    def autocorr_penalty(self, series: pl.Series) -> float:
        """Calculate the autocorrelation penalty for risk-adjusted metrics.

        Computes a penalty factor that accounts for autocorrelation in returns,
        which can inflate Sharpe and Sortino ratios.

        Args:
            series (pl.Series): The series to calculate autocorrelation penalty for.

        Returns:
            float: Autocorrelation penalty factor (>= 1).

        """
        arr = series.drop_nulls().to_numpy()
        num = len(arr)
        coef = float(np.abs(np.corrcoef(arr[:-1], arr[1:])[0, 1]))
        x = np.arange(1, num)
        corr = ((num - x) / num) * (coef**x)
        return float(np.sqrt(1 + 2 * np.sum(corr)))

    @staticmethod
    def _max_consecutive(mask: pl.Series) -> int:
        """Return the longest run of True values in a boolean mask.

        Args:
            mask (pl.Series): Boolean series (True = qualifying period).

        Returns:
            int: Length of the longest consecutive True run.

        """
        group_ids = mask.rle_id()
        df = pl.DataFrame({"v": mask.cast(pl.Int32), "g": group_ids})
        result = (
            df.with_columns((pl.int_range(pl.len()).over("g") + 1).alias("rank"))
            .select((pl.col("v") * pl.col("rank")).max())
            .item()
        )
        return int(result) if result is not None else 0

    @columnwise_stat
    def consecutive_wins(self, series: pl.Series) -> int:
        """Calculate the maximum number of consecutive winning periods.

        Args:
            series (pl.Series): The series to calculate consecutive wins for.

        Returns:
            int: Maximum number of consecutive winning periods.

        """
        return self._max_consecutive(series > 0)

    @columnwise_stat
    def consecutive_losses(self, series: pl.Series) -> int:
        """Calculate the maximum number of consecutive losing periods.

        Args:
            series (pl.Series): The series to calculate consecutive losses for.

        Returns:
            int: Maximum number of consecutive losing periods.

        """
        return self._max_consecutive(series < 0)

    @columnwise_stat
    def risk_of_ruin(self, series: pl.Series) -> float:
        """Calculate the risk of ruin (probability of losing all capital).

        Uses the formula: ((1 - win_rate) / (1 + win_rate)) ^ n,
        where n is the number of periods.

        Args:
            series (pl.Series): The series to calculate risk of ruin for.

        Returns:
            float: The risk of ruin probability.

        """
        num_pos = self._positive(series).count()
        num_nonzero = series.filter(series != 0).count()
        wins = float(num_pos / num_nonzero)
        n = series.len()
        return ((1 - wins) / (1 + wins)) ** n

    @columnwise_stat
    def tail_ratio(self, series: pl.Series, cutoff: float = 0.95) -> float:
        """Calculate the tail ratio (right tail / left tail).

        Measures the ratio between the upper and lower tails of the return
        distribution: abs(quantile(cutoff) / quantile(1 - cutoff)).

        Args:
            series (pl.Series): The series to calculate tail ratio for.
            cutoff (float): Percentile cutoff for tail analysis. Defaults to 0.95.

        Returns:
            float: Tail ratio.


        Returns NaN when:
            ``float("nan")`` when either quantile is missing or the lower quantile
            is zero.
        """
        upper = cast(float, series.quantile(cutoff, interpolation="linear"))
        lower = cast(float, series.quantile(1 - cutoff, interpolation="linear"))
        if upper is None or lower is None or lower == 0:
            return float("nan")  # indeterminate: zero or missing quantile
        return float(np.abs(upper / lower))

    def cpc_index(self) -> dict[str, float]:
        """Calculate the CPC Index (Profit Factor * Win Rate * Win-Loss Ratio).

        Returns:
            dict[str, float]: Dictionary mapping asset names to CPC Index values.

        """
        pf = self.profit_factor()
        wr = self.win_rate()
        wlr = self.payoff_ratio()
        return {col: pf[col] * wr[col] * wlr[col] for col in pf}

    def common_sense_ratio(self) -> dict[str, float]:
        """Calculate the Common Sense Ratio (Profit Factor * Tail Ratio).

        Returns:
            dict[str, float]: Dictionary mapping asset names to Common Sense Ratio values.

        """
        pf = self.profit_factor()
        tr = self.tail_ratio()
        return {col: pf[col] * tr[col] for col in pf}

    def outliers(self, quantile: float = 0.95) -> dict[str, pl.Series]:
        """Return only the returns above a quantile threshold.

        Args:
            quantile (float): Upper quantile threshold. Defaults to 0.95.

        Returns:
            dict[str, pl.Series]: Filtered series per asset containing only
                returns above the quantile.

        """
        result = {}
        for col, series in self._data.items():
            threshold = cast(float, series.quantile(quantile, interpolation="linear"))
            result[col] = series.filter(series > threshold).drop_nulls()
        return result

    def remove_outliers(self, quantile: float = 0.95) -> dict[str, pl.Series]:
        """Return returns with values above a quantile threshold removed.

        Args:
            quantile (float): Upper quantile threshold. Defaults to 0.95.

        Returns:
            dict[str, pl.Series]: Filtered series per asset containing only
                returns below the quantile.

        """
        result = {}
        for col, series in self._data.items():
            threshold = cast(float, series.quantile(quantile, interpolation="linear"))
            result[col] = series.filter(series < threshold)
        return result

    @columnwise_stat
    def outlier_win_ratio(self, series: pl.Series, quantile: float = 0.99) -> float:
        """Calculate the outlier winners ratio.

        Ratio of the high-quantile return to the mean positive return,
        showing how much outlier wins contribute to overall performance.

        Args:
            series (pl.Series): The series to calculate outlier win ratio for.
            quantile (float): Quantile for the outlier threshold. Defaults to 0.99.

        Returns:
            float: Outlier win ratio.


        Returns NaN when:
            ``float("nan")`` when the mean of non-negative returns is zero.
        """
        positive_mean = _mean(series.filter(series >= 0))
        if positive_mean == 0:
            return float("nan")  # indeterminate: zero mean of positive returns
        quantile_val = cast(float, series.quantile(quantile, interpolation="linear"))
        return float(quantile_val / positive_mean)

    @columnwise_stat
    def outlier_loss_ratio(self, series: pl.Series, quantile: float = 0.01) -> float:
        """Calculate the outlier losers ratio.

        Ratio of the low-quantile return to the mean negative return,
        showing how much outlier losses contribute to overall risk.

        Args:
            series (pl.Series): The series to calculate outlier loss ratio for.
            quantile (float): Quantile for the outlier threshold. Defaults to 0.01.

        Returns:
            float: Outlier loss ratio.


        Returns NaN when:
            ``float("nan")`` when the mean of negative returns is zero.
        """
        negative_mean = self._mean_negative_expr(series)
        if negative_mean == 0:  # pragma: no cover
            return float("nan")  # indeterminate: zero mean of negative returns
        quantile_val = cast(float, series.quantile(quantile, interpolation="linear"))
        return float(quantile_val / negative_mean)

    @columnwise_stat
    def gain_to_pain_ratio(self, series: pl.Series) -> float:
        """Calculate Jack Schwager's Gain-to-Pain Ratio.

        The ratio is calculated as total return / sum of losses (in absolute value).

        Args:
            series (pl.Series): The series to calculate gain to pain ratio for.

        Returns:
            float: The gain to pain ratio value.


        Returns NaN when:
            ``float("nan")`` when there are no losses (the denominator is zero).
        """
        total_gain = series.sum()
        total_pain = self._negative(series).abs().sum()
        try:
            return float(float(total_gain) / float(total_pain))
        except ZeroDivisionError:
            return float("nan")  # indeterminate: no losses (denominator is zero)

    @columnwise_stat
    def risk_return_ratio(self, series: pl.Series) -> float:
        """Calculate the return/risk ratio.

        This is equivalent to the Sharpe ratio without a risk-free rate.

        Args:
            series (pl.Series): The series to calculate risk return ratio for.

        Returns:
            float: The risk return ratio value.

        """
        mean_val = _mean(series)
        std_val = cast(float, series.std())
        return mean_val / (std_val if std_val is not None else 1.0)

    def kelly_criterion(self) -> dict[str, float]:
        """Calculate the optimal capital allocation per column.

        Uses the Kelly Criterion formula: f* = [(b * p) - q] / b
        where:
          - b = payoff ratio
          - p = win rate
          - q = 1 - p.

        Returns:
            dict[str, float]: Dictionary mapping asset names to Kelly criterion values.

        """
        b = self.payoff_ratio()
        p = self.win_rate()

        return {col: ((b[col] * p[col]) - (1 - p[col])) / b[col] for col in b}

    @columnwise_stat
    def best(self, series: pl.Series) -> float | None:
        """Find the maximum return per column (best period).

        Args:
            series (pl.Series): The series to find the best return for.

        Returns:
            float: The maximum return value.

        """
        val = cast(float, series.max())
        return val if val is not None else None

    @columnwise_stat
    def worst(self, series: pl.Series) -> float | None:
        """Find the minimum return per column (worst period).

        Args:
            series (pl.Series): The series to find the worst return for.

        Returns:
            float: The minimum return value.

        """
        val = cast(float, series.min())
        return val if val is not None else None

    @columnwise_stat
    def exposure(self, series: pl.Series) -> float:
        """Calculate the market exposure time (returns != 0).

        Args:
            series (pl.Series): The series to calculate exposure for.

        Returns:
            float: The exposure value.

        """
        all_data = self.all
        ex = series.filter(series != 0).count() / all_data.height
        return math.ceil(ex * 100) / 100

    @staticmethod
    def _pearson_corr_shifted(series: pl.Series, lag: int) -> float:
        """Compute Pearson correlation between *series* and its lag-*lag* shift.

        Args:
            series (pl.Series): The input series.
            lag (int): Number of positions to shift.

        Returns:
            float: Pearson correlation coefficient, or NaN if no valid pairs remain.

        """
        shifted = series.shift(lag)
        paired = pl.DataFrame({"x": series, "y": shifted}).drop_nulls()
        # Large lags or null-only overlap can leave no aligned observations to correlate.
        if paired.is_empty():
            return float("nan")
        return float(np.corrcoef(paired["x"].to_numpy(), paired["y"].to_numpy())[0, 1])

    @columnwise_stat
    def autocorr(self, series: pl.Series, lag: int = 1) -> float:
        """Compute lag-n autocorrelation of returns.

        Args:
            series (pl.Series): The series to calculate autocorrelation for.
            lag (int): Number of periods to lag. Must be a positive integer.

        Returns:
            float: Pearson correlation between returns and their lagged values.

        Raises:
            TypeError: If *lag* is not an ``int``.
            ValueError: If *lag* is not a positive integer (>= 1).

        """
        if not isinstance(lag, int):
            msg = f"lag must be an int, got {type(lag).__name__}"
            raise TypeError(msg)
        if lag <= 0:
            msg = f"lag must be a positive integer, got {lag}"
            raise ValueError(msg)
        return self._pearson_corr_shifted(series, lag)

    def acf(self, nlags: int = 20) -> pl.DataFrame:
        """Compute the autocorrelation function up to nlags.

        Args:
            nlags (int): Maximum number of lags to include. Default is 20.

        Returns:
            pl.DataFrame: DataFrame with a ``lag`` column (0..nlags) and one
                          column per asset containing the ACF values.

        Raises:
            TypeError: If *nlags* is not an ``int``.
            ValueError: If *nlags* is negative.

        """
        if not isinstance(nlags, int):
            msg = f"nlags must be an int, got {type(nlags).__name__}"
            raise TypeError(msg)
        if nlags < 0:
            msg = f"nlags must be non-negative, got {nlags}"
            raise ValueError(msg)
        result: dict[str, list[float]] = {"lag": list(range(nlags + 1))}
        for col, series in self._data.items():
            acf_values: list[float] = [1.0]
            for k in range(1, nlags + 1):
                acf_values.append(self._pearson_corr_shifted(series, k))
            result[col] = acf_values
        return pl.DataFrame(result)
