"""Basic statistical metrics for financial returns data."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from scipy.stats import norm

from ._core import _mean, columnwise_stat
from ._internals import _annualization_factor, _comp_return

if TYPE_CHECKING:
    from ..data import Data

# ── Basic statistics mixin ───────────────────────────────────────────────────


class _BasicStatsMixin:
    """Mixin providing basic return/risk and win/loss financial statistics.

    Covers: basic statistics (skew, kurtosis, avg return/win/loss), volatility,
    win/loss metrics (payoff ratio, profit factor), and risk metrics (VaR, CVaR,
    win rate, kelly criterion, best/worst, exposure).
    """

    _data: Data
    all: pl.DataFrame

    if TYPE_CHECKING:
        from ._protocol import DataLike

        data: DataLike
        all: pl.DataFrame | None

    @staticmethod
    def _positive(series: pl.Series) -> pl.Series:
        """Return only the positive values in *series*."""
        return series.filter(series > 0)

    @staticmethod
    def _negative(series: pl.Series) -> pl.Series:
        """Return only the negative values in *series*."""
        return series.filter(series < 0)

    @staticmethod
    def _mean_positive_expr(series: pl.Series) -> float:
        """Return the mean of all positive values in *series*, or NaN if none exist."""
        return _mean(_BasicStatsMixin._positive(series))

    @staticmethod
    def _mean_negative_expr(series: pl.Series) -> float:
        """Return the mean of all negative values in *series*, or NaN if none exist."""
        return _mean(_BasicStatsMixin._negative(series))

    # ── Basic statistics ──────────────────────────────────────────────────────

    @columnwise_stat
    def skew(self, series: pl.Series) -> int | float | None:
        """Calculate skewness (asymmetry) for each numeric column.

        Args:
            series (pl.Series): The series to calculate skewness for.

        Returns:
            float: The skewness value.

        """
        return series.skew(bias=False)

    @columnwise_stat
    def kurtosis(self, series: pl.Series) -> int | float | None:
        """Calculate the kurtosis of returns.

        The degree to which a distribution peak compared to a normal distribution.

        Args:
            series (pl.Series): The series to calculate kurtosis for.

        Returns:
            float: The kurtosis value.

        """
        return series.kurtosis(bias=False)

    @columnwise_stat
    def avg_return(self, series: pl.Series) -> float:
        """Calculate average return per non-zero value.

        Args:
            series (pl.Series): The series to calculate average return for.

        Returns:
            float: The average return value.

        """
        return _mean(series.filter(series.is_not_null() & (series != 0)))

    @columnwise_stat
    def avg_win(self, series: pl.Series) -> float:
        """Calculate the average winning return/trade for an asset.

        Args:
            series (pl.Series): The series to calculate average win for.

        Returns:
            float: The average winning return.

        """
        return self._mean_positive_expr(series)

    @columnwise_stat
    def avg_loss(self, series: pl.Series) -> float:
        """Calculate the average loss return/trade for a period.

        Args:
            series (pl.Series): The series to calculate average loss for.

        Returns:
            float: The average loss return.

        """
        return self._mean_negative_expr(series)

    @columnwise_stat
    def comp(self, series: pl.Series) -> float:
        """Calculate the total compounded return over the full period.

        Computed as product(1 + r) - 1.

        Args:
            series (pl.Series): The series to calculate compounded return for.

        Returns:
            float: Total compounded return.

        """
        return _comp_return(series)

    @columnwise_stat
    def geometric_mean(self, series: pl.Series, periods: int | float | None = None, annualize: bool = False) -> float:
        """Calculate the geometric mean of returns.

        Computed as the per-period geometric average: (∏(1 + rᵢ))^(1/n) - 1.
        When annualized, raises to the power of periods_per_year instead of 1/n.

        Args:
            series (pl.Series): The series to calculate geometric mean for.
            periods (int | float, optional): Periods per year for annualization. Defaults to periods_per_year.
            annualize (bool): Whether to annualize the result. Defaults to False.

        Returns:
            float: The geometric mean return.

        """
        clean = series.drop_nulls().cast(pl.Float64)
        n = clean.len()
        if n == 0:
            return float("nan")  # indeterminate: no observations
        compound = float((1.0 + clean).product())
        if compound <= 0:
            return float("nan")  # indeterminate: non-positive compound return
        exponent = (periods or self._data._periods_per_year) / n if annualize else (1.0 / n)
        return float(compound**exponent) - 1.0

    # ── Volatility & risk ─────────────────────────────────────────────────────

    @columnwise_stat
    def volatility(self, series: pl.Series, periods: int | float | None = None, annualize: bool = True) -> float:
        """Calculate the volatility of returns.

        - Std dev of returns
        - Annualized by sqrt(periods) if `annualize` is True.

        Args:
            series (pl.Series): The series to calculate volatility for.
            periods (int, optional): Number of periods per year. Defaults to 252.
            annualize (bool, optional): Whether to annualize the result. Defaults to True.

        Returns:
            float: The volatility value.

        """
        raw_periods = periods or self._data._periods_per_year

        # Ensure it's numeric
        if not isinstance(raw_periods, int | float):
            raise TypeError(f"Expected int or float for periods, got {type(raw_periods).__name__}")  # noqa: TRY003

        factor = _annualization_factor(raw_periods) if annualize else 1.0
        std_val = cast(float, series.std())
        return (std_val if std_val is not None else 0.0) * factor

    # ── Win / loss metrics ────────────────────────────────────────────────────

    @columnwise_stat
    def payoff_ratio(self, series: pl.Series) -> float:
        """Measure the payoff ratio.

        The payoff ratio is calculated as average win / abs(average loss).

        Args:
            series (pl.Series): The series to calculate payoff ratio for.

        Returns:
            float: The payoff ratio value.

        """
        avg_win = self._mean_positive_expr(series)
        avg_loss = float(np.abs(self._mean_negative_expr(series)))
        return avg_win / avg_loss

    @columnwise_stat
    def profit_ratio(self, series: pl.Series) -> float:
        """Measure the profit ratio.

        The profit ratio is calculated as win ratio / loss ratio.

        Args:
            series (pl.Series): The series to calculate profit ratio for.

        Returns:
            float: The profit ratio value.

        """
        wins = series.filter(series >= 0)
        losses = self._negative(series)

        # Filtering can legitimately leave no wins or no losses for one-sided return series.
        if wins.is_empty() or losses.is_empty():
            return float("nan")  # indeterminate: no wins or no losses

        win_mean = _mean(wins)
        loss_mean = _mean(losses)
        win_ratio = float(np.abs(win_mean / wins.count()))
        loss_ratio = float(np.abs(loss_mean / losses.count()))

        return win_ratio / loss_ratio

    @columnwise_stat
    def profit_factor(self, series: pl.Series) -> float:
        """Measure the profit factor.

        The profit factor is calculated as wins / loss.

        Args:
            series (pl.Series): The series to calculate profit factor for.

        Returns:
            float: The profit factor value.

        """
        wins = self._positive(series)
        losses = self._negative(series)
        wins_sum = wins.sum()
        losses_sum = losses.sum()

        return float(np.abs(float(wins_sum) / float(losses_sum)))

    # ── Risk metrics ──────────────────────────────────────────────────────────

    @columnwise_stat
    def value_at_risk(self, series: pl.Series, sigma: float = 1.0, alpha: float = 0.05) -> float:
        """Calculate the daily value-at-risk.

        Uses variance-covariance calculation with confidence level.

        Args:
            series (pl.Series): The series to calculate value at risk for.
            alpha (float, optional): Confidence level. Defaults to 0.05.
            sigma (float, optional): Standard deviation multiplier. Defaults to 1.0.

        Returns:
            float: The value at risk.

        """
        mean_val = _mean(series)
        std_val = cast(float, series.std())
        mu = mean_val
        sigma *= std_val if std_val is not None else 0.0

        return float(norm.ppf(alpha, mu, sigma))

    @columnwise_stat
    def _conditional_value_at_risk_impl(self, series: pl.Series, sigma: float = 1.0, alpha: float = 0.05) -> float:
        """Inner per-series implementation of conditional value-at-risk."""
        mean_val = _mean(series)
        std_val = cast(float, series.std())
        mu = mean_val
        sigma *= std_val if std_val is not None else 0.0

        var = norm.ppf(alpha, mu, sigma)

        # Compute mean of returns less than or equal to VaR
        # Cast to Any or pl.Series to suppress Ty error
        # Cast the mask to pl.Expr to satisfy type checker
        mask = cast(Iterable[bool], series < var)
        return _mean(series.filter(mask))

    def conditional_value_at_risk(
        self, sigma: float = 1.0, confidence: float = 0.95, **kwargs: float
    ) -> dict[str, float]:
        """Calculate the conditional value-at-risk (CVaR / Expected Shortfall).

        Also known as CVaR or expected shortfall, calculated for each numeric column.

        Args:
            sigma (float, optional): Standard deviation multiplier. Defaults to 1.0.
            confidence (float, optional): Confidence level (e.g. 0.95 for 95 %).
                Converted internally to ``alpha = 1 - confidence``. Defaults to 0.95.
            alpha (float, optional): Tail probability (lower tail).  ``alpha`` is the
                probability mass in the *loss* tail, so ``alpha = 1 - confidence``.
                For example, a 95 % confidence level corresponds to ``alpha = 0.05``
                (the default).
            **kwargs: Legacy keyword arguments.  Passing ``confidence`` (e.g.
                ``confidence=0.95``) is accepted for backwards compatibility with
                QuantStats but emits a `DeprecationWarning`.  Use
                ``alpha = 1 - confidence`` instead.

        Returns:
            dict[str, float]: The conditional value at risk per asset column.

        Raises:
            TypeError: If unexpected keyword arguments are passed.

        """
        return self._conditional_value_at_risk_impl(sigma=sigma, alpha=1.0 - confidence)

    @staticmethod
    def _drawdown_with_baseline(series: pl.Series) -> pl.Series:
        """Compute drawdown series with a phantom zero-return baseline prepended.

        Matches the quantstats convention: a negative first return is treated as
        a drawdown from the initial capital of 1.0, not as the new high-water mark.
        """
        extended = pl.concat([pl.Series([0.0]), series.cast(pl.Float64)])
        nav = (1.0 + extended).cum_prod()
        hwm = nav.cum_max()
        # The phantom baseline pins nav[0] = 1.0, so hwm >= 1.0 throughout and
        # the 1e-10 floor is purely defensive (unreachable); a -100 % return
        # correctly reports as a full drawdown of 1.0 here.
        dd = ((hwm - nav) / hwm.clip(lower_bound=1e-10)).clip(lower_bound=0.0)
        return dd[1:]  # drop phantom point

    @staticmethod
    def _ulcer_index_series(series: pl.Series) -> float:
        """Compute ulcer index for a single returns series."""
        dd = _BasicStatsMixin._drawdown_with_baseline(series)
        n = series.len()
        return float(np.sqrt(float((dd**2).sum()) / (n - 1)))

    @columnwise_stat
    def ulcer_index(self, series: pl.Series) -> float:
        """Calculate the Ulcer Index (downside risk measurement).

        Measures the depth and duration of drawdowns as the root mean square
        of squared drawdowns: sqrt(sum(dd²) / (n - 1)).

        Args:
            series (pl.Series): The series to calculate ulcer index for.

        Returns:
            float: Ulcer Index value.

        """
        return self._ulcer_index_series(series)

    @columnwise_stat
    def ulcer_performance_index(self, series: pl.Series, rf: float = 0.0) -> float:
        """Calculate the Ulcer Performance Index (UPI).

        Risk-adjusted return using Ulcer Index as the risk measure:
        (compounded_return - rf) / ulcer_index.

        Args:
            series (pl.Series): The series to calculate UPI for.
            rf (float): Risk-free rate. Defaults to 0.

        Returns:
            float: Ulcer Performance Index.

        """
        comp = _comp_return(series)
        ui = self._ulcer_index_series(series)
        return float("nan") if ui == 0 else (comp - rf) / ui

    @columnwise_stat
    def serenity_index(self, series: pl.Series, rf: float = 0.0) -> float:
        """Calculate the Serenity Index.

        Combines the Ulcer Index with a CVaR-based pitfall measure:
        (sum_returns - rf) / (ulcer_index * pitfall), where
        pitfall = -CVaR(drawdowns) / std(returns).

        Args:
            series (pl.Series): The series to calculate serenity index for.
            rf (float): Risk-free rate. Defaults to 0.

        Returns:
            float: Serenity Index.

        """
        std_val = cast(float, series.std())
        if not std_val:
            return float("nan")  # indeterminate: zero variance

        # Negate drawdowns to match quantstats sign convention (negative = below peak)
        dd_neg = -self._drawdown_with_baseline(series)
        mu = _mean(dd_neg)
        sigma = cast(float, dd_neg.std())
        var_threshold = float(norm.ppf(0.05, mu, sigma))
        mask = cast(Iterable[bool], dd_neg < var_threshold)
        cvar_val = _mean(dd_neg.filter(mask))

        pitfall = -cvar_val / std_val
        ui = self._ulcer_index_series(series)
        denominator = ui * pitfall
        return float("nan") if denominator == 0 else (float(series.sum()) - rf) / denominator

    @columnwise_stat
    def win_rate(self, series: pl.Series) -> float:
        """Calculate the win ratio for a period.

        Args:
            series (pl.Series): The series to calculate win rate for.

        Returns:
            float: The win rate value.

        """
        num_pos = self._positive(series).count()
        num_nonzero = series.filter(series != 0).count()
        return float(num_pos / num_nonzero)

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
        all_data = cast(pl.DataFrame, self.all)
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
