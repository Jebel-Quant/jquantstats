"""Core descriptive, return, volatility and risk statistics.

The `_BasicCoreMixin` here holds the leaf statistics that the composite
ratios in `_basic` build on, plus the shared static helpers.
"""

from __future__ import annotations

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


class _BasicCoreMixin:
    """Mixin providing basic return/risk and win/loss financial statistics.

    Covers: basic statistics (skew, kurtosis, avg return/win/loss), volatility,
    win/loss metrics (payoff ratio, profit factor), and risk metrics (VaR, CVaR,
    win rate, kelly criterion, best/worst, exposure).
    """

    _data: Data
    all: pl.DataFrame

    if TYPE_CHECKING:
        from .._protocol import DataLike

        data: DataLike

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
        return _mean(_BasicCoreMixin._positive(series))

    @staticmethod
    def _mean_negative_expr(series: pl.Series) -> float:
        """Return the mean of all negative values in *series*, or NaN if none exist."""
        return _mean(_BasicCoreMixin._negative(series))

    @staticmethod
    def _gaussian_quantile(alpha: float, mu: float, sigma: float) -> float:
        """Gaussian inverse-CDF (``norm.ppf``) returning NaN for a zero-scale input.

        ``norm.ppf(alpha, mu, 0.0)`` already returns ``nan`` for a degenerate
        (zero-variance) distribution — but it emits an ``invalid value
        encountered in multiply`` RuntimeWarning while doing so (``inf * 0``
        internally).  Degenerate scale arises for a single observation (undefined
        std) or a constant series.  Short-circuiting to ``float("nan")`` keeps the
        exact same result while suppressing the spurious warning; downstream
        masking relies on this NaN (Polars treats ``x < nan`` as ``True``).
        """
        return float("nan") if sigma == 0.0 else float(norm.ppf(alpha, mu, sigma))

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


        Returns NaN when:
            ``float("nan")`` when the series has no non-null observations or the
            compounded return ``product(1 + r)`` is non-positive.
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


        Returns NaN when:
            ``float("nan")`` when the series has no wins or no losses.
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

        return self._gaussian_quantile(alpha, mu, sigma)

    @columnwise_stat
    def _conditional_value_at_risk_impl(self, series: pl.Series, sigma: float = 1.0, alpha: float = 0.05) -> float:
        """Inner per-series implementation of conditional value-at-risk."""
        mean_val = _mean(series)
        std_val = cast(float, series.std())
        mu = mean_val
        sigma *= std_val if std_val is not None else 0.0

        var = self._gaussian_quantile(alpha, mu, sigma)

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
        dd = _BasicCoreMixin._drawdown_with_baseline(series)
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


        Returns NaN when:
            ``float("nan")`` when the ulcer index is zero (no drawdowns).
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


        Returns NaN when:
            ``float("nan")`` when the returns have zero (or undefined) standard
            deviation or the denominator ``ulcer_index * pitfall`` is zero.
        """
        std_val = cast(float, series.std())
        if not std_val:
            return float("nan")  # indeterminate: zero variance

        # Negate drawdowns to match quantstats sign convention (negative = below peak)
        dd_neg = -self._drawdown_with_baseline(series)
        mu = _mean(dd_neg)
        sigma = cast(float, dd_neg.std())
        var_threshold = self._gaussian_quantile(0.05, mu, sigma)
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
