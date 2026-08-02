"""Temporal reporting metrics.

Capture ratios live in `_capture.py` and the aggregating `summary` /
`annual_breakdown` pair in `_summary.py`; all three are composed into `Stats`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ._core import _drawdown_series, _to_float, columnwise_stat
from ._internals import _comp_return

if TYPE_CHECKING:
    from ..data import Data

# ── Reporting statistics mixin ───────────────────────────────────────────────


class _ReportingStatsMixin:
    """Mixin providing temporal reporting metrics.

    Covers: periods per year, average drawdown, CAGR, expected return, RAR,
    Calmar ratio, recovery factor, max drawdown duration, and monthly win rate.

    Cross-mixin dependencies:
        - _BasicStatsMixin: exposure
    """

    _data: Data
    all: pl.DataFrame

    if TYPE_CHECKING:
        from .._protocol import DataLike

        data: DataLike

        def exposure(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

    @property
    def periods_per_year(self) -> float:
        """Estimate the number of periods per year from the data index spacing.

        Returns:
            float: Estimated number of observations per calendar year.
        """
        return self._data._periods_per_year

    @columnwise_stat
    def avg_drawdown(self, series: pl.Series) -> float:
        """Average drawdown across all underwater periods.

        Returns 0.0 when there are no underwater periods.

        Matches the QuantStats sign convention: drawdown is expressed as a
        negative fraction (e.g. ``-0.2`` for 20% below peak).

        Args:
            series (pl.Series): Series of additive daily returns.

        Returns:
            float: Mean drawdown in [-1, 0].
        """
        dd = _drawdown_series(series)
        in_dd = dd.filter(dd > 0)
        # A series that never falls below its high-water mark has an average drawdown of exactly 0.0.
        if in_dd.is_empty():
            return 0.0
        return -_to_float(in_dd.mean())

    @columnwise_stat
    def cagr(
        self,
        series: pl.Series,
        rf: float = 0.0,
        compounded: bool = True,
        periods: int | float | None = None,
    ) -> float:
        """Calculate the Compound Annual Growth Rate (CAGR) of excess returns.

        CAGR represents the geometric mean annual growth rate, providing a
        smoothed annualized return that accounts for compounding effects.

        Args:
            series (pl.Series): Series of additive daily returns.
            rf (float): Annualized risk-free rate. Defaults to 0.0.
            compounded (bool): Whether to compound returns. Defaults to True.
            periods: Periods per year for annualisation. Defaults to ``periods_per_year``.

        Returns:
            float: CAGR of excess returns.

        Returns NaN when:
            ``float("nan")`` when the series is empty.
        """
        raw_periods = periods or self._data._periods_per_year
        n = len(series)
        if n == 0:
            return float("nan")  # pragma: no cover
        excess = series.cast(pl.Float64) - rf / raw_periods
        total = _comp_return(excess) if compounded else _to_float(excess.sum())
        years = n / raw_periods
        return float(abs(1.0 + total) ** (1.0 / years) - 1.0)

    def expected_return(
        self,
        aggregate: str | None = None,
        compounded: bool = True,
    ) -> dict[str, float]:
        """Expected return with optional period aggregation.

        Returns the arithmetic mean of per-period returns.  When *aggregate* is
        provided the returns are first compounded (or summed) within each
        calendar period, and the mean is taken over those period returns.

        Args:
            aggregate (str | None): Period to aggregate to before computing the
                mean. Accepted values: ``'weekly'``, ``'monthly'``,
                ``'quarterly'``, ``'annual'`` / ``'yearly'``. Defaults to
                ``None`` (raw per-period mean).
            compounded (bool): Compound returns within each period when
                *aggregate* is set. Defaults to ``True``.

        Returns:
            dict[str, float]: Mean return per asset for the specified period.

        Raises:
            ValueError: If *aggregate* is an unrecognised string.

        Note:
            Requires a temporal (Date / Datetime) index when *aggregate* is not
            ``None``; falls back to the raw per-period mean otherwise.

        Returns NaN when:
            Entries are ``float("nan")`` when an asset has no non-null
            observations.
        """
        _freq_map: dict[str, str] = {
            "weekly": "1w",
            "monthly": "1mo",
            "quarterly": "3mo",
            "annual": "1y",
            "yearly": "1y",
        }

        def _geomean(s: pl.Series) -> float:
            """Per-period geometric mean: (product(1 + r))^(1/n) - 1."""
            n = s.count()
            if n == 0:
                return float("nan")
            return float(_to_float((1.0 + s.cast(pl.Float64)).product()) ** (1.0 / n) - 1.0)

        def _raw_expected_returns() -> dict[str, float]:
            """Return the geometric mean of each raw return series."""
            return {col: _geomean(series.drop_nulls()) for col, series in self._data.items()}

        if aggregate is None:
            return _raw_expected_returns()

        if aggregate.lower() not in _freq_map:
            raise ValueError(f"aggregate must be one of {list(_freq_map)}, got {aggregate!r}")  # noqa: TRY003

        all_df = self.all
        date_col_name = self._data.date_col[0] if self._data.date_col else None
        if date_col_name is None or not all_df[date_col_name].dtype.is_temporal():
            return _raw_expected_returns()

        trunc = _freq_map[aggregate.lower()]
        agg_expr = ((1.0 + pl.col("ret")).product() - 1.0) if compounded else pl.col("ret").sum()

        result: dict[str, float] = {}
        for col, series in self._data.items():
            df = (
                pl.DataFrame({"date": all_df[date_col_name], "ret": series})
                .drop_nulls()
                .with_columns(pl.col("date").dt.truncate(trunc).alias("period"))
            )
            period_rets = df.group_by("period").agg(agg_expr.alias("ret"))["ret"]
            result[col] = _geomean(period_rets)
        return result

    def rar(self, periods: int | float = 252) -> dict[str, float]:
        """Risk-Adjusted Return: CAGR divided by exposure.

        Measures annualised return per unit of market participation time,
        matching the quantstats convention.

        Args:
            periods: Periods per year for CAGR annualisation. Defaults to ``periods_per_year``.

        Returns:
            dict[str, float]: RAR per asset.
        """
        cagr = self.cagr(periods=periods)
        exp = self.exposure()
        return {col: cagr[col] / exp[col] for col in cagr}

    @columnwise_stat
    def calmar(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calmar ratio (CAGR divided by maximum drawdown).

        Returns ``nan`` when the maximum drawdown is zero.

        Args:
            series (pl.Series): Series of additive daily returns.
            periods: Annualisation factor. Defaults to ``periods_per_year``.

        Returns:
            float: Calmar ratio, or ``nan`` if max drawdown is zero.
        """
        raw_periods = float(periods or self._data._periods_per_year)
        max_dd = _to_float(_drawdown_series(series).max())
        if max_dd <= 0:
            return float("nan")
        n = len(series)
        comp_return = _comp_return(series)
        cagr = float((1.0 + comp_return) ** (raw_periods / n)) - 1.0
        return cagr / max_dd

    @columnwise_stat
    def recovery_factor(self, series: pl.Series) -> float:
        """Recovery factor (total return divided by maximum drawdown).

        Matches the quantstats convention: total return is the simple sum of
        returns, not compounded.  Returns ``nan`` when the maximum drawdown
        is zero.

        Args:
            series (pl.Series): Series of additive daily returns.

        Returns:
            float: Recovery factor, or ``nan`` if max drawdown is zero.
        """
        max_dd = _to_float(_drawdown_series(series).max())
        if max_dd <= 0:
            return float("nan")
        total_return = _to_float(series.sum())
        return abs(total_return) / max_dd

    def max_drawdown_duration(self) -> dict[str, float | int | None]:
        """Maximum drawdown duration in calendar days (or periods) per asset.

        When the index is a temporal column (``Date`` / ``Datetime``) the
        duration is expressed as calendar days spanned by the longest
        underwater run.  For integer-indexed data each row counts as one
        period.

        Returns:
            dict[str, float | int | None]: Asset → max drawdown duration.
            Returns 0 when there are no underwater periods.
        """
        all_df = self.all
        date_col_name = self._data.date_col[0] if self._data.date_col else None
        has_date = date_col_name is not None and all_df[date_col_name].dtype.is_temporal()
        result: dict[str, float | int | None] = {}
        for col, series in self._data.items():
            nav = 1.0 + series.cast(pl.Float64).cum_sum()
            hwm = nav.cum_max()
            in_dd = nav < hwm

            if not in_dd.any():
                result[col] = 0
                continue

            if has_date and date_col_name is not None:
                frame = pl.DataFrame({"date": all_df[date_col_name], "in_dd": in_dd})
            else:
                frame = pl.DataFrame({"date": pl.Series(list(range(len(series))), dtype=pl.Int64), "in_dd": in_dd})

            frame = frame.with_columns(pl.col("in_dd").rle_id().alias("run_id"))
            dd_runs = (
                frame.filter(pl.col("in_dd"))
                .group_by("run_id")
                .agg([pl.col("date").min().alias("start"), pl.col("date").max().alias("end")])
            )

            if has_date:
                dd_runs = dd_runs.with_columns(
                    ((pl.col("end") - pl.col("start")).dt.total_days() + 1).alias("duration")
                )
            else:
                dd_runs = dd_runs.with_columns((pl.col("end") - pl.col("start") + 1).alias("duration"))

            result[col] = int(_to_float(dd_runs["duration"].max()))
        return result

    def monthly_win_rate(self) -> dict[str, float]:
        """Fraction of calendar months with a positive compounded return per asset.

        Requires a temporal (Date / Datetime) index.  Returns ``nan`` per
        asset when no temporal index is present.

        Returns:
            dict[str, float]: Monthly win rate in [0, 1] per asset.

        Returns NaN when:
            Entries are ``float("nan")`` when no temporal index is present or an
            asset has no non-null observations.
        """
        all_df = self.all
        date_col_name = self._data.date_col[0] if self._data.date_col else None
        if date_col_name is None or not all_df[date_col_name].dtype.is_temporal():
            return {col: float("nan") for col, _ in self._data.items()}

        result: dict[str, float] = {}
        for col, _ in self._data.items():
            df = (
                all_df.select([date_col_name, col])
                .drop_nulls()
                .with_columns(
                    [
                        pl.col(date_col_name).dt.year().alias("_year"),
                        pl.col(date_col_name).dt.month().alias("_month"),
                    ]
                )
            )
            monthly = (
                df.group_by(["_year", "_month"])
                .agg((pl.col(col) + 1.0).product().alias("gross"))
                .with_columns((pl.col("gross") - 1.0).alias("monthly_return"))
            )
            n_total = len(monthly)
            if n_total == 0:
                result[col] = float("nan")
            else:
                n_positive = int((monthly["monthly_return"] > 0).sum())
                result[col] = n_positive / n_total
        return result
