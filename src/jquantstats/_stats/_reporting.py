"""Temporal reporting, capture ratios, and summary statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import polars as pl

from ._core import _drawdown_series, _to_float, columnwise_stat
from ._internals import _comp_return

if TYPE_CHECKING:
    from ..data import Data

# ── Reporting statistics mixin ───────────────────────────────────────────────


class _ReportingStatsMixin:
    """Mixin providing temporal, capture, and summary reporting metrics.

    Covers: periods per year, average drawdown, Calmar ratio, recovery factor,
    max drawdown duration, monthly win rate, worst-N periods, up/down capture
    ratios, annual breakdown, and summary statistics table.

    Cross-mixin dependencies:
        - _BasicStatsMixin: avg_return, avg_win, avg_loss, win_rate, profit_factor,
          payoff_ratio, best, worst, volatility, skew, kurtosis, value_at_risk,
          conditional_value_at_risk, exposure
        - _RiskStatsMixin: sharpe, max_drawdown
    """

    _data: Data
    all: pl.DataFrame

    if TYPE_CHECKING:
        from .._protocol import DataLike

        data: DataLike

        def avg_return(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def avg_win(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def avg_loss(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def win_rate(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def profit_factor(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def payoff_ratio(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def best(self) -> dict[str, float | None]:
            """Defined on _BasicStatsMixin."""

        def worst(self) -> dict[str, float | None]:
            """Defined on _BasicStatsMixin."""

        def volatility(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def sharpe(self) -> dict[str, float]:
            """Defined on _RiskStatsMixin."""

        def skew(self) -> dict[str, int | float | None]:
            """Defined on _BasicStatsMixin."""

        def kurtosis(self) -> dict[str, int | float | None]:
            """Defined on _BasicStatsMixin."""

        def value_at_risk(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def conditional_value_at_risk(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def max_drawdown(self) -> dict[str, float]:
            """Defined on _RiskStatsMixin."""

        def exposure(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

    # ── Temporal & reporting ──────────────────────────────────────────────────

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

    def monthly_returns(self, eoy: bool = True, compounded: bool = True) -> dict[str, pl.DataFrame]:
        """Calculate monthly returns in a pivot-table format.

        Groups returns by calendar month and year, producing a DataFrame with
        years as rows and months (JAN-DEC) as columns, plus an optional EOY
        column with the full-year compounded return.

        Args:
            eoy (bool): Include an EOY column with the annual compounded return.
                Defaults to True.
            compounded (bool): Compound returns within each period. Defaults to True.

        Returns:
            dict[str, pl.DataFrame]: Per-asset pivot tables with columns
                ``year``, ``JAN`` … ``DEC``, and optionally ``EOY``.

        """
        all_df = self.all
        date_col_name = self._data.date_col[0]
        month_names = {
            1: "JAN",
            2: "FEB",
            3: "MAR",
            4: "APR",
            5: "MAY",
            6: "JUN",
            7: "JUL",
            8: "AUG",
            9: "SEP",
            10: "OCT",
            11: "NOV",
            12: "DEC",
        }
        month_order = list(month_names.values())

        result: dict[str, pl.DataFrame] = {}
        for col, series in self._data.items():
            df = pl.DataFrame({"date": all_df[date_col_name], "ret": series}).drop_nulls()
            df = df.with_columns(
                [
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month_num"),
                ]
            )

            agg_expr = ((1.0 + pl.col("ret")).product() - 1.0) if compounded else pl.col("ret").sum()
            monthly = (
                df.group_by(["year", "month_num"])
                .agg(agg_expr.alias("ret"))
                .with_columns(
                    pl.col("month_num")
                    .replace_strict(
                        list(month_names.keys()),
                        list(month_names.values()),
                        return_dtype=pl.String,
                    )
                    .alias("month_name")
                )
                .sort(["year", "month_num"])
            )

            pivoted = monthly.pivot(on="month_name", index="year", values="ret", aggregate_function="first")
            for m in month_order:
                if m not in pivoted.columns:
                    pivoted = pivoted.with_columns(pl.lit(0.0).alias(m))
            pivoted = (
                pivoted.select(["year", *month_order])
                .fill_null(0.0)
                .with_columns(pl.col("year").cast(pl.Int32))
                .sort("year")
            )

            if eoy:
                eoy_agg = (
                    df.group_by("year")
                    .agg(agg_expr.alias("EOY"))
                    .with_columns(pl.col("year").cast(pl.Int32))
                    .sort("year")
                )
                pivoted = pivoted.join(eoy_agg, on="year").sort("year")

            result[col] = pivoted
        return result

    def distribution(self, compounded: bool = True) -> dict[str, dict[str, dict[str, list[float]]]]:
        """Analyse return distributions across daily, weekly, monthly, quarterly, and yearly periods.

        For each period, splits values into inliers and outliers using the
        IQR method (1.5 * IQR beyond Q1/Q3).

        Args:
            compounded (bool): Compound returns within each period. Defaults to True.

        Returns:
            dict: Nested dict ``{asset: {period: {"values": [...], "outliers": [...]}}}``
                where period is one of ``"Daily"``, ``"Weekly"``, ``"Monthly"``,
                ``"Quarterly"``, ``"Yearly"``.

        """
        all_df = self.all
        date_col_name = self._data.date_col[0]

        def _agg(df: pl.DataFrame, group_col: str) -> pl.Series:
            """Aggregate returns within each group using product or sum."""
            expr = ((1.0 + pl.col("ret")).product() - 1.0) if compounded else pl.col("ret").sum()
            return df.group_by(group_col).agg(expr.alias("ret"))["ret"]

        def _iqr_split(s: pl.Series) -> dict[str, list[float]]:
            """Split series into inliers and outliers using the IQR method."""
            q1 = cast(float, s.quantile(0.25))
            q3 = cast(float, s.quantile(0.75))
            iqr = q3 - q1
            mask = (s >= q1 - 1.5 * iqr) & (s <= q3 + 1.5 * iqr)
            return {"values": s.filter(mask).to_list(), "outliers": s.filter(~mask).to_list()}

        result: dict[str, dict[str, dict[str, list[float]]]] = {}
        for col, series in self._data.items():
            df = pl.DataFrame({"date": all_df[date_col_name], "ret": series}).drop_nulls()
            df = df.with_columns(
                [
                    pl.col("date").dt.truncate("1w").alias("week"),
                    pl.col("date").dt.truncate("1mo").alias("month"),
                    pl.col("date").dt.truncate("3mo").alias("quarter"),
                    pl.col("date").dt.truncate("1y").alias("year"),
                ]
            )
            result[col] = {
                "Daily": _iqr_split(df["ret"]),
                "Weekly": _iqr_split(_agg(df, "week")),
                "Monthly": _iqr_split(_agg(df, "month")),
                "Quarterly": _iqr_split(_agg(df, "quarter")),
                "Yearly": _iqr_split(_agg(df, "year")),
            }
        return result

    def compare(
        self,
        aggregate: str | None = None,
        compounded: bool = True,
        round_vals: int | None = None,
    ) -> dict[str, pl.DataFrame]:
        """Compare each asset's returns against the benchmark.

        Aligns returns and benchmark by date, multiplies by 100 (percentage),
        then computes a ``Multiplier`` (Returns / Benchmark) and ``Won``
        indicator (``"+"`` when the asset outperformed, ``"-"`` otherwise).

        Args:
            aggregate (str | None): Pandas-style resample frequency for
                period aggregation (e.g. ``"ME"``, ``"QE"``, ``"YE"``).
                ``None`` returns daily rows. Defaults to None.
            compounded (bool): Compound returns when aggregating. Defaults to True.
            round_vals (int | None): Decimal places to round. Defaults to None.

        Returns:
            dict[str, pl.DataFrame]: Per-asset DataFrames with columns
                ``Benchmark``, ``Returns``, ``Multiplier``, ``Won``.

        Raises:
            AttributeError: If no benchmark data is attached.

        """
        if self._data.benchmark is None:
            raise AttributeError("No benchmark data available")  # noqa: TRY003

        all_df = self.all
        date_col_name = self._data.date_col[0]
        bench_col = self._data.benchmark.columns[0]

        _freq_map = {"ME": "1mo", "QE": "3mo", "YE": "1y", "W": "1w"}

        def _agg_series(df: pl.DataFrame, period_col: str, val_col: str) -> pl.DataFrame:
            """Aggregate a value column grouped by period using product or sum."""
            expr = ((1.0 + pl.col(val_col)).product() - 1.0) if compounded else pl.col(val_col).sum()
            return df.group_by(period_col).agg(expr.alias(val_col)).sort(period_col)

        result: dict[str, pl.DataFrame] = {}
        for col in self._data.returns.columns:
            df = all_df.select(
                [
                    pl.col(date_col_name),
                    pl.col(col).alias("ret"),
                    pl.col(bench_col).alias("bench"),
                ]
            )

            if aggregate is not None and aggregate in _freq_map:
                trunc = _freq_map[aggregate]
                df = df.with_columns(pl.col(date_col_name).dt.truncate(trunc).alias("period"))
                ret_agg = _agg_series(df.drop_nulls(subset=["ret"]), "period", "ret")
                bench_agg = _agg_series(df.drop_nulls(subset=["bench"]), "period", "bench")
                df = ret_agg.join(bench_agg, on="period", how="full", coalesce=True).sort("period")
                ret_col, bench_col_name, _date_alias = "ret", "bench", "period"
            else:
                ret_col, bench_col_name, _date_alias = "ret", "bench", date_col_name

            ret_pct = (df[ret_col] * 100).alias("Returns")
            bench_pct = (df[bench_col_name] * 100).alias("Benchmark")
            out = pl.DataFrame(
                {
                    "Benchmark": bench_pct,
                    "Returns": ret_pct,
                }
            )
            out = out.with_columns(
                [
                    (pl.col("Returns") / pl.col("Benchmark").replace(0.0, None)).alias("Multiplier"),
                    pl.when(pl.col("Returns") >= pl.col("Benchmark"))
                    .then(pl.lit("+"))
                    .otherwise(pl.lit("-"))
                    .alias("Won"),
                ]
            )

            if round_vals is not None:
                out = out.with_columns(
                    [
                        pl.col("Benchmark").round(round_vals),
                        pl.col("Returns").round(round_vals),
                        pl.col("Multiplier").round(round_vals),
                    ]
                )

            result[col] = out
        return result

    def worst_n_periods(self, n: int = 5) -> dict[str, list[float | None]]:
        """Return the N worst return periods per asset.

        If a series has fewer than ``n`` non-null observations the list is
        padded with ``None`` on the right.

        Args:
            n: Number of worst periods to return. Defaults to 5.

        Returns:
            dict[str, list[float | None]]: Sorted worst returns per asset.
        """
        result: dict[str, list[float | None]] = {}
        for col, series in self._data.items():
            nonnull = series.drop_nulls()
            worst: list[float | None] = nonnull.sort(descending=False).head(n).to_list()
            while len(worst) < n:
                worst.append(None)
            result[col] = worst
        return result

    # ── Capture ratios ────────────────────────────────────────────────────────

    def up_capture(self, benchmark: pl.Series) -> dict[str, float]:
        """Up-market capture ratio relative to an explicit benchmark series.

        Measures the fraction of the benchmark's upside that the strategy
        captures.  A value greater than 1.0 means the strategy outperformed
        the benchmark in rising markets.

        Args:
            benchmark: Benchmark return series aligned row-by-row with the data.

        Returns:
            dict[str, float]: Up capture ratio per asset.

        Returns NaN when:
            Entries are ``float("nan")`` when the benchmark has no positive
            periods, its up-market geometric mean is zero, or an asset has no
            usable returns during those periods.
        """
        up_mask = benchmark > 0
        bench_up = benchmark.filter(up_mask).drop_nulls()
        # A benchmark with no positive periods makes up-capture undefined for every asset.
        if bench_up.is_empty():
            return {col: float("nan") for col, _ in self._data.items()}
        bench_geom = float((bench_up + 1.0).product()) ** (1.0 / len(bench_up)) - 1.0
        if bench_geom == 0.0:  # pragma: no cover
            return {col: float("nan") for col, _ in self._data.items()}
        result: dict[str, float] = {}
        for col, series in self._data.items():
            strat_up = series.filter(up_mask).drop_nulls()
            # An asset may have no usable returns during the benchmark's up periods after null filtering.
            if strat_up.is_empty():
                result[col] = float("nan")
            else:
                strat_geom = float((strat_up + 1.0).product()) ** (1.0 / len(strat_up)) - 1.0
                result[col] = strat_geom / bench_geom
        return result

    def down_capture(self, benchmark: pl.Series) -> dict[str, float]:
        """Down-market capture ratio relative to an explicit benchmark series.

        A value less than 1.0 means the strategy lost less than the benchmark
        in falling markets (a desirable property).

        Args:
            benchmark: Benchmark return series aligned row-by-row with the data.

        Returns:
            dict[str, float]: Down capture ratio per asset.

        Returns NaN when:
            Entries are ``float("nan")`` when the benchmark has no negative
            periods, its down-market geometric mean is zero, or an asset has no
            usable returns during those periods.
        """
        down_mask = benchmark < 0
        bench_down = benchmark.filter(down_mask).drop_nulls()
        # A benchmark with no negative periods makes down-capture undefined for every asset.
        if bench_down.is_empty():
            return {col: float("nan") for col, _ in self._data.items()}
        bench_geom = float((bench_down + 1.0).product()) ** (1.0 / len(bench_down)) - 1.0
        if bench_geom == 0.0:  # pragma: no cover
            return {col: float("nan") for col, _ in self._data.items()}
        result: dict[str, float] = {}
        for col, series in self._data.items():
            strat_down = series.filter(down_mask).drop_nulls()
            # An asset may have no usable returns during the benchmark's down periods after null filtering.
            if strat_down.is_empty():
                result[col] = float("nan")
            else:
                strat_geom = float((strat_down + 1.0).product()) ** (1.0 / len(strat_down)) - 1.0
                result[col] = strat_geom / bench_geom
        return result

    # ── Summary & breakdown ────────────────────────────────────────────────────

    def annual_breakdown(self) -> pl.DataFrame:
        """Summary statistics broken down by calendar year.

        Groups the data by calendar year using the date index, computes a
        full `summary` for each year, and stacks the results with an
        additional ``year`` column.

        Returns:
            pl.DataFrame: Columns ``year``, ``metric``, one per asset, sorted
            by ``year``.

        Raises:
            ValueError: If the data has no date index.
        """
        all_df = self.all
        date_col_name = self._data.date_col[0] if self._data.date_col else None
        has_temporal = date_col_name is not None and all_df[date_col_name].dtype.is_temporal()

        from ..data import Data

        if not has_temporal:
            # Integer-index fallback: group by chunks of ~_periods_per_year rows
            chunk = round(self._data._periods_per_year)
            total = all_df.height
            frames_int: list[pl.DataFrame] = []
            for i, start in enumerate(range(0, total, chunk), start=1):
                chunk_all = all_df.slice(start, chunk)
                if chunk_all.height < max(5, chunk // 4):
                    continue
                chunk_index = chunk_all.select(self._data.date_col)
                chunk_returns = chunk_all.select(self._data.returns.columns)
                chunk_benchmark = (
                    chunk_all.select(self._data.benchmark.columns) if self._data.benchmark is not None else None
                )
                chunk_data = Data(returns=chunk_returns, index=chunk_index, benchmark=chunk_benchmark)
                chunk_summary = cast(Any, type(self))(chunk_data).summary()
                chunk_summary = chunk_summary.with_columns(pl.lit(i).alias("year"))
                frames_int.append(chunk_summary)
            if not frames_int:
                return pl.DataFrame()
            result_int = pl.concat(frames_int)
            ordered_int = ["year", "metric", *[c for c in result_int.columns if c not in ("year", "metric")]]
            return result_int.select(ordered_int)

        if date_col_name is None:  # unreachable: has_temporal guarantees non-None  # pragma: no cover
            return pl.DataFrame()  # pragma: no cover
        years = all_df[date_col_name].dt.year().unique().sort().to_list()

        frames: list[pl.DataFrame] = []
        for year in years:
            year_all = all_df.filter(pl.col(date_col_name).dt.year() == year)
            if year_all.height < 2:
                continue
            year_index = year_all.select([date_col_name])
            year_returns = year_all.select(self._data.returns.columns)
            year_benchmark = year_all.select(self._data.benchmark.columns) if self._data.benchmark is not None else None
            year_data = Data(returns=year_returns, index=year_index, benchmark=year_benchmark)
            year_summary = cast(Any, type(self))(year_data).summary()
            year_summary = year_summary.with_columns(pl.lit(year).alias("year"))
            frames.append(year_summary)

        if not frames:
            asset_cols = list(self._data.returns.columns)
            schema: dict[str, type[pl.DataType]] = {
                "year": pl.Int32,
                "metric": pl.String,
                **dict.fromkeys(asset_cols, pl.Float64),
            }
            return pl.DataFrame(schema=schema)

        result = pl.concat(frames)
        ordered = ["year", "metric", *[c for c in result.columns if c not in ("year", "metric")]]
        return result.select(ordered)

    def summary(self) -> pl.DataFrame:
        """Summary statistics for each asset as a tidy DataFrame.

        Each row is one metric; each column beyond ``metric`` is one asset.

        Returns:
            pl.DataFrame: A DataFrame with a ``metric`` column followed by one
            column per asset.

        Returns NaN when:
            Cells are ``float("nan")`` when the underlying metric is unavailable
            for the data (e.g. no temporal index or no benchmark).
        """
        assets = [col for col, _ in self._data.items()]

        def _safe(fn: Any) -> dict[str, Any]:
            """Call *fn()* and return its result; return NaN for each asset on any exception."""
            try:
                result: dict[str, Any] = fn()
            except Exception:
                return dict.fromkeys(assets, float("nan"))
            return result

        metrics: dict[str, dict[str, Any]] = {
            "avg_return": _safe(self.avg_return),
            "avg_win": _safe(self.avg_win),
            "avg_loss": _safe(self.avg_loss),
            "win_rate": _safe(self.win_rate),
            "profit_factor": _safe(self.profit_factor),
            "payoff_ratio": _safe(self.payoff_ratio),
            "monthly_win_rate": _safe(self.monthly_win_rate),
            "best": _safe(self.best),
            "worst": _safe(self.worst),
            "volatility": _safe(self.volatility),
            "sharpe": _safe(self.sharpe),
            "skew": _safe(self.skew),
            "kurtosis": _safe(self.kurtosis),
            "value_at_risk": _safe(self.value_at_risk),
            "conditional_value_at_risk": _safe(self.conditional_value_at_risk),
            "max_drawdown": _safe(self.max_drawdown),
            "avg_drawdown": _safe(self.avg_drawdown),
            "max_drawdown_duration": _safe(self.max_drawdown_duration),
            "calmar": _safe(self.calmar),
            "recovery_factor": _safe(self.recovery_factor),
        }

        rows: list[dict[str, Any]] = [
            {"metric": name, **{asset: values.get(asset) for asset in assets}} for name, values in metrics.items()
        ]
        return pl.DataFrame(rows)
