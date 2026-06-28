"""Period-bucketed reporting tables for financial returns data.

Tabular, period-grouped views of returns: the monthly-returns pivot, the
inlier/outlier distribution across calendar frequencies, the benchmark
comparison table, and the worst-N-periods list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl

from ..exceptions import NoBenchmarkError

if TYPE_CHECKING:
    from ..data import Data

# ── Periodic reporting mixin ──────────────────────────────────────────────────


class _PeriodicReportingMixin:
    """Mixin providing period-bucketed reporting tables.

    Covers: monthly-returns pivot table, distribution across calendar
    frequencies (daily…yearly), benchmark comparison table, and worst-N periods.
    """

    _data: Data
    all: pl.DataFrame

    if TYPE_CHECKING:
        from .._protocol import DataLike

        data: DataLike

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
            raise NoBenchmarkError

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
