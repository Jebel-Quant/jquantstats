"""Drawdown and cumulative-return metrics for financial returns data."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl

from ._core import columnwise_stat, to_frame
from ._internals import _nav_series

if TYPE_CHECKING:
    from ..data import Data

# ── Drawdown statistics mixin ─────────────────────────────────────────────────


class _DrawdownMixin:
    """Mixin providing cumulative-return and drawdown metrics.

    Covers: compounded cumulative returns (``compsum``), the drawdown series,
    price (NAV) conversion, maximum drawdown, and per-episode drawdown details.
    """

    _data: Data
    all: pl.DataFrame

    if TYPE_CHECKING:
        from .._protocol import DataLike

        data: DataLike

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
        price = _DrawdownMixin.prices(series)
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
        return _DrawdownMixin.max_drawdown_single_series(series)

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
        all_df = self.all
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
