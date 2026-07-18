"""Financial returns data container and manipulation utilities."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from datetime import timedelta
from typing import TYPE_CHECKING, Literal, cast

import polars as pl

from ._data_reshape import _ReshapeMixin
from ._types import NativeFrame, NativeFrameOrScalar
from ._utils._construction import (
    _align_returns_benchmark,
    _apply_null_strategy,
    _prices_to_returns,
    _require_date_col,
    _subtract_risk_free,
    _to_polars,
)
from ._utils._construction import (
    interpolate as interpolate,  # re-exported for `from jquantstats import interpolate`
)

if TYPE_CHECKING:
    from ._plots import DataPlots
    from ._reports import Reports
    from ._stats import Stats
    from ._utils import DataUtils


@dataclasses.dataclass(frozen=True, slots=True)
class Data(_ReshapeMixin):
    """A container for financial returns data and an optional benchmark.

    Provides methods for analyzing and manipulating financial returns data,
    including resampling, truncation, and access to statistical metrics and
    visualizations via the ``stats`` and ``plots`` properties.

    Attributes:
        returns (pl.DataFrame): DataFrame containing returns data with assets
            as columns.
        benchmark (pl.DataFrame | None): Optional benchmark returns DataFrame.
            Defaults to None.
        index (pl.DataFrame): DataFrame containing the date index for the
            returns data.

    """

    returns: pl.DataFrame
    index: pl.DataFrame
    benchmark: pl.DataFrame | None = None

    def __post_init__(self) -> None:
        """Validate the Data object after initialization."""
        # You need at least two points
        if self.index.shape[0] < 2:
            raise ValueError("Index must contain at least two timestamps.")  # noqa: TRY003

        # Check index is monotonically increasing
        datetime_col = self.index[self.index.columns[0]]
        if not datetime_col.is_sorted():
            raise ValueError("Index must be monotonically increasing.")  # noqa: TRY003

        # Check row count matches returns
        if self.returns.shape[0] != self.index.shape[0]:
            raise ValueError("Returns and index must have the same number of rows.")  # noqa: TRY003

        # Check row count matches benchmark (if provided)
        if self.benchmark is not None and self.benchmark.shape[0] != self.index.shape[0]:
            raise ValueError("Benchmark and index must have the same number of rows.")  # noqa: TRY003

    @classmethod
    def from_returns(
        cls,
        returns: NativeFrame,
        rf: NativeFrameOrScalar = 0.0,
        benchmark: NativeFrame | None = None,
        date_col: str = "Date",
        null_strategy: Literal["raise", "drop", "forward_fill"] | None = None,
    ) -> Data:
        """Create a Data object from returns and optional benchmark.

        Args:
            returns (NativeFrame): Financial returns data. First column should
                be the date column, remaining columns are asset returns.
            rf (float | NativeFrame): Risk-free rate. Defaults to 0.0 (no
                risk-free rate adjustment).

                - If float: Constant risk-free rate applied to all dates.
                - If NativeFrame: Time-varying risk-free rate with dates
                  matching returns.

            benchmark (NativeFrame | None): Benchmark returns. Defaults to
                None (no benchmark). First column should be the date column,
                remaining columns are benchmark returns. Returns and
                benchmark are aligned on their common dates; if either frame
                contains dates the other lacks, those rows are dropped and a
                `BenchmarkAlignmentWarning` is emitted.
            date_col (str): Name of the date column in the DataFrames.
                Defaults to ``"Date"``.
            null_strategy ({"raise", "drop", "forward_fill"} | None): How to
                handle ``null`` (missing) values in *returns* and *benchmark*.
                Defaults to ``None`` (nulls propagate through calculations).

                - ``None`` — no null checking; nulls propagate through all
                  downstream calculations.
                - ``"raise"`` — raise `NullsInReturnsError` if any null is
                  found.
                - ``"drop"`` — silently drop every row that contains at least
                  one null.
                - ``"forward_fill"`` — fill each null with the most recent
                  non-null value in the same column.

                Note: Affects only Polars ``null`` values (i.e. ``None`` /
                missing entries). IEEE-754 ``NaN`` values are **not** affected
                and continue to propagate as per IEEE-754 semantics.

        Returns:
            Data: Object containing excess returns and benchmark (if any),
            with methods for analysis and visualization through the ``stats``
            and ``plots`` properties.

        Raises:
            MissingDateColumnError: If *date_col* is not a column of
                *returns*, *benchmark*, or a DataFrame-valued *rf*. Raised
                before any joins so the offending frame is named explicitly.
            NullsInReturnsError: If *null_strategy* is ``"raise"`` and the
                data contains null values.
            ValueError: If there are no overlapping dates between returns and
                benchmark.

        Warns:
            BenchmarkAlignmentWarning: If aligning returns and benchmark on
                their common dates drops rows from either frame.

        Examples:
            Basic usage:

            ```python
            from jquantstats import Data
            import polars as pl

            returns = pl.DataFrame({
                "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Asset1": [0.01, -0.02, 0.03]
            }).with_columns(pl.col("Date").str.to_date())

            data = Data.from_returns(returns=returns)
            ```

            With benchmark and risk-free rate:

            ```python
            benchmark = pl.DataFrame({
                "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Market": [0.005, -0.01, 0.02]
            }).with_columns(pl.col("Date").str.to_date())

            data = Data.from_returns(returns=returns, benchmark=benchmark, rf=0.0002)
            ```

            Handling nulls automatically:

            ```python
            returns_with_nulls = pl.DataFrame({
                "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Asset1": [0.01, None, 0.03]
            }).with_columns(pl.col("Date").str.to_date())

            # Drop rows with nulls (mirrors pandas/QuantStats behaviour)
            data = Data.from_returns(returns=returns_with_nulls, null_strategy="drop")

            # Or forward-fill nulls
            data = Data.from_returns(returns=returns_with_nulls, null_strategy="forward_fill")
            ```

        """
        returns_pl = _to_polars(returns)
        benchmark_pl = _to_polars(benchmark) if benchmark is not None else None
        # accept ints (e.g. rf=0) by coercing to float
        rf_converted: float | pl.DataFrame = float(rf) if isinstance(rf, int | float) else _to_polars(rf)

        frames: list[tuple[str, pl.DataFrame | None]] = [("returns", returns_pl), ("benchmark", benchmark_pl)]
        if isinstance(rf_converted, pl.DataFrame):
            frames.append(("rf", rf_converted))
        _require_date_col(frames, date_col)

        returns_pl = _apply_null_strategy(returns_pl, date_col, "returns", null_strategy)
        if benchmark_pl is not None:
            benchmark_pl = _apply_null_strategy(benchmark_pl, date_col, "benchmark", null_strategy)
            returns_pl, benchmark_pl = _align_returns_benchmark(returns_pl, benchmark_pl, date_col)

        index = returns_pl.select(date_col)
        excess_returns = _subtract_risk_free(returns_pl, rf_converted, date_col).drop(date_col)
        excess_benchmark = (
            _subtract_risk_free(benchmark_pl, rf_converted, date_col).drop(date_col)
            if benchmark_pl is not None
            else None
        )

        return cls(returns=excess_returns, benchmark=excess_benchmark, index=index)

    @classmethod
    def from_prices(
        cls,
        prices: NativeFrame,
        rf: NativeFrameOrScalar = 0.0,
        benchmark: NativeFrame | None = None,
        date_col: str = "Date",
        null_strategy: Literal["raise", "drop", "forward_fill"] | None = None,
    ) -> Data:
        """Create a Data object from prices and optional benchmark.

        Converts price levels to returns via percentage change and delegates
        to `from_returns`. The first row of each asset is dropped because no
        prior price is available to compute a return.

        Args:
            prices (NativeFrame): Price-level data. First column should be
                the date column; remaining columns are asset prices.
            rf (float | NativeFrame): Risk-free rate. Forwarded unchanged to
                `from_returns`. Defaults to 0.0 (no risk-free rate
                adjustment).
            benchmark (NativeFrame | None): Benchmark prices. Converted to
                returns in the same way as ``prices`` before being forwarded
                to `from_returns`. Defaults to None (no benchmark).
            date_col (str): Name of the date column in the DataFrames.
                Defaults to ``"Date"``.
            null_strategy ({"raise", "drop", "forward_fill"} | None): How to
                handle ``null`` (missing) values after converting prices to
                returns. Forwarded unchanged to `from_returns`. Defaults to
                ``None`` (nulls propagate through calculations).

                - ``None`` — no null checking; nulls propagate.
                - ``"raise"`` — raise `NullsInReturnsError` if any null is
                  found in the derived returns.
                - ``"drop"`` — silently drop every row that contains at least
                  one null.
                - ``"forward_fill"`` — fill each null with the most recent
                  non-null value.

                Note: Prices that contain nulls will produce null returns via
                ``pct_change()``. If you expect missing price entries, pass
                ``null_strategy="drop"`` or ``null_strategy="forward_fill"``.

        Returns:
            Data: Object containing excess returns derived from the supplied
            prices, with methods for analysis and visualization through the
            ``stats`` and ``plots`` properties.

        Raises:
            MissingDateColumnError: If *date_col* is not a column of *prices*
                or *benchmark*. Raised before returns are derived so the
                offending frame is named explicitly.

        Examples:
            ```python
            from jquantstats import Data
            import polars as pl

            prices = pl.DataFrame({
                "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Asset1": [100.0, 101.0, 99.0]
            }).with_columns(pl.col("Date").str.to_date())

            data = Data.from_prices(prices=prices)
            ```

        """
        returns_pl = _prices_to_returns(_to_polars(prices), date_col, "prices")

        benchmark_returns: NativeFrame | None = None
        if benchmark is not None:
            benchmark_returns = _prices_to_returns(_to_polars(benchmark), date_col, "benchmark")

        return cls.from_returns(
            returns=returns_pl,
            rf=rf,
            benchmark=benchmark_returns,
            date_col=date_col,
            null_strategy=null_strategy,
        )

    def __repr__(self) -> str:
        """Return a string representation of the Data object."""
        rows = len(self.index)
        date_cols = self.date_col
        if date_cols:
            date_column = date_cols[0]
            start = self.index[date_column].min()
            end = self.index[date_column].max()
            return f"Data(assets={self.assets}, rows={rows}, start={start!s}, end={end!s})"
        return f"Data(assets={self.assets}, rows={rows})"  # pragma: no cover  # __post_init__ requires ≥1 index column

    @property
    def plots(self) -> DataPlots:
        """Provides access to visualization methods for the financial data.

        Returns:
            DataPlots: An instance of the DataPlots class initialized with this data.

        """
        # Deferred to break the data <-> accessors import cycle: _plots imports
        # DataLike from this module, so hoisting this to module top re-forms the
        # cycle. Keep the import local.
        from ._plots import DataPlots

        return DataPlots(self)

    @property
    def stats(self) -> Stats:
        """Provides access to statistical analysis methods for the financial data.

        Returns:
            Stats: An instance of the Stats class initialized with this data.

        """
        # Deferred to break the data <-> accessors import cycle (see .plots).
        from ._stats import Stats

        return Stats(self)

    @property
    def reports(self) -> Reports:
        """Provides access to reporting methods for the financial data.

        Returns:
            Reports: An instance of the Reports class initialized with this data.

        """
        # Deferred to break the data <-> accessors import cycle (see .plots).
        from ._reports import Reports

        return Reports(self)

    @property
    def utils(self) -> DataUtils:
        """Provides access to utility transforms and conversions for the financial data.

        Returns:
            DataUtils: An instance of the DataUtils class initialized with this data.

        """
        # Deferred to break the data <-> accessors import cycle (see .plots).
        from ._utils import DataUtils

        return DataUtils(self)

    @property
    def date_col(self) -> list[str]:
        """Return the column names of the index DataFrame.

        Returns:
            list[str]: List of column names in the index DataFrame, typically containing
                      the date column name.

        """
        return list(self.index.columns)

    @property
    def assets(self) -> list[str]:
        """Return the combined list of asset column names from returns and benchmark.

        Returns:
            list[str]: List of all asset column names from both returns and benchmark
                      (if available).

        """
        if self.benchmark is not None:
            return list(self.returns.columns) + list(self.benchmark.columns)
        return list(self.returns.columns)

    @property
    def all(self) -> pl.DataFrame:
        """Combine index, returns, and benchmark data into a single DataFrame.

        This property provides a convenient way to access all data in a single DataFrame,
        which is useful for analysis and visualization.

        Returns:
            pl.DataFrame: A DataFrame containing the index, all returns data, and benchmark data
                         (if available) combined horizontally.

        """
        if self.benchmark is None:
            return pl.concat([self.index, self.returns], how="horizontal_extend")
        else:
            return pl.concat([self.index, self.returns, self.benchmark], how="horizontal_extend")

    def describe(self) -> pl.DataFrame:
        """Return a tidy summary of shape, date range and asset names.

        Returns:
            pl.DataFrame: One row per asset with columns: asset, start, end,
            rows, has_benchmark.

        """
        date_column = self.date_col[0]
        start = self.index[date_column].min()
        end = self.index[date_column].max()
        rows = len(self.index)
        return pl.DataFrame(
            {
                "asset": self.returns.columns,
                "start": [start] * len(self.returns.columns),
                "end": [end] * len(self.returns.columns),
                "rows": [rows] * len(self.returns.columns),
                "has_benchmark": [self.benchmark is not None] * len(self.returns.columns),
            }
        )

    @property
    def _periods_per_year(self) -> float:
        """Estimate the number of periods per year based on average frequency in the index.

        For temporal (Date/Datetime) indices, computes the mean gap between observations
        and converts to an annualised period count (e.g. ~252 for daily, ~52 for weekly).

        For integer indices (date-free portfolios), falls back to 252 trading days per year
        because integer diffs have no time meaning.
        """
        datetime_col = self.index[self.index.columns[0]]

        if not datetime_col.dtype.is_temporal():
            return 252.0

        sorted_dt = datetime_col.sort()
        diffs = sorted_dt.diff().drop_nulls()
        mean_diff = diffs.mean()

        if isinstance(mean_diff, timedelta):
            seconds = mean_diff.total_seconds()
        else:  # pragma: no cover  # Polars always returns timedelta for temporal diff
            seconds = cast(float, mean_diff) if mean_diff is not None else 1.0

        return (365 * 24 * 60 * 60) / seconds

    def items(self) -> Iterator[tuple[str, pl.Series]]:
        """Iterate over all assets and their corresponding data series.

        This method provides a convenient way to iterate over all assets in the data,
        yielding each asset name and its corresponding data series.

        Yields:
            tuple[str, pl.Series]: A tuple containing the asset name and its data series.

        """
        matrix = self.all

        for col in self.assets:
            yield col, matrix.get_column(col)
