"""Financial returns data container and manipulation utilities."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Literal, cast

import narwhals as nw
import polars as pl

from ._types import NativeFrame, NativeFrameOrScalar
from .exceptions import (
    BenchmarkAlignmentWarning,
    IntegerIndexBoundError,
    MissingDateColumnError,
    NullsInReturnsError,
)

if TYPE_CHECKING:
    from ._plots import DataPlots
    from ._reports import Reports
    from ._stats import Stats
    from ._utils import DataUtils


def _to_polars(df: NativeFrame) -> pl.DataFrame:
    """Convert any narwhals-compatible DataFrame to a polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df
    return nw.from_native(df, eager_only=True).to_polars()


def _apply_null_strategy(
    dframe: pl.DataFrame,
    date_col: str,
    frame_name: str,
    null_strategy: Literal["raise", "drop", "forward_fill"] | None,
) -> pl.DataFrame:
    """Check for nulls in *dframe* and apply *null_strategy*.

    Args:
        dframe (pl.DataFrame): DataFrame to inspect. The date column is
            excluded from the null scan.
        date_col (str): Name of the column to treat as the date index
            (excluded from null check).
        frame_name (str): Descriptive name used in the error message
            (e.g. ``"returns"``).
        null_strategy ({"raise", "drop", "forward_fill"} | None): How to
            handle null values:

            - ``None`` — leave nulls as-is (nulls will propagate through
              calculations).
            - ``"raise"`` — raise `NullsInReturnsError` if any null is found.
            - ``"drop"`` — drop every row that contains at least one null.
            - ``"forward_fill"`` — fill each null with the most recent
              non-null value in the same column.

    Returns:
        pl.DataFrame: The original DataFrame (``None`` / ``"raise"``), a
        filtered DataFrame (``"drop"``), or a filled DataFrame
        (``"forward_fill"``).

    Raises:
        NullsInReturnsError: When *null_strategy* is ``"raise"`` and nulls
            are present.

    """
    if null_strategy is None:
        return dframe

    value_cols = [c for c in dframe.columns if c != date_col]
    null_counts = dframe.select(value_cols).null_count().row(0)
    cols_with_nulls = [col for col, count in zip(value_cols, null_counts, strict=False) if count > 0]

    if not cols_with_nulls:
        return dframe

    if null_strategy == "raise":
        raise NullsInReturnsError(frame_name, cols_with_nulls)
    if null_strategy == "drop":
        return dframe.drop_nulls(subset=value_cols)
    # forward_fill
    return dframe.with_columns([pl.col(c).forward_fill() for c in value_cols])


def interpolate(df: pl.DataFrame) -> pl.DataFrame:
    """Forward-fill numeric columns only between first and last non-null values.

    For each numeric column, forward-fill is applied strictly within the span
    bounded by its first and last non-null samples. Values outside this span
    are left as-is (including leading/trailing nulls). Non-numeric columns are
    returned unchanged.

    Args:
        df: Input frame possibly containing nulls.

    Returns:
        pl.DataFrame: Frame where numeric columns have been interior-forward-
        filled; schema and dtypes of the original columns are preserved.

    Examples:
        ```python
        import polars as pl
        from jquantstats import interpolate

        df = pl.DataFrame({"a": [None, 1.0, None, 3.0, None], "b": ["x", "y", "z", "w", "v"]})
        result = interpolate(df)
        # a: [None, 1.0, 1.0, 3.0, None]  (leading/trailing nulls untouched)
        # b: ["x", "y", "z", "w", "v"]    (non-numeric unchanged)
        ```

    """
    # Choose a temp column name guaranteed not to collide with any user column.
    tmp_col = "__row_idx__"
    while tmp_col in df.columns:
        tmp_col = f"_{tmp_col}_"

    out = []

    for col in df.columns:
        s = df[col]
        if s.dtype.is_numeric():
            non_null_mask = s.is_not_null()
            if non_null_mask.any():
                _fwd = non_null_mask.arg_max()
                _rev = non_null_mask.reverse().arg_max()
                if _fwd is None or _rev is None:  # pragma: no cover
                    out.append(pl.col(col))
                    continue
                first_valid_idx = _fwd
                last_valid_idx = len(s) - 1 - _rev
            else:
                out.append(pl.col(col))
                continue

            mask = (pl.col(tmp_col) >= pl.lit(first_valid_idx)) & (pl.col(tmp_col) <= pl.lit(last_valid_idx))
            filled_col = pl.when(mask).then(pl.col(col).fill_null(strategy="forward")).otherwise(pl.col(col)).alias(col)
            out.append(filled_col)
        else:
            out.append(pl.col(col))

    return df.with_columns(pl.int_range(0, df.height).alias(tmp_col)).select(out)


def _subtract_risk_free(dframe: pl.DataFrame, rf: float | pl.DataFrame, date_col: str) -> pl.DataFrame:
    """Subtract the risk-free rate from all numeric columns in the DataFrame.

    Args:
        dframe (pl.DataFrame): DataFrame containing returns data with a date
            column and one or more numeric columns representing asset returns.
        rf (float | pl.DataFrame): Risk-free rate to subtract from returns.

            - If float: A constant risk-free rate applied to all dates.
            - If pl.DataFrame: A DataFrame with a date column and a second
              column containing time-varying risk-free rates.

        date_col (str): Name of the date column in both DataFrames for
            joining when rf is a DataFrame.

    Returns:
        pl.DataFrame: DataFrame with the risk-free rate subtracted from all
        numeric columns, preserving the original column names.

    """
    if isinstance(rf, float):
        rf_dframe = dframe.select([pl.col(date_col), pl.lit(rf).alias("rf")])
    else:
        if not isinstance(rf, pl.DataFrame):
            raise TypeError("rf must be a float or DataFrame")  # noqa: TRY003
        if rf.columns[1] != "rf":
            warnings.warn(
                f"Risk-free rate column '{rf.columns[1]}' has been renamed to 'rf' for internal alignment.",
                stacklevel=3,
            )
        rf_dframe = rf.rename({rf.columns[1]: "rf"}) if rf.columns[1] != "rf" else rf

    dframe = dframe.join(rf_dframe, on=date_col, how="inner")
    return dframe.select(
        [pl.col(date_col)]
        + [(pl.col(col) - pl.col("rf")).alias(col) for col in dframe.columns if col not in {date_col, "rf"}]
    )


def _require_date_col(frames: list[tuple[str, pl.DataFrame | None]], date_col: str) -> None:
    """Verify *date_col* is present in every supplied (non-None) frame.

    Args:
        frames: ``(name, frame)`` pairs; ``None`` frames are skipped.
        date_col: The required date column name.

    Raises:
        MissingDateColumnError: If any frame lacks *date_col*, naming that frame.
    """
    for frame_name, frame in frames:
        if frame is not None and date_col not in frame.columns:
            raise MissingDateColumnError(frame_name, column=date_col, available=list(frame.columns))


def _align_returns_benchmark(
    returns_pl: pl.DataFrame, benchmark_pl: pl.DataFrame, date_col: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Inner-join returns and benchmark on their common dates.

    Args:
        returns_pl: Returns frame with a *date_col* column.
        benchmark_pl: Benchmark frame with a *date_col* column.
        date_col: The shared date column name.

    Returns:
        The two frames filtered to their overlapping dates.

    Raises:
        ValueError: If the frames share no dates.

    Warns:
        BenchmarkAlignmentWarning: If aligning drops rows from either frame.
    """
    joined_dates = returns_pl.join(benchmark_pl, on=date_col, how="inner").select(date_col)
    if joined_dates.is_empty():
        raise ValueError("No overlapping dates between returns and benchmark.")  # noqa: TRY003
    dropped_returns = returns_pl.height - joined_dates.height
    dropped_benchmark = benchmark_pl.height - joined_dates.height
    if dropped_returns > 0 or dropped_benchmark > 0:
        warnings.warn(
            f"Aligning returns and benchmark on common dates dropped "
            f"{dropped_returns} of {returns_pl.height} returns row(s) and "
            f"{dropped_benchmark} of {benchmark_pl.height} benchmark row(s); "
            f"{joined_dates.height} row(s) remain. Pass a benchmark covering "
            f"the same dates as the returns to avoid this.",
            BenchmarkAlignmentWarning,
            stacklevel=2,
        )
    returns_pl = returns_pl.join(joined_dates, on=date_col, how="inner")
    benchmark_pl = benchmark_pl.join(joined_dates, on=date_col, how="inner")
    return returns_pl, benchmark_pl


@dataclasses.dataclass(frozen=True, slots=True)
class Data:
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
        prices_pl = _to_polars(prices)
        if date_col not in prices_pl.columns:
            raise MissingDateColumnError("prices", column=date_col, available=list(prices_pl.columns))
        asset_cols = [c for c in prices_pl.columns if c != date_col]
        returns_pl = prices_pl.with_columns([pl.col(c).pct_change().alias(c) for c in asset_cols]).slice(1)

        benchmark_returns: NativeFrame | None = None
        if benchmark is not None:
            benchmark_pl = _to_polars(benchmark)
            if date_col not in benchmark_pl.columns:
                raise MissingDateColumnError("benchmark", column=date_col, available=list(benchmark_pl.columns))
            bench_cols = [c for c in benchmark_pl.columns if c != date_col]
            benchmark_returns = benchmark_pl.with_columns([pl.col(c).pct_change().alias(c) for c in bench_cols]).slice(
                1
            )

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

    def resample(self, every: str = "1mo") -> Data:
        """Resample returns and benchmark to a different frequency.

        Args:
            every (str): Resampling frequency (e.g., ``'1mo'``, ``'1y'``).
                Defaults to ``'1mo'``.

        Returns:
            Data: Resampled data at the requested frequency.

        """

        def resample_frame(dframe: pl.DataFrame) -> pl.DataFrame:
            """Resample a single DataFrame to the target frequency using compound returns."""
            dframe = self.index.hstack(dframe)  # Add the date column for resampling

            return dframe.group_by_dynamic(
                index_column=self.index.columns[0], every=every, period=every, closed="right", label="right"
            ).agg(
                [
                    ((pl.col(col) + 1.0).product() - 1.0).alias(col)
                    for col in dframe.columns
                    if col != self.index.columns[0]
                ]
            )

        resampled_returns = resample_frame(self.returns)
        resampled_benchmark = resample_frame(self.benchmark) if self.benchmark is not None else None
        resampled_index = resampled_returns.select(self.index.columns[0])

        return Data(
            returns=resampled_returns.drop(self.index.columns[0]),
            benchmark=resampled_benchmark.drop(self.index.columns[0]) if resampled_benchmark is not None else None,
            index=resampled_index,
        )

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

    def copy(self) -> Data:
        """Create a deep copy of the Data object.

        Returns:
            Data: A new Data object with copies of the returns and benchmark.

        """
        if self.benchmark is not None:
            return Data(returns=self.returns.clone(), benchmark=self.benchmark.clone(), index=self.index.clone())
        return Data(returns=self.returns.clone(), index=self.index.clone())

    def head(self, n: int = 5) -> Data:
        """Return the first n rows of the combined returns and benchmark data.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            Data: A new Data object containing the first n rows of the combined data.

        """
        benchmark_head = self.benchmark.head(n) if self.benchmark is not None else None
        return Data(returns=self.returns.head(n), benchmark=benchmark_head, index=self.index.head(n))

    def tail(self, n: int = 5) -> Data:
        """Return the last n rows of the combined returns and benchmark data.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            Data: A new Data object containing the last n rows of the combined data.

        """
        benchmark_tail = self.benchmark.tail(n) if self.benchmark is not None else None
        return Data(returns=self.returns.tail(n), benchmark=benchmark_tail, index=self.index.tail(n))

    def truncate(
        self,
        start: date | datetime | str | int | None = None,
        end: date | datetime | str | int | None = None,
    ) -> Data:
        """Return a new Data object truncated to the inclusive [start, end] range.

        When the index is temporal (Date/Datetime), truncation is performed by
        comparing the date column against ``start`` and ``end`` values.

        When the index is integer-based, row slicing is used instead, and
        ``start`` and ``end`` must be non-negative integers.  Passing
        non-integer bounds to an integer-indexed Data raises `TypeError`.

        Args:
            start: Optional lower bound (inclusive).  A date/datetime value
                when the index is temporal; a non-negative `int` row
                index when the data has no temporal index.
            end: Optional upper bound (inclusive).  Same type rules as
                ``start``.

        Returns:
            Data: A new Data object filtered to the specified range.

        Raises:
            TypeError: When the index is not temporal and a non-integer bound
                is supplied.

        """
        date_column = self.index.columns[0]

        if self.index[date_column].dtype.is_temporal():
            new_index, new_returns, new_benchmark = self._truncate_temporal(date_column, start, end)
        else:
            new_index, new_returns, new_benchmark = self._truncate_integer(start, end)

        return Data(returns=new_returns, benchmark=new_benchmark, index=new_index)

    def _truncate_temporal(
        self,
        date_column: str,
        start: date | datetime | str | int | None,
        end: date | datetime | str | int | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
        """Truncate a temporal index by comparing the date column to [start, end]."""
        cond = pl.lit(True)
        if start is not None:
            cond = cond & (pl.col(date_column) >= pl.lit(start))
        if end is not None:
            cond = cond & (pl.col(date_column) <= pl.lit(end))
        mask = self.index.select(cond.alias("mask"))["mask"]
        new_benchmark = self.benchmark.filter(mask) if self.benchmark is not None else None
        return self.index.filter(mask), self.returns.filter(mask), new_benchmark

    def _truncate_integer(
        self,
        start: date | datetime | str | int | None,
        end: date | datetime | str | int | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
        """Truncate an integer index by row slicing; bounds must be integers."""
        if start is not None and not isinstance(start, int):
            raise IntegerIndexBoundError("start", type(start).__name__)
        if end is not None and not isinstance(end, int):
            raise IntegerIndexBoundError("end", type(end).__name__)
        row_start = start if start is not None else 0
        row_end = end + 1 if end is not None else self.index.height
        length = max(0, row_end - row_start)
        new_benchmark = self.benchmark.slice(row_start, length) if self.benchmark is not None else None
        return self.index.slice(row_start, length), self.returns.slice(row_start, length), new_benchmark

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
