"""Frame-construction and interpolation helpers for building `Data` objects.

These free functions carry the returns/prices ingestion logic that used to live
inline in ``data.py``: native-frame coercion, null handling, interior
interpolation, risk-free subtraction, date-column validation, and
returns/benchmark alignment. Keeping them here shrinks ``data.py`` and isolates
the construction concern from the `Data` container itself.
"""

from __future__ import annotations

import warnings
from typing import Literal

import narwhals as nw
import polars as pl

from .._types import NativeFrame
from ..exceptions import (
    BenchmarkAlignmentWarning,
    MissingDateColumnError,
    NullsInReturnsError,
)

__all__ = [
    "interpolate",
]


def _to_polars(df: NativeFrame) -> pl.DataFrame:
    """Convert any narwhals-compatible DataFrame to a polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df
    return nw.from_native(df, eager_only=True).to_polars()


def _value_columns(dframe: pl.DataFrame, date_col: str) -> list[str]:
    """Return every column of *dframe* except the date column."""
    return [c for c in dframe.columns if c != date_col]


def _columns_with_nulls(dframe: pl.DataFrame, value_cols: list[str]) -> list[str]:
    """Return the subset of *value_cols* that contain at least one null."""
    null_counts = dframe.select(value_cols).null_count().row(0)
    return [col for col, count in zip(value_cols, null_counts, strict=False) if count > 0]


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

    value_cols = _value_columns(dframe, date_col)
    cols_with_nulls = _columns_with_nulls(dframe, value_cols)

    if not cols_with_nulls:
        return dframe

    if null_strategy == "raise":
        raise NullsInReturnsError(frame_name, cols_with_nulls)
    if null_strategy == "drop":
        return dframe.drop_nulls(subset=value_cols)
    # forward_fill
    return dframe.with_columns(pl.col(value_cols).forward_fill())


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


def _prices_to_returns(frame: pl.DataFrame, date_col: str, frame_name: str) -> pl.DataFrame:
    """Convert a price-level frame to a returns frame via percentage change.

    The first row is dropped because no prior price is available to compute a
    return for it.

    Args:
        frame: Price-level frame with a *date_col* column and asset columns.
        date_col: Name of the date column (passed through unchanged).
        frame_name: Descriptive name used in the error message when *date_col*
            is missing (e.g. ``"prices"`` or ``"benchmark"``).

    Returns:
        pl.DataFrame: Returns frame with the same columns as *frame*, one row
        shorter.

    Raises:
        MissingDateColumnError: If *date_col* is not a column of *frame*.
    """
    if date_col not in frame.columns:
        raise MissingDateColumnError(frame_name, column=date_col, available=list(frame.columns))
    asset_cols = _value_columns(frame, date_col)
    return frame.with_columns([pl.col(c).pct_change().alias(c) for c in asset_cols]).slice(1)
