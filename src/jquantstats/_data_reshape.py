"""Reshaping operations for `Data`: resampling, copying, slicing, truncation.

`_ReshapeMixin` collects the methods that return a *new* `Data` derived from an
existing one. They are factored out of ``data.py`` to keep that module focused
on construction and accessors; the mixin only reads the three dataclass fields
(``returns``, ``index``, ``benchmark``) and rebuilds via `_rebuild`, which
imports `Data` lazily to avoid re-forming the ``data`` ↔ mixin import cycle.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import polars as pl

from .exceptions import IntegerIndexBoundError

if TYPE_CHECKING:
    from .data import Data


class _ReshapeMixin:
    """Mixin providing the `Data` operations that yield a new `Data`.

    The concrete class (`Data`) supplies the ``returns``, ``index`` and
    ``benchmark`` dataclass fields; they are annotated here so the mixin's
    methods type-check without importing `Data` at module load (which would
    re-form an import cycle). No runtime attributes are created — the mixin
    carries empty slots.
    """

    __slots__ = ()

    # Provided by the concrete Data dataclass; declared for type-checkers only.
    returns: pl.DataFrame
    index: pl.DataFrame
    benchmark: pl.DataFrame | None

    def _rebuild(
        self,
        *,
        returns: pl.DataFrame,
        index: pl.DataFrame,
        benchmark: pl.DataFrame | None = None,
    ) -> Data:
        """Build a fresh `Data` from the given frames.

        `Data` is imported lazily so this module stays importable from
        ``data.py`` without a cycle.

        Args:
            returns: Returns frame for the new object.
            index: Date/row index frame for the new object.
            benchmark: Optional benchmark frame for the new object.

        Returns:
            Data: A new `Data` built from the supplied frames.
        """
        from .data import Data

        return Data(returns=returns, index=index, benchmark=benchmark)

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

        return self._rebuild(
            returns=resampled_returns.drop(self.index.columns[0]),
            benchmark=resampled_benchmark.drop(self.index.columns[0]) if resampled_benchmark is not None else None,
            index=resampled_index,
        )

    def copy(self) -> Data:
        """Create a deep copy of the Data object.

        Returns:
            Data: A new Data object with copies of the returns and benchmark.

        """
        benchmark = self.benchmark.clone() if self.benchmark is not None else None
        return self._rebuild(returns=self.returns.clone(), benchmark=benchmark, index=self.index.clone())

    def head(self, n: int = 5) -> Data:
        """Return the first n rows of the combined returns and benchmark data.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            Data: A new Data object containing the first n rows of the combined data.

        """
        benchmark_head = self.benchmark.head(n) if self.benchmark is not None else None
        return self._rebuild(returns=self.returns.head(n), benchmark=benchmark_head, index=self.index.head(n))

    def tail(self, n: int = 5) -> Data:
        """Return the last n rows of the combined returns and benchmark data.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            Data: A new Data object containing the last n rows of the combined data.

        """
        benchmark_tail = self.benchmark.tail(n) if self.benchmark is not None else None
        return self._rebuild(returns=self.returns.tail(n), benchmark=benchmark_tail, index=self.index.tail(n))

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

        return self._rebuild(returns=new_returns, benchmark=new_benchmark, index=new_index)

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

    @staticmethod
    def _resolve_row_bound(name: str, value: date | datetime | str | int | None, default: int) -> int:
        """Validate and resolve an integer truncation bound to a row index.

        Args:
            name: The bound's name (``"start"`` or ``"end"``) for the message.
            value: The supplied bound; ``None`` and ``int`` are accepted.
            default: The row index to use when *value* is ``None``.

        Returns:
            int: *value* when it is an ``int``, otherwise *default*.

        Raises:
            IntegerIndexBoundError: If *value* is neither ``None`` nor ``int``.
        """
        if value is not None and not isinstance(value, int):
            raise IntegerIndexBoundError(name, type(value).__name__)
        return value if value is not None else default

    def _truncate_integer(
        self,
        start: date | datetime | str | int | None,
        end: date | datetime | str | int | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
        """Truncate an integer index by row slicing; bounds must be integers."""
        row_start = self._resolve_row_bound("start", start, 0)
        row_end = self._resolve_row_bound("end", end, self.index.height - 1) + 1
        length = max(0, row_end - row_start)
        new_benchmark = self.benchmark.slice(row_start, length) if self.benchmark is not None else None
        return self.index.slice(row_start, length), self.returns.slice(row_start, length), new_benchmark
