"""Reshaping operations for `Data`: resampling, copying, slicing, truncation.

`_ReshapeMixin` collects the methods that return a *new* `Data` derived from an
existing one. They are factored out of ``data.py`` to keep that module focused
on construction and accessors; the mixin only reads the three dataclass fields
(``returns``, ``index``, ``benchmark``) and rebuilds via `_rebuild`, which
constructs through ``type(self)`` so `Data` never enters this module's import
graph — not even lazily.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, cast

import polars as pl

from ._truncate import resolve_bounds

if TYPE_CHECKING:
    from collections.abc import Callable

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

        Constructs via ``type(self)`` rather than importing `Data`. This mixin is
        only ever mixed into `Data`, so ``type(self)`` *is* the concrete class at
        runtime — which keeps `Data` out of this module's import graph entirely
        (a lazy import still puts it there) and rebuilds a subclass as its own
        type rather than downcasting it to `Data`.

        Args:
            returns: Returns frame for the new object.
            index: Date/row index frame for the new object.
            benchmark: Optional benchmark frame for the new object.

        Returns:
            Data: A new `Data` built from the supplied frames.
        """
        factory = cast("Callable[..., Data]", type(self))
        return factory(returns=returns, index=index, benchmark=benchmark)

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

        **The bound type picks the axis.**  A ``date``, ``datetime`` or ISO-8601
        string is compared against the date column; an ``int`` is a 0-based row
        index and slices positionally.  Row indices work on a temporal index too
        — ``truncate(start=10)`` drops the first ten rows of dated data — but the
        two kinds cannot be combined in one call.

        An integer-indexed Data (no temporal index) accepts row indices only.

        Args:
            start: Optional inclusive lower bound.  A ``date``/``datetime``, an
                ISO-8601 string, or an ``int`` row index; ``int`` only when the
                index is not temporal.
            end: Optional inclusive upper bound.  Same type rules as ``start``,
                and must address the same axis.

        Returns:
            Data: A new Data object filtered to the specified range.

        Raises:
            IntegerIndexBoundError: When the index is not temporal and a bound
                is not an ``int``.
            InvalidTruncateBoundError: When a bound is of an unsupported type,
                or is a string that is not ISO-8601.
            MixedTruncateBoundsError: When one bound is a row index and the
                other a date.
        """
        date_column = self.index.columns[0]
        mode, lower, upper = resolve_bounds(start, end, temporal=self.index[date_column].dtype.is_temporal())

        if mode == "dates":
            new_index, new_returns, new_benchmark = self._truncate_temporal(date_column, lower, upper)
        else:
            # "none" resolves to a full-width slice, so it needs no separate branch.
            # The casts record what resolve_bounds guarantees for these modes but
            # cannot express in its return type.
            new_index, new_returns, new_benchmark = self._truncate_integer(
                cast("int | None", lower), cast("int | None", upper)
            )

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

    def _truncate_integer(
        self,
        start: int | None,
        end: int | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
        """Truncate by 0-based row slicing.

        Bounds arrive already validated by `resolve_bounds`, so this only
        substitutes the open-ended defaults.  ``length`` is clamped at 0 so an
        inverted range yields an empty frame rather than a negative slice.
        """
        row_start = start if start is not None else 0
        row_end = (end if end is not None else self.index.height - 1) + 1
        length = max(0, row_end - row_start)
        new_benchmark = self.benchmark.slice(row_start, length) if self.benchmark is not None else None
        return self.index.slice(row_start, length), self.returns.slice(row_start, length), new_benchmark
