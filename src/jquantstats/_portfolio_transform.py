"""Range/lag/smoothing transforms and correlation mixin for Portfolio.

`PortfolioTransformMixin` groups the methods that derive a *new* Portfolio
from an existing one (`truncate`, `lag`, `smoothed_holding`) plus the
`correlation` utility. New portfolios are built through
``type(self).from_cash_position`` so the transforms inherit the standard
construction path and return the concrete ``Self`` type.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Self

import polars as pl
import polars.selectors as cs

from ._portfolio_base import _PortfolioMembers
from .exceptions import IntegerIndexBoundError


class PortfolioTransformMixin(_PortfolioMembers):
    """Mixin providing range/lag/smoothing transforms and correlation for Portfolio."""

    if TYPE_CHECKING:

        @classmethod
        def from_cash_position(
            cls,
            prices: pl.DataFrame,
            cash_position: pl.DataFrame,
            aum: float,
            cost_per_unit: float = 0.0,
            cost_bps: float = 0.0,
        ) -> Self:
            """Create a Portfolio directly from cash positions aligned with prices."""
            ...

    def truncate(
        self,
        start: date | datetime | str | int | None = None,
        end: date | datetime | str | int | None = None,
    ) -> Self:
        """Return a new Portfolio truncated to the inclusive [start, end] range.

        When a ``'date'`` column is present in both prices and cash positions,
        truncation is performed by comparing the ``'date'`` column against
        ``start`` and ``end`` (which should be date/datetime values or strings
        parseable by Polars).

        When the ``'date'`` column is absent, integer-based row slicing is
        used instead.  In this case ``start`` and ``end`` must be non-negative
        integers representing 0-based row indices.  Passing non-integer bounds
        to an integer-indexed portfolio raises `TypeError`.

        In all cases the ``aum`` value is preserved.

        Args:
            start: Optional lower bound (inclusive). A date/datetime or
                Polars-parseable string when a ``'date'`` column exists; a
                non-negative int row index when the data has no ``'date'``
                column.
            end: Optional upper bound (inclusive). Same type rules as
                ``start``.

        Returns:
            A new Portfolio instance with prices and cash positions filtered
            to the specified range.

        Raises:
            TypeError: When the portfolio has no ``'date'`` column and a
                non-integer bound is supplied.
        """
        has_date = "date" in self.prices.columns
        if has_date:
            cond = pl.lit(True)
            if start is not None:
                cond = cond & (pl.col("date") >= pl.lit(start))
            if end is not None:
                cond = cond & (pl.col("date") <= pl.lit(end))
            pr = self.prices.filter(cond)
            cp = self.cashposition.filter(cond)
        else:
            if start is not None and not isinstance(start, int):
                raise IntegerIndexBoundError("start", type(start).__name__)
            if end is not None and not isinstance(end, int):
                raise IntegerIndexBoundError("end", type(end).__name__)
            row_start = int(start) if start is not None else 0
            row_end = int(end) + 1 if end is not None else self.prices.height
            length = max(0, row_end - row_start)
            pr = self.prices.slice(row_start, length)
            cp = self.cashposition.slice(row_start, length)
        return type(self).from_cash_position(
            prices=pr,
            cash_position=cp,
            aum=self.aum,
            cost_per_unit=self.cost_per_unit,
            cost_bps=self.cost_bps,
        )

    def lag(self, n: int) -> Self:
        """Return a new Portfolio with cash positions lagged by ``n`` steps.

        This method shifts the numeric asset columns in the cashposition
        DataFrame by ``n`` rows, preserving the ``'date'`` column and any
        non-numeric columns unchanged.  Positive ``n`` delays weights (moves
        them down); negative ``n`` leads them (moves them up); ``n == 0``
        returns the current portfolio unchanged.

        Notes:
            Missing values introduced by the shift are left as nulls;
            downstream profit computation already guards and treats nulls as
            zero when multiplying by returns.

        Args:
            n: Number of rows to shift (can be negative, zero, or positive).

        Returns:
            A new Portfolio instance with lagged cash positions and the same
            prices/AUM as the original.
        """
        if not isinstance(n, int):
            raise TypeError
        if n == 0:
            return self

        assets = [c for c in self.cashposition.columns if c != "date" and self.cashposition[c].dtype.is_numeric()]
        cp_lagged = self.cashposition.with_columns(pl.col(c).shift(n) for c in assets)
        return type(self).from_cash_position(
            prices=self.prices,
            cash_position=cp_lagged,
            aum=self.aum,
            cost_per_unit=self.cost_per_unit,
            cost_bps=self.cost_bps,
        )

    def smoothed_holding(self, n: int) -> Self:
        """Return a new Portfolio with cash positions smoothed by a rolling mean.

        Applies a trailing window average over the last ``n`` steps for each
        numeric asset column (excluding ``'date'``). The window length is
        ``n + 1`` so that:

        - n=0 returns the original weights (no smoothing),
        - n=1 averages the current and previous weights,
        - n=k averages the current and last k weights.

        Args:
            n: Non-negative integer specifying how many previous steps to
                include.

        Returns:
            A new Portfolio with smoothed cash positions and the same
            prices/AUM.
        """
        if not isinstance(n, int):
            raise TypeError(f"n must be an integer, got {type(n).__name__}")  # noqa: TRY003
        if n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n}")  # noqa: TRY003
        if n == 0:
            return self

        assets = [c for c in self.cashposition.columns if c != "date" and self.cashposition[c].dtype.is_numeric()]
        window = n + 1
        cp_smoothed = self.cashposition.with_columns(
            pl.col(c).rolling_mean(window_size=window, min_samples=1).alias(c) for c in assets
        )
        return type(self).from_cash_position(
            prices=self.prices,
            cash_position=cp_smoothed,
            aum=self.aum,
            cost_per_unit=self.cost_per_unit,
            cost_bps=self.cost_bps,
        )

    # ── Utility ────────────────────────────────────────────────────────────────

    def correlation(self, frame: pl.DataFrame, name: str = "portfolio") -> pl.DataFrame:
        """Compute a correlation matrix of asset returns plus the portfolio.

        Computes percentage changes for all numeric columns in ``frame``,
        appends the portfolio profit series under the provided ``name``, and
        returns the Pearson correlation matrix across all numeric columns.

        Args:
            frame: A Polars DataFrame containing at least the asset price
                columns (and a date column which will be ignored if
                non-numeric).
            name: The column name to use when adding the portfolio profit
                series to the input frame.

        Returns:
            A square Polars DataFrame where each cell is the correlation
            between a pair of series (values in [-1, 1]).
        """
        p = frame.with_columns(cs.by_dtype(pl.Float32, pl.Float64).pct_change())
        p = p.with_columns(pl.Series(name, self.profit["profit"]))
        corr_matrix = p.select(cs.numeric()).fill_null(0.0).corr()
        return corr_matrix
