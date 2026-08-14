"""Range/lag/smoothing transforms and correlation mixin for Portfolio.

`PortfolioTransformMixin` groups the methods that derive a *new* Portfolio
from an existing one (`truncate`, `lag`, `smoothed_holding`) plus the
`correlation` utility. New portfolios are built through
``type(self).from_cash_position`` so the transforms inherit the standard
construction path and return the concrete ``Self`` type.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Self, cast

import polars as pl
import polars.selectors as cs

from ._portfolio_base import _PortfolioMembers
from ._truncate import resolve_bounds


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

    # ── Shared construction helpers ────────────────────────────────────────────

    def _rebuild(self, cash_position: pl.DataFrame, prices: pl.DataFrame | None = None) -> Self:
        """Build a new Portfolio of the same concrete type from derived frames.

        Every transform in this mixin ends the same way: hand the derived
        frames back to ``from_cash_position`` while carrying ``aum`` and both
        cost parameters across unchanged. Centralising that here keeps a new
        construction parameter from having to be threaded through each
        transform individually.

        Args:
            cash_position: The derived cash-position frame.
            prices: The derived price frame; defaults to the current prices,
                which the lag and smoothing transforms leave untouched.

        Returns:
            A new Portfolio of the same concrete type.
        """
        return type(self).from_cash_position(
            prices=self.prices if prices is None else prices,
            cash_position=cash_position,
            aum=self.aum,
            cost_per_unit=self.cost_per_unit,
            cost_bps=self.cost_bps,
        )

    @property
    def _numeric_assets(self) -> list[str]:
        """Names of the numeric asset columns in the cash-position frame.

        Excludes ``'date'`` and any non-numeric column, so column-wise
        transforms touch only the asset series.

        Returns:
            The numeric asset column names, in frame order.
        """
        return [c for c in self.cashposition.columns if c != "date" and self.cashposition[c].dtype.is_numeric()]

    @staticmethod
    def _date_range_mask(
        start: date | datetime | str | int | None,
        end: date | datetime | str | int | None,
    ) -> pl.Expr:
        """Build the inclusive ``[start, end]`` filter over the ``'date'`` column.

        Args:
            start: Optional inclusive lower bound; no lower bound when None.
            end: Optional inclusive upper bound; no upper bound when None.

        Returns:
            A boolean Polars expression, ``lit(True)`` when both bounds are None.
        """
        cond = pl.lit(True)
        if start is not None:
            cond = cond & (pl.col("date") >= pl.lit(start))
        if end is not None:
            cond = cond & (pl.col("date") <= pl.lit(end))
        return cond

    @staticmethod
    def _row_slice_bounds(start: int | None, end: int | None, height: int) -> tuple[int, int]:
        """Resolve integer row bounds into an ``(offset, length)`` slice.

        Bounds arrive already validated by `resolve_bounds`, so this only
        substitutes the open-ended defaults.

        Args:
            start: Optional inclusive lower row index; 0 when None.
            end: Optional inclusive upper row index; the last row when None.
            height: Row count of the frame being sliced.

        Returns:
            The ``(offset, length)`` pair to pass to ``DataFrame.slice``.
            ``length`` is clamped at 0 so an inverted range yields an empty
            frame rather than a negative slice.
        """
        row_start = start if start is not None else 0
        row_end = end + 1 if end is not None else height
        return row_start, max(0, row_end - row_start)

    # ── Transforms ─────────────────────────────────────────────────────────────

    def truncate(
        self,
        start: date | datetime | str | int | None = None,
        end: date | datetime | str | int | None = None,
    ) -> Self:
        """Return a new Portfolio truncated to the inclusive [start, end] range.

        **The bound type picks the axis.**  A ``date``, ``datetime`` or ISO-8601
        string is compared against the ``'date'`` column; an ``int`` is a 0-based
        row index and slices positionally.  Row indices work on a dated portfolio
        too — ``truncate(start=10)`` drops the first ten rows — but the two kinds
        cannot be combined in one call.

        A portfolio with no ``'date'`` column accepts row indices only.

        In all cases the ``aum`` value is preserved.

        Args:
            start: Optional inclusive lower bound.  A ``date``/``datetime``, an
                ISO-8601 string, or an ``int`` row index; ``int`` only when there
                is no ``'date'`` column.
            end: Optional inclusive upper bound.  Same type rules as ``start``,
                and must address the same axis.

        Returns:
            A new Portfolio instance with prices and cash positions filtered
            to the specified range.

        Raises:
            IntegerIndexBoundError: When the portfolio has no ``'date'`` column
                and a bound is not an ``int``.
            InvalidTruncateBoundError: When a bound is of an unsupported type,
                or is a string that is not ISO-8601.
            MixedTruncateBoundsError: When one bound is a row index and the
                other a date.
        """
        mode, lower, upper = resolve_bounds(start, end, temporal="date" in self.prices.columns)

        if mode == "dates":
            cond = self._date_range_mask(lower, upper)
            pr = self.prices.filter(cond)
            cp = self.cashposition.filter(cond)
        else:
            # "none" resolves to a full-width slice, so it needs no separate branch.
            # The casts record what resolve_bounds guarantees for these modes but
            # cannot express in its return type.
            offset, length = self._row_slice_bounds(
                cast("int | None", lower), cast("int | None", upper), self.prices.height
            )
            pr = self.prices.slice(offset, length)
            cp = self.cashposition.slice(offset, length)
        return self._rebuild(prices=pr, cash_position=cp)

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

        cp_lagged = self.cashposition.with_columns(pl.col(c).shift(n) for c in self._numeric_assets)
        return self._rebuild(cash_position=cp_lagged)

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

        window = n + 1
        cp_smoothed = self.cashposition.with_columns(
            pl.col(c).rolling_mean(window_size=window, min_samples=1).alias(c) for c in self._numeric_assets
        )
        return self._rebuild(cash_position=cp_smoothed)

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
