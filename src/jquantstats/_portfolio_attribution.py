"""Tilt/timing attribution mixin for Portfolio."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import polars as pl

if TYPE_CHECKING:
    pass


class PortfolioAttributionMixin:
    """Mixin providing tilt/timing attribution properties for Portfolio."""

    if TYPE_CHECKING:
        cashposition: pl.DataFrame
        prices: pl.DataFrame
        aum: float
        cost_per_unit: float
        cost_bps: float
        assets: list[str]
        nav_accumulated: pl.DataFrame
        _tilt_cache: Self | None

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

    @property
    def tilt(self) -> Self:
        """Return the 'tilt' portfolio with constant average weights.

        Computes the time-average of each asset's cash position (ignoring
        nulls/NaNs) and builds a new Portfolio with those constant weights
        applied across time. Prices and AUM are preserved.

        The result is cached after first access so repeated calls are O(1).

        Note:
            Caching is not thread-safe.  Concurrent access from multiple
            threads may trigger redundant computation, but will never produce
            incorrect results because each thread stores the same deterministic
            value.
        """
        cache = getattr(self, "_tilt_cache", None)
        if cache is not None:
            return cache
        const_position = self.cashposition.with_columns(
            pl.col(col).drop_nulls().drop_nans().mean().alias(col) for col in self.assets
        )
        result = type(self).from_cash_position(
            self.prices,
            const_position,
            aum=self.aum,
            cost_per_unit=self.cost_per_unit,
            cost_bps=self.cost_bps,
        )
        # Direct write is safe: Portfolio is a frozen, slotted dataclass that
        # declares every cache field, so object.__setattr__ cannot fail here.
        object.__setattr__(self, "_tilt_cache", result)
        return result

    @property
    def timing(self) -> Self:
        """Return the 'timing' portfolio capturing deviations from the tilt.

        Constructs weights as original cash positions minus the tilt's
        constant positions, per asset. This isolates timing (alloc-demeaned)
        effects. Prices and AUM are preserved.
        """
        const_position = self.tilt.cashposition
        position = self.cashposition.with_columns((pl.col(col) - const_position[col]).alias(col) for col in self.assets)
        return type(self).from_cash_position(
            self.prices,
            position,
            aum=self.aum,
            cost_per_unit=self.cost_per_unit,
            cost_bps=self.cost_bps,
        )

    @property
    def tilt_timing_decomp(self) -> pl.DataFrame:
        """Return the portfolio's tilt/timing NAV decomposition.

        When a ``'date'`` column is present the three NAV series are joined on
        it. When data is integer-indexed the frames are stacked horizontally.
        """
        if "date" in self.nav_accumulated.columns:
            nav_portfolio = self.nav_accumulated.select(["date", "NAV_accumulated"])
            nav_tilt = self.tilt.nav_accumulated.select(["date", "NAV_accumulated"])
            nav_timing = self.timing.nav_accumulated.select(["date", "NAV_accumulated"])

            merged_df = nav_portfolio.join(nav_tilt, on="date", how="inner", suffix="_tilt").join(
                nav_timing, on="date", how="inner", suffix="_timing"
            )
        else:
            nav_portfolio = self.nav_accumulated.select(["NAV_accumulated"])
            nav_tilt = self.tilt.nav_accumulated.select(["NAV_accumulated"]).rename(
                {"NAV_accumulated": "NAV_accumulated_tilt"}
            )
            nav_timing = self.timing.nav_accumulated.select(["NAV_accumulated"]).rename(
                {"NAV_accumulated": "NAV_accumulated_timing"}
            )
            merged_df = nav_portfolio.hstack(nav_tilt).hstack(nav_timing)

        merged_df = merged_df.rename(
            {"NAV_accumulated_tilt": "tilt", "NAV_accumulated_timing": "timing", "NAV_accumulated": "portfolio"}
        )
        return merged_df
