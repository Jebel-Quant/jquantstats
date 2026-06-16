"""Shared type-only declarations for the Portfolio mixins.

The four Portfolio mixins (`PortfolioNavMixin`, `PortfolioAttributionMixin`,
`PortfolioTurnoverMixin`, `PortfolioCostMixin`) each consume attributes and
properties that are actually defined on *sibling* mixins or on the composed
`Portfolio` dataclass itself. So that every mixin type-checks in isolation under
``mypy --strict``, they all inherit this base, which declares that shared
surface once under ``TYPE_CHECKING``.

At runtime the body is skipped entirely, so `_PortfolioMembers` is an empty
class: it adds nothing to instance layout (the mixins are slot-free already) and
nothing to behaviour — it exists purely to give the type checker a single source
of truth for the cross-mixin surface instead of repeating the stubs in every
mixin module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .data import Data


class _PortfolioMembers:
    """Type-only declaration of the cross-mixin Portfolio surface (see module docstring)."""

    if TYPE_CHECKING:
        # Raw inputs — real ``Portfolio`` dataclass fields.
        cashposition: pl.DataFrame
        prices: pl.DataFrame
        aum: float
        cost_per_unit: float
        cost_bps: float

        # Derived series / accessors defined on sibling mixins or ``Portfolio``.
        @property
        def data(self) -> Data:
            """Bridge to the legacy `Data` object (defined on `Portfolio`)."""

        @property
        def assets(self) -> list[str]:
            """Asset column names (defined on `Portfolio`)."""

        @property
        def returns(self) -> pl.DataFrame:
            """Daily returns (defined on `PortfolioNavMixin`)."""

        @property
        def profit(self) -> pl.DataFrame:
            """Aggregate daily portfolio profit (defined on `PortfolioNavMixin`)."""

        @property
        def nav_accumulated(self) -> pl.DataFrame:
            """Cumulative additive NAV (defined on `PortfolioNavMixin`)."""

        @property
        def turnover(self) -> pl.DataFrame:
            """Turnover frame (defined on `PortfolioTurnoverMixin`)."""
