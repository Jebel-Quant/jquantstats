"""Shared protocol definitions used across jquantstats subpackages.

Design rationale
----------------
The analytics subpackages (``_stats``, ``_plots``, ``_reports``, ``_utils``)
must not import the concrete `Data` / `Portfolio` classes at runtime — that
would create circular imports, since those classes compose the subpackages.
Instead, each consumer annotates against a structural Protocol:

- `DataLike` and `StatsLike` (this module) are shared by every subpackage —
  there is exactly one definition of each.
- ``PortfolioLike`` is deliberately *not* shared: each subpackage declares its
  own (``_plots/_protocol.py``, ``_reports/_protocol.py``,
  ``_utils/_protocol.py``) listing only the members it actually consumes
  (interface segregation). Keep it that way — a merged PortfolioLike would
  re-couple the subpackages to the full Portfolio surface.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import polars as pl


class StatsLike(Protocol):  # pragma: no cover
    """Structural interface for the statistics facade used by reports."""

    def summary(self) -> pl.DataFrame:
        """Full summary DataFrame (one row per metric, one column per asset)."""
        ...


@runtime_checkable
class DataLike(Protocol):  # pragma: no cover
    """Authoritative structural interface for Data consumers.

    Union of the members required by the stats mixins, plots, reports, and
    utils — annotating against the superset is harmless for consumers that
    use only part of it, and keeps a single definition.
    """

    returns: pl.DataFrame
    index: pl.DataFrame
    benchmark: pl.DataFrame | None

    @property
    def all(self) -> pl.DataFrame:
        """Combined DataFrame of date index, return, and benchmark columns."""
        ...

    @property
    def assets(self) -> list[str]:
        """Names of the asset return columns."""
        ...

    @property
    def date_col(self) -> list[str]:
        """Column names used as the date/time index."""
        ...

    @property
    def stats(self) -> StatsLike:
        """Statistics facade used by reports."""
        ...

    @property
    def _periods_per_year(self) -> float:
        """Estimated number of return periods per calendar year."""
        ...

    def items(self) -> Iterator[tuple[str, pl.Series]]:
        """Iterate over (asset_name, returns_series) pairs."""
        ...
