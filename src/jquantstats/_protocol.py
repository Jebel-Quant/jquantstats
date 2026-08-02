"""Shared protocol definitions used across jquantstats subpackages.

Design rationale
----------------
The analytics subpackages (``_stats``, ``_plots``, ``_reports``, ``_utils``)
must not import the concrete `Data` / `Portfolio` classes at runtime — that
would create circular imports, since those classes compose the subpackages.

**This is enforced, not just documented.** ``[tool.importlinter]`` in
``pyproject.toml`` declares it as a contract that ``make arch`` checks and
``make test`` runs, so a layering inversion fails CI rather than review. Imports
under ``if TYPE_CHECKING:`` are excluded from the contract — annotating against
the concrete class costs nothing at runtime and forms no cycle, so it stays
allowed. Where a subpackage needs to *construct* a `Data` (rather than annotate
one), it goes through ``type(self._data)`` instead of importing the class; see
``_stats/_reporting.py::_summary_frame`` and ``_data_reshape.py::_rebuild``.

Each consumer annotates against a structural Protocol:

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

    @property
    def returns(self) -> pl.DataFrame:
        """Return DataFrame (asset columns only, no benchmark or date)."""
        ...

    @property
    def index(self) -> pl.DataFrame:
        """Date / time index DataFrame."""
        ...

    @property
    def benchmark(self) -> pl.DataFrame | None:
        """Benchmark DataFrame, or None when no benchmark was provided."""
        ...

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
