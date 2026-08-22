"""The :class:`PortfolioPlots` facade combining the portfolio plot-family mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._diagnostics import _DiagnosticPlotsMixin
from ._nav import _NavPlotsMixin
from ._rolling import _RollingPortfolioPlotsMixin

if TYPE_CHECKING:
    from .._protocol import PortfolioLike


class PortfolioPlots(
    _NavPlotsMixin,
    _RollingPortfolioPlotsMixin,
    _DiagnosticPlotsMixin,
):
    """Facade for portfolio plots built with Plotly.

    Provides convenience methods to visualize portfolio performance and
    diagnostics directly from a Portfolio instance (e.g., snapshot charts,
    lagged performance, smoothed holdings, and lead/lag IR).

    Charts are organised into focused mixins:

    - `_NavPlotsMixin` — accumulated-NAV curves: snapshot, lag sweep,
      smoothed holdings.
    - `_RollingPortfolioPlotsMixin` — rolling Sharpe/volatility and the
      per-year Sharpe breakdown.
    - `_DiagnosticPlotsMixin` — lead/lag IR, correlation heatmap, monthly
      returns calendar, trading-cost impact.
    """

    __slots__ = ("_portfolio",)

    def __init__(self, portfolio: PortfolioLike) -> None:
        self._portfolio = portfolio
