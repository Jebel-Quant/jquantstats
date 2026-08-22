"""Diagnostic charts: lead/lag IR, correlation, monthly calendar, cost impact.

Split out of the former single-module `_plots/_portfolio.py`; composed into
:class:`PortfolioPlots` by `_core.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from .._render import render
from .._specs import (
    correlation_heatmap_spec,
    lead_lag_ir_spec,
    monthly_returns_heatmap_spec,
    trading_cost_impact_spec,
)

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure as PlotlyFigure

    from .._backend import Backend
    from .._protocol import PortfolioLike
    from .._render import Figure


class _DiagnosticPlotsMixin:
    """Diagnostic charts for :class:`PortfolioPlots`."""

    __slots__ = ()

    _portfolio: PortfolioLike

    @overload
    def lead_lag_ir_plot(
        self, start: int = ..., end: int = ..., *, backend: Literal["plotly"] | None = ...
    ) -> PlotlyFigure: ...

    @overload
    def lead_lag_ir_plot(self, start: int = ..., end: int = ..., *, backend: Literal["matplotlib"]) -> MplFigure: ...

    def lead_lag_ir_plot(
        self,
        start: int = -10,
        end: int = 19,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Plot Sharpe ratio (IR) across lead/lag variants of the portfolio.

        Builds portfolios with cash positions lagged from ``start`` to ``end``
        (inclusive) and plots a bar chart of the Sharpe ratio for each lag.
        Positive lags delay weights; negative lags lead them.

        Args:
            start: First lag to include (default: -10).
            end: Last lag to include (default: +19).
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: One bar per lag, labelled by the lag value.

        Raises:
            TypeError: If ``start`` or ``end`` is not an integer.

        """
        return render(lead_lag_ir_spec(self._portfolio, start, end), backend)

    @overload
    def correlation_heatmap(
        self,
        frame: pl.DataFrame | None = ...,
        name: str = ...,
        title: str = ...,
        *,
        backend: Literal["plotly"] | None = ...,
    ) -> PlotlyFigure: ...

    @overload
    def correlation_heatmap(
        self,
        frame: pl.DataFrame | None = ...,
        name: str = ...,
        title: str = ...,
        *,
        backend: Literal["matplotlib"],
    ) -> MplFigure: ...

    def correlation_heatmap(
        self,
        frame: pl.DataFrame | None = None,
        name: str = "portfolio",
        title: str = "Correlation heatmap",
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Plot the correlation matrix of the holdings and the portfolio.

        Args:
            frame: Series to correlate against.  Defaults to the portfolio's
                prices.
            name: Column name given to the portfolio's own profit series.
            title: Chart title.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A square correlation matrix.

        """
        return render(correlation_heatmap_spec(self._portfolio, frame, name, title), backend)

    @overload
    def monthly_returns_heatmap(self, *, backend: Literal["plotly"] | None = ...) -> PlotlyFigure: ...

    @overload
    def monthly_returns_heatmap(self, *, backend: Literal["matplotlib"]) -> MplFigure: ...

    def monthly_returns_heatmap(self, *, backend: Backend | None = None) -> Figure:
        """Plot a monthly returns calendar heatmap.

        Groups portfolio returns by calendar year and month, then renders a
        heatmap with months on the x-axis and years on the y-axis.  Green
        cells indicate positive months; red cells indicate negative months.
        Cell text shows the percentage return for that month.

        Args:
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A calendar heatmap of monthly returns.

        """
        return render(monthly_returns_heatmap_spec(self._portfolio), backend)

    @overload
    def trading_cost_impact_plot(
        self, max_bps: int = ..., *, backend: Literal["plotly"] | None = ...
    ) -> PlotlyFigure: ...

    @overload
    def trading_cost_impact_plot(self, max_bps: int = ..., *, backend: Literal["matplotlib"]) -> MplFigure: ...

    def trading_cost_impact_plot(self, max_bps: int = 20, *, backend: Backend | None = None) -> Figure:
        """Plot the Sharpe ratio as a function of one-way trading costs.

        Evaluates the portfolio's annualised Sharpe ratio at each integer
        cost level from 0 up to ``max_bps`` basis points and renders the
        result as a line chart.  The zero-cost Sharpe is shown as a
        reference horizontal line so that the reader can quickly gauge
        at what cost level the strategy's edge is eroded.

        Args:
            max_bps: Maximum one-way trading cost to evaluate, in basis
                points.  Defaults to 20.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: One line showing Sharpe against cost.

        Raises:
            ValueError: If ``max_bps`` is not a positive integer.

        """
        return render(trading_cost_impact_spec(self._portfolio, max_bps), backend)
