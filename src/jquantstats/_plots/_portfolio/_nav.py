"""Accumulated-NAV charts for a portfolio.

Split out of the former single-module `_plots/_portfolio.py`; composed into
:class:`PortfolioPlots` by `_core.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from .._render import render
from .._specs import (
    lagged_performance_spec,
    portfolio_snapshot_spec,
    smoothed_holdings_performance_spec,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure as PlotlyFigure

    from .._backend import Backend
    from .._protocol import PortfolioLike
    from .._render import Figure


class _NavPlotsMixin:
    """Accumulated-NAV charts for :class:`PortfolioPlots`."""

    __slots__ = ()

    _portfolio: PortfolioLike

    @overload
    def snapshot(self, log_scale: bool = ..., *, backend: Literal["plotly"] | None = ...) -> PlotlyFigure: ...

    @overload
    def snapshot(self, log_scale: bool = ..., *, backend: Literal["matplotlib"]) -> MplFigure: ...

    def snapshot(self, log_scale: bool = False, *, backend: Backend | None = None) -> Figure:
        """Return a snapshot dashboard of NAV and drawdown.

        When the portfolio has a non-zero ``cost_model.cost_per_unit``, an additional
        ``"Net-of-Cost NAV"`` trace is overlaid on the NAV panel showing the
        realised NAV path after deducting position-delta trading costs.

        Args:
            log_scale: If True, display NAV on a log scale. Defaults to False.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: Accumulated NAV (including tilt/timing) over a shaded
            drawdown panel.

        """
        return render(portfolio_snapshot_spec(self._portfolio, log_scale=log_scale), backend)

    @overload
    def lagged_performance_plot(
        self,
        lags: list[int] | None = ...,
        log_scale: bool = ...,
        *,
        backend: Literal["plotly"] | None = ...,
    ) -> PlotlyFigure: ...

    @overload
    def lagged_performance_plot(
        self,
        lags: list[int] | None = ...,
        log_scale: bool = ...,
        *,
        backend: Literal["matplotlib"],
    ) -> MplFigure: ...

    def lagged_performance_plot(
        self,
        lags: list[int] | None = None,
        log_scale: bool = False,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Plot NAV_accumulated for multiple lagged portfolios.

        Creates a figure with one line per lag value showing the accumulated
        NAV series for the portfolio with cash positions shifted by that lag.
        By default, lags [0, 1, 2, 3, 4] are used.

        Args:
            lags: A list of integer lags to apply; defaults to [0, 1, 2, 3, 4].
            log_scale: If True, set the primary y-axis to logarithmic scale.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: One trace per requested lag.

        Raises:
            TypeError: If ``lags`` is not a list of integers.

        """
        return render(lagged_performance_spec(self._portfolio, lags, log_scale), backend)

    @overload
    def smoothed_holdings_performance_plot(
        self,
        windows: list[int] | None = ...,
        log_scale: bool = ...,
        *,
        backend: Literal["plotly"] | None = ...,
    ) -> PlotlyFigure: ...

    @overload
    def smoothed_holdings_performance_plot(
        self,
        windows: list[int] | None = ...,
        log_scale: bool = ...,
        *,
        backend: Literal["matplotlib"],
    ) -> MplFigure: ...

    def smoothed_holdings_performance_plot(
        self,
        windows: list[int] | None = None,
        log_scale: bool = False,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Plot NAV_accumulated for smoothed-holding portfolios.

        Builds portfolios with cash positions smoothed by a trailing rolling
        mean over the previous ``n`` steps (window size n+1) for n in
        ``windows`` (defaults to [0, 1, 2, 3, 4]) and plots their
        accumulated NAV curves.

        Args:
            windows: List of non-negative integers specifying smoothing steps
                to include; defaults to [0, 1, 2, 3, 4].
            log_scale: If True, set the primary y-axis to logarithmic scale.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: One line per requested smoothing level.

        Raises:
            TypeError: If ``windows`` is not a list of non-negative integers.

        """
        return render(smoothed_holdings_performance_spec(self._portfolio, windows, log_scale), backend)
