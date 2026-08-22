"""Rolling-window and per-year risk charts for a portfolio.

Split out of the former single-module `_plots/_portfolio.py`; composed into
:class:`PortfolioPlots` by `_core.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from .._render import render
from .._specs import (
    annual_sharpe_spec,
    portfolio_rolling_sharpe_spec,
    portfolio_rolling_volatility_spec,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure as PlotlyFigure

    from .._backend import Backend
    from .._protocol import PortfolioLike
    from .._render import Figure


class _RollingPortfolioPlotsMixin:
    """Rolling-window and annual risk charts for :class:`PortfolioPlots`."""

    __slots__ = ()

    _portfolio: PortfolioLike

    @overload
    def rolling_sharpe_plot(self, window: int = ..., *, backend: Literal["plotly"] | None = ...) -> PlotlyFigure: ...

    @overload
    def rolling_sharpe_plot(self, window: int = ..., *, backend: Literal["matplotlib"]) -> MplFigure: ...

    def rolling_sharpe_plot(self, window: int = 63, *, backend: Backend | None = None) -> Figure:
        """Plot rolling annualised Sharpe ratio over time.

        Computes the rolling Sharpe for each asset column using the given
        window and renders one line per asset.

        Args:
            window: Rolling-window size in periods. Defaults to 63.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A chart with one line per asset.

        Raises:
            ValueError: If ``window`` is not a positive integer.

        """
        return render(portfolio_rolling_sharpe_spec(self._portfolio, window), backend)

    @overload
    def rolling_volatility_plot(
        self, window: int = ..., *, backend: Literal["plotly"] | None = ...
    ) -> PlotlyFigure: ...

    @overload
    def rolling_volatility_plot(self, window: int = ..., *, backend: Literal["matplotlib"]) -> MplFigure: ...

    def rolling_volatility_plot(self, window: int = 63, *, backend: Backend | None = None) -> Figure:
        """Plot rolling annualised volatility over time.

        Computes the rolling volatility for each asset column using the given
        window and renders one line per asset.

        Args:
            window: Rolling-window size in periods. Defaults to 63.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A chart with one line per asset.

        Raises:
            ValueError: If ``window`` is not a positive integer.

        """
        return render(portfolio_rolling_volatility_spec(self._portfolio, window), backend)

    @overload
    def annual_sharpe_plot(self, *, backend: Literal["plotly"] | None = ...) -> PlotlyFigure: ...

    @overload
    def annual_sharpe_plot(self, *, backend: Literal["matplotlib"]) -> MplFigure: ...

    def annual_sharpe_plot(self, *, backend: Backend | None = None) -> Figure:
        """Plot annualised Sharpe ratio broken down by calendar year.

        Computes the Sharpe ratio for each calendar year from the portfolio
        returns and renders a grouped bar chart with one bar per year per
        asset.

        Args:
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A grouped bar chart, one bar group per asset.

        """
        return render(annual_sharpe_spec(self._portfolio), backend)
