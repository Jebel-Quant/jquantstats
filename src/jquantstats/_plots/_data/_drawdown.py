"""Drawdown charts (underwater curve and worst-period shading)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from .._render import render
from .._specs import drawdown_spec, drawdowns_periods_spec

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure as PlotlyFigure

    from jquantstats._protocol import DataLike

    from .._backend import Backend
    from .._render import Figure


class _DrawdownPlotsMixin:
    """Drawdown plots for :class:`DataPlots`."""

    __slots__ = ()

    _data: DataLike

    @overload
    def drawdown(self, title: str = ..., *, backend: Literal["plotly"] | None = ...) -> PlotlyFigure: ...

    @overload
    def drawdown(self, title: str = ..., *, backend: Literal["matplotlib"]) -> MplFigure: ...

    def drawdown(self, title: str = "Drawdowns", *, backend: Backend | None = None) -> Figure:
        """Underwater equity curve (drawdown) chart.

        Shows the percentage decline from the running peak for every column
        in the dataset (assets and benchmark where present).

        Args:
            title: Chart title. Defaults to ``"Drawdowns"``.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A filled-area chart.

        """
        return render(drawdown_spec(self._data, title=title), backend)

    @overload
    def drawdowns_periods(
        self,
        n: int = ...,
        title: str = ...,
        asset: str | None = ...,
        *,
        backend: Literal["plotly"] | None = ...,
    ) -> PlotlyFigure: ...

    @overload
    def drawdowns_periods(
        self,
        n: int = ...,
        title: str = ...,
        asset: str | None = ...,
        *,
        backend: Literal["matplotlib"],
    ) -> MplFigure: ...

    def drawdowns_periods(
        self,
        n: int = 5,
        title: str = "Top Drawdown Periods",
        asset: str | None = None,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Cumulative returns chart with the worst *n* drawdown periods shaded.

        Identifies the *n* deepest drawdown periods and overlays coloured
        rectangular shading on the cumulative returns line.  One asset is
        shown per call.

        Args:
            n: Number of worst drawdown periods to highlight. Defaults to 5.
            title: Chart title. Defaults to ``"Top Drawdown Periods"``.
            asset: Asset column name.  Defaults to the first non-date column.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: An equity curve with the worst episodes shaded.

        """
        return render(drawdowns_periods_spec(self._data, n=n, title=title, asset=asset), backend)
