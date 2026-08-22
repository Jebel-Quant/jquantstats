"""Return-distribution charts (overlaid histograms and by-period box plots)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from .._render import render
from .._specs import distribution_spec, histogram_spec

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure as PlotlyFigure

    from jquantstats._protocol import DataLike

    from .._backend import Backend
    from .._render import Figure


class _DistributionPlotsMixin:
    """Return-distribution plots for :class:`DataPlots`."""

    __slots__ = ()

    _data: DataLike

    @overload
    def histogram(
        self, title: str = ..., bins: int = ..., *, backend: Literal["plotly"] | None = ...
    ) -> PlotlyFigure: ...

    @overload
    def histogram(self, title: str = ..., bins: int = ..., *, backend: Literal["matplotlib"]) -> MplFigure: ...

    def histogram(
        self,
        title: str = "Returns Distribution",
        bins: int = 50,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Overlaid return histograms, one per series.

        Each asset (and the benchmark, when present) is drawn as a
        semi-transparent histogram on shared axes, so the distributions can be
        compared directly — a fat-tailed asset against a tightly peaked
        benchmark, for instance.

        Args:
            title: Chart title. Defaults to ``"Returns Distribution"``.
            bins: Number of histogram bins. Defaults to 50.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A histogram figure.

        """
        return render(histogram_spec(self._data, title=title, bins=bins), backend)

    @overload
    def distribution(
        self, title: str = ..., compounded: bool = ..., *, backend: Literal["plotly"] | None = ...
    ) -> PlotlyFigure: ...

    @overload
    def distribution(
        self, title: str = ..., compounded: bool = ..., *, backend: Literal["matplotlib"]
    ) -> MplFigure: ...

    def distribution(
        self,
        title: str = "Return Distribution by Period",
        compounded: bool = True,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Return distributions across daily, weekly, monthly, quarterly and yearly periods.

        Renders a box plot for each aggregation period so the user can compare
        how the distribution widens as the holding period lengthens.  One
        subplot column is produced per asset.

        Args:
            title: Chart title. Defaults to ``"Return Distribution by Period"``.
            compounded: Compound returns within each period. Defaults to True.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A figure with one panel per asset.

        """
        return render(distribution_spec(self._data, title=title, compounded=compounded), backend)
