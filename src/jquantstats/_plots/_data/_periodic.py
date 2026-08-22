"""Periodic-return bar charts and the monthly-return heatmap."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from .._render import render
from .._specs import (
    daily_returns_spec,
    monthly_heatmap_spec,
    monthly_returns_spec,
    yearly_returns_spec,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure as PlotlyFigure

    from jquantstats._protocol import DataLike

    from .._backend import Backend
    from .._render import Figure


class _PeriodicPlotsMixin:
    """Daily/monthly/yearly bar charts and the monthly heatmap for :class:`DataPlots`."""

    __slots__ = ()

    _data: DataLike

    @overload
    def daily_returns(self, title: str = ..., *, backend: Literal["plotly"] | None = ...) -> PlotlyFigure: ...

    @overload
    def daily_returns(self, title: str = ..., *, backend: Literal["matplotlib"]) -> MplFigure: ...

    def daily_returns(self, title: str = "Daily Returns", *, backend: Backend | None = None) -> Figure:
        """Daily returns as a bar chart.

        Each bar is coloured green for positive returns and red for negative
        returns.  When multiple assets are present each asset gets its own
        trace in the palette colour with opacity used for positive/negative
        differentiation.

        Args:
            title: Chart title. Defaults to ``"Daily Returns"``.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A bar chart.

        """
        return render(daily_returns_spec(self._data, title=title), backend)

    @overload
    def yearly_returns(
        self, title: str = ..., compounded: bool = ..., *, backend: Literal["plotly"] | None = ...
    ) -> PlotlyFigure: ...

    @overload
    def yearly_returns(
        self, title: str = ..., compounded: bool = ..., *, backend: Literal["matplotlib"]
    ) -> MplFigure: ...

    def yearly_returns(
        self,
        title: str = "Yearly Returns",
        compounded: bool = True,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Annual compounded (or summed) returns as a grouped bar chart.

        Args:
            title: Chart title. Defaults to ``"Yearly Returns"``.
            compounded: Compound returns within each year. Defaults to True.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A grouped bar chart.

        """
        return render(yearly_returns_spec(self._data, title=title, compounded=compounded), backend)

    @overload
    def monthly_returns(
        self, title: str = ..., compounded: bool = ..., *, backend: Literal["plotly"] | None = ...
    ) -> PlotlyFigure: ...

    @overload
    def monthly_returns(
        self, title: str = ..., compounded: bool = ..., *, backend: Literal["matplotlib"]
    ) -> MplFigure: ...

    def monthly_returns(
        self,
        title: str = "Monthly Returns",
        compounded: bool = True,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Monthly compounded (or summed) returns as a bar chart.

        Args:
            title: Chart title. Defaults to ``"Monthly Returns"``.
            compounded: Compound returns within each month. Defaults to True.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A bar chart.

        """
        return render(monthly_returns_spec(self._data, title=title, compounded=compounded), backend)

    @overload
    def monthly_heatmap(
        self,
        title: str = ...,
        compounded: bool = ...,
        asset: str | None = ...,
        *,
        backend: Literal["plotly"] | None = ...,
    ) -> PlotlyFigure: ...

    @overload
    def monthly_heatmap(
        self,
        title: str = ...,
        compounded: bool = ...,
        asset: str | None = ...,
        *,
        backend: Literal["matplotlib"],
    ) -> MplFigure: ...

    def monthly_heatmap(
        self,
        title: str = "Monthly Returns Heatmap",
        compounded: bool = True,
        asset: str | None = None,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Monthly returns calendar heatmap (year x month).

        One heatmap is produced per call for a single asset.  Green cells
        indicate positive months; red cells indicate negative months.

        Args:
            title: Chart title. Defaults to ``"Monthly Returns Heatmap"``.
            compounded: Compound intra-month returns. Defaults to True.
            asset: Asset column name to display.  Defaults to the first
                non-date column in the dataset.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A calendar heatmap.

        """
        spec = monthly_heatmap_spec(self._data, title=title, compounded=compounded, asset=asset)
        return render(spec, backend)
