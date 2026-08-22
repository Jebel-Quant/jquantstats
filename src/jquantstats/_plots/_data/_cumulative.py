"""Cumulative-return and equity-curve line charts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from .._render import render
from .._specs import compare_spec, cumulative_returns_spec, earnings_spec, log_returns_spec

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure as PlotlyFigure

    from jquantstats._protocol import DataLike

    from .._backend import Backend
    from .._render import Figure


class _CumulativePlotsMixin:
    """Cumulative-return and equity-curve plots for :class:`DataPlots`.

    Every method here accepts a ``backend`` keyword selecting the renderer, and
    is overloaded on it: omitting it (or naming ``"plotly"``) is typed as
    returning a `plotly.graph_objects.Figure`, so callers written before the
    matplotlib backend existed keep exactly the type they had.

    One caveat that no type system can express: after
    `jquantstats.set_plot_backend` changes the process-wide default, a call
    that passes no ``backend`` returns the new default's figure type while
    still being *typed* as Plotly's. Pass ``backend=`` explicitly if you change
    the global default and care about static types.
    """

    __slots__ = ()

    _data: DataLike

    @overload
    def returns(
        self, title: str = ..., log_scale: bool = ..., *, backend: Literal["plotly"] | None = ...
    ) -> PlotlyFigure: ...

    @overload
    def returns(self, title: str = ..., log_scale: bool = ..., *, backend: Literal["matplotlib"]) -> MplFigure: ...

    def returns(
        self,
        title: str = "Cumulative Returns",
        log_scale: bool = False,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Cumulative compounded returns over time.

        Plots ``(1 + r).cumprod()`` for every column in the dataset (including
        benchmark when present).

        Args:
            title: Chart title. Defaults to ``"Cumulative Returns"``.
            log_scale: Use a logarithmic y-axis. Defaults to False.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A line chart.

        """
        return render(cumulative_returns_spec(self._data, title=title, log_scale=log_scale), backend)

    @overload
    def compare(
        self,
        title: str = ...,
        figsize: tuple[int, int] | None = ...,
        *,
        backend: Literal["plotly"] | None = ...,
    ) -> PlotlyFigure: ...

    @overload
    def compare(
        self, title: str = ..., figsize: tuple[int, int] | None = ..., *, backend: Literal["matplotlib"]
    ) -> MplFigure: ...

    def compare(
        self,
        title: str = "Comparison vs Benchmark",
        figsize: tuple[int, int] | None = None,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Compare cumulative returns of each asset against the benchmark.

        Args:
            title: Chart title. Defaults to ``"Comparison vs Benchmark"``.
            figsize: Optional ``(width, height)`` in pixels.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A line chart.

        Raises:
            AttributeError: If no benchmark data is available.

        """
        return render(compare_spec(self._data, title=title, figsize=figsize), backend)

    @overload
    def log_returns(
        self,
        title: str = ...,
        figsize: tuple[int, int] | None = ...,
        *,
        backend: Literal["plotly"] | None = ...,
    ) -> PlotlyFigure: ...

    @overload
    def log_returns(
        self, title: str = ..., figsize: tuple[int, int] | None = ..., *, backend: Literal["matplotlib"]
    ) -> MplFigure: ...

    def log_returns(
        self,
        title: str = "Log Returns",
        figsize: tuple[int, int] | None = None,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Cumulative log returns over time.

        Plots ``log((1 + r).cumprod())`` — the natural log of the compounded
        growth factor — which linearises exponential growth and makes
        multi-asset comparisons on a common scale.

        Args:
            title: Chart title. Defaults to ``"Log Returns"``.
            figsize: Optional ``(width, height)`` in pixels.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A line chart.

        """
        return render(log_returns_spec(self._data, title=title, figsize=figsize), backend)

    @overload
    def earnings(
        self,
        start_balance: float = ...,
        title: str = ...,
        compounded: bool = ...,
        *,
        backend: Literal["plotly"] | None = ...,
    ) -> PlotlyFigure: ...

    @overload
    def earnings(
        self,
        start_balance: float = ...,
        title: str = ...,
        compounded: bool = ...,
        *,
        backend: Literal["matplotlib"],
    ) -> MplFigure: ...

    def earnings(
        self,
        start_balance: float = 1e5,
        title: str = "Portfolio Earnings",
        compounded: bool = True,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Dollar equity curve showing portfolio value over time.

        Scales cumulative returns by *start_balance* so the y-axis reflects
        an absolute portfolio value rather than a dimensionless growth factor.

        Args:
            start_balance: Starting portfolio value in currency units.
                Defaults to 100 000.
            title: Chart title. Defaults to ``"Portfolio Earnings"``.
            compounded: Use compounded returns (``cumprod``). When False uses
                cumulative sum. Defaults to True.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A line chart.

        """
        spec = earnings_spec(self._data, start_balance=start_balance, title=title, compounded=compounded)
        return render(spec, backend)
