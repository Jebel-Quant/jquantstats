"""Monte Carlo simulation charts (fan chart and metric distribution)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from .._render import render
from .._specs import montecarlo_distribution_spec, montecarlo_spec

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure as PlotlyFigure

    from jquantstats._protocol import DataLike

    from .._backend import Backend
    from .._render import Figure


class _MonteCarloPlotsMixin:
    """Monte Carlo simulation plots for :class:`DataPlots`."""

    __slots__ = ()

    _data: DataLike

    @overload
    def montecarlo(
        self,
        n: int = ...,
        period: int = ...,
        title: str = ...,
        figsize: tuple[int, int] | None = ...,
        *,
        backend: Literal["plotly"] | None = ...,
    ) -> PlotlyFigure: ...

    @overload
    def montecarlo(
        self,
        n: int = ...,
        period: int = ...,
        title: str = ...,
        figsize: tuple[int, int] | None = ...,
        *,
        backend: Literal["matplotlib"],
    ) -> MplFigure: ...

    def montecarlo(
        self,
        n: int = 100,
        period: int = 252,
        title: str = "Monte Carlo Simulation",
        figsize: tuple[int, int] | None = None,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Fan chart of Monte Carlo simulated cumulative return paths.

        For each asset column, draws ``n`` bootstrapped paths sampled with
        replacement from historical returns and overlays the observed path for
        the trailing *period* observations.

        Args:
            n: Number of simulated paths per asset. Defaults to 100.
            period: Number of observations per path. Defaults to 252.
            title: Chart title. Defaults to ``"Monte Carlo Simulation"``.
            figsize: Optional figure ``(width, height)`` in pixels.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A fan chart.

        Raises:
            ValueError: If ``n`` or ``period`` is not a positive integer.

        """
        return render(montecarlo_spec(self._data, n, period, title, figsize), backend)

    @overload
    def montecarlo_distribution(
        self,
        n: int = ...,
        period: int = ...,
        metric: str = ...,
        title: str = ...,
        figsize: tuple[int, int] | None = ...,
        *,
        backend: Literal["plotly"] | None = ...,
    ) -> PlotlyFigure: ...

    @overload
    def montecarlo_distribution(
        self,
        n: int = ...,
        period: int = ...,
        metric: str = ...,
        title: str = ...,
        figsize: tuple[int, int] | None = ...,
        *,
        backend: Literal["matplotlib"],
    ) -> MplFigure: ...

    def montecarlo_distribution(
        self,
        n: int = 1000,
        period: int = 252,
        metric: str = "sharpe",
        title: str = "Monte Carlo Distribution",
        figsize: tuple[int, int] | None = None,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Distribution of Monte Carlo simulation metrics.

        Computes one metric per simulated path and shows the resulting
        distribution as a histogram with the observed trailing-period value
        overlaid as a vertical reference line.

        Supported metrics:
            - ``"sharpe"`` (annualized, 252 periods/year)
            - ``"drawdown"`` (maximum drawdown, negative value)
            - ``"cagr"`` (annualized geometric return)

        Args:
            n: Number of simulations per asset. Defaults to 1000.
            period: Number of observations in each simulation. Defaults to 252.
            metric: Metric to evaluate. One of ``"sharpe"``, ``"drawdown"``,
                or ``"cagr"``.
            title: Chart title. Defaults to ``"Monte Carlo Distribution"``.
            figsize: Optional figure ``(width, height)`` in pixels.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A histogram figure.

        Raises:
            ValueError: If ``n`` or ``period`` is not a positive integer, or
                ``metric`` is not one of the supported names.

        """
        spec = montecarlo_distribution_spec(self._data, n, period, metric, title, figsize)
        return render(spec, backend)
