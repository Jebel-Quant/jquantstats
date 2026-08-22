"""Rolling risk/return metric line charts (Sharpe, Sortino, volatility, beta)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from .._render import render
from .._specs import (
    rolling_beta_spec,
    rolling_sharpe_spec,
    rolling_sortino_spec,
    rolling_volatility_spec,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure as PlotlyFigure

    from jquantstats._protocol import DataLike

    from .._backend import Backend
    from .._render import Figure


class _RollingPlotsMixin:
    """Rolling-window metric plots for :class:`DataPlots`."""

    __slots__ = ()

    _data: DataLike

    @overload
    def rolling_sharpe(
        self,
        rolling_period: int = ...,
        periods_per_year: int = ...,
        title: str = ...,
        *,
        backend: Literal["plotly"] | None = ...,
    ) -> PlotlyFigure: ...

    @overload
    def rolling_sharpe(
        self,
        rolling_period: int = ...,
        periods_per_year: int = ...,
        title: str = ...,
        *,
        backend: Literal["matplotlib"],
    ) -> MplFigure: ...

    def rolling_sharpe(
        self,
        rolling_period: int = 126,
        periods_per_year: int = 252,
        title: str = "Rolling Sharpe Ratio",
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Rolling annualised Sharpe ratio over time.

        Computes ``rolling_mean / rolling_std * sqrt(periods_per_year)`` with a
        trailing window of *rolling_period* observations for every column in the
        dataset (assets and benchmark when present).

        Args:
            rolling_period: Trailing window size. Defaults to 126 (6 months).
            periods_per_year: Annualisation factor. Defaults to 252.
            title: Chart title. Defaults to ``"Rolling Sharpe Ratio"``.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A line chart.

        """
        spec = rolling_sharpe_spec(self._data, rolling_period, periods_per_year, title)
        return render(spec, backend)

    @overload
    def rolling_sortino(
        self,
        rolling_period: int = ...,
        periods_per_year: int = ...,
        title: str = ...,
        *,
        backend: Literal["plotly"] | None = ...,
    ) -> PlotlyFigure: ...

    @overload
    def rolling_sortino(
        self,
        rolling_period: int = ...,
        periods_per_year: int = ...,
        title: str = ...,
        *,
        backend: Literal["matplotlib"],
    ) -> MplFigure: ...

    def rolling_sortino(
        self,
        rolling_period: int = 126,
        periods_per_year: int = 252,
        title: str = "Rolling Sortino Ratio",
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Rolling annualised Sortino ratio over time.

        Computes ``rolling_mean / rolling_downside_std * sqrt(periods_per_year)``
        where downside deviation considers only negative returns.

        Args:
            rolling_period: Trailing window size. Defaults to 126 (6 months).
            periods_per_year: Annualisation factor. Defaults to 252.
            title: Chart title. Defaults to ``"Rolling Sortino Ratio"``.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A line chart.

        """
        spec = rolling_sortino_spec(self._data, rolling_period, periods_per_year, title)
        return render(spec, backend)

    @overload
    def rolling_volatility(
        self,
        rolling_period: int = ...,
        periods_per_year: int = ...,
        title: str = ...,
        *,
        backend: Literal["plotly"] | None = ...,
    ) -> PlotlyFigure: ...

    @overload
    def rolling_volatility(
        self,
        rolling_period: int = ...,
        periods_per_year: int = ...,
        title: str = ...,
        *,
        backend: Literal["matplotlib"],
    ) -> MplFigure: ...

    def rolling_volatility(
        self,
        rolling_period: int = 126,
        periods_per_year: int = 252,
        title: str = "Rolling Volatility",
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Rolling annualised volatility over time.

        Computes ``rolling_std * sqrt(periods_per_year)`` for every column in
        the dataset.

        Args:
            rolling_period: Trailing window size. Defaults to 126 (6 months).
            periods_per_year: Annualisation factor. Defaults to 252.
            title: Chart title. Defaults to ``"Rolling Volatility"``.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A line chart.

        """
        spec = rolling_volatility_spec(self._data, rolling_period, periods_per_year, title)
        return render(spec, backend)

    @overload
    def rolling_beta(
        self,
        rolling_period: int = ...,
        rolling_period2: int | None = ...,
        title: str = ...,
        figsize: tuple[int, int] | None = ...,
        *,
        backend: Literal["plotly"] | None = ...,
    ) -> PlotlyFigure: ...

    @overload
    def rolling_beta(
        self,
        rolling_period: int = ...,
        rolling_period2: int | None = ...,
        title: str = ...,
        figsize: tuple[int, int] | None = ...,
        *,
        backend: Literal["matplotlib"],
    ) -> MplFigure: ...

    def rolling_beta(
        self,
        rolling_period: int = 126,
        rolling_period2: int | None = 252,
        title: str = "Rolling Beta",
        figsize: tuple[int, int] | None = None,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Rolling beta versus the benchmark.

        Plots one line per asset per window size.  Beta is estimated via the
        standard OLS formula: ``cov(asset, bench) / var(bench)`` computed over
        a trailing window.

        Args:
            rolling_period: Primary trailing window size. Defaults to 126.
            rolling_period2: Optional second window size overlaid on the same
                chart. Defaults to 252. Pass ``None`` to omit.
            title: Chart title. Defaults to ``"Rolling Beta"``.
            figsize: Optional ``(width, height)`` in pixels.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A line chart.

        Raises:
            NoBenchmarkError: If no benchmark columns are present in the data.

        """
        spec = rolling_beta_spec(self._data, rolling_period, rolling_period2, title, figsize)
        return render(spec, backend)
