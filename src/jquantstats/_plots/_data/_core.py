"""The :class:`DataPlots` facade combining the plot-family mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from .._render import render
from .._specs import data_snapshot_spec
from ._cumulative import _CumulativePlotsMixin
from ._distribution import _DistributionPlotsMixin
from ._drawdown import _DrawdownPlotsMixin
from ._montecarlo import _MonteCarloPlotsMixin
from ._periodic import _PeriodicPlotsMixin
from ._rolling import _RollingPlotsMixin

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure as PlotlyFigure

    from jquantstats._protocol import DataLike

    from .._backend import Backend
    from .._render import Figure


class DataPlots(
    _CumulativePlotsMixin,
    _PeriodicPlotsMixin,
    _DistributionPlotsMixin,
    _MonteCarloPlotsMixin,
    _DrawdownPlotsMixin,
    _RollingPlotsMixin,
):
    """Visualization tools for financial returns data.

    This class provides methods for creating various plots and visualizations
    of financial returns data, including:

    - Returns bar charts
    - Portfolio performance snapshots
    - Monthly returns heatmaps

    The class is designed to work with the _Data class and uses Plotly
    for creating interactive visualizations.
    """

    __slots__ = ("_data",)

    def __init__(self, data: DataLike) -> None:
        self._data = data

    @property
    def assets(self) -> list[str]:
        """Asset column names from the underlying data."""
        return self._data.assets

    def __repr__(self) -> str:
        """Return a string representation of the DataPlots object."""
        return f"DataPlots(assets={self._data.assets})"

    @overload
    def snapshot(
        self, title: str = ..., log_scale: bool = ..., *, backend: Literal["plotly"] | None = ...
    ) -> PlotlyFigure: ...

    @overload
    def snapshot(self, title: str = ..., log_scale: bool = ..., *, backend: Literal["matplotlib"]) -> MplFigure: ...

    def snapshot(
        self,
        title: str = "Portfolio Summary",
        log_scale: bool = False,
        *,
        backend: Backend | None = None,
    ) -> Figure:
        """Create a comprehensive dashboard with multiple plots for portfolio analysis.

        This function generates a three-panel plot showing:
        1. Cumulative returns over time
        2. Drawdowns over time
        3. Monthly returns over time

        This provides a complete visual summary of portfolio performance.

        Args:
            title: Accepted for backward compatibility but not used — the
                chart titles itself from the assets it plots.
            log_scale: Whether to use logarithmic scale for cumulative returns.
                Defaults to False.
            backend: Renderer to use. Defaults to the ambient selection.

        Returns:
            Figure: A three-panel dashboard.

        Example:
            >>> import polars as pl
            >>> from jquantstats import Data
            >>> # minimal demo dataset with a Date column and one asset
            >>> returns = pl.DataFrame({
            ...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            ...     "Asset": [0.01, -0.02, 0.03],
            ... }).with_columns(pl.col("Date").str.to_date())
            >>> data = Data.from_returns(returns=returns)
            >>> fig = data.plots.snapshot(title="My Portfolio Performance")
            >>> # Optional: display the interactive figure
            >>> fig.show()  # doctest: +SKIP

        """
        return render(data_snapshot_spec(self._data, log_scale=log_scale), backend)
