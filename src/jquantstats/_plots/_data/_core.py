"""The :class:`DataPlots` facade combining the plot-family mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from ._cumulative import _CumulativePlotsMixin
from ._dashboard import _plot_performance_dashboard
from ._distribution import _DistributionPlotsMixin
from ._drawdown import _DrawdownPlotsMixin
from ._montecarlo import _MonteCarloPlotsMixin
from ._periodic import _PeriodicPlotsMixin
from ._rolling import _RollingPlotsMixin

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike


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

    def snapshot(self, title: str = "Portfolio Summary", log_scale: bool = False) -> go.Figure:
        """Create a comprehensive dashboard with multiple plots for portfolio analysis.

        This function generates a three-panel plot showing:
        1. Cumulative returns over time
        2. Drawdowns over time
        3. Daily returns over time

        This provides a complete visual summary of portfolio performance.

        Args:
            title (str, optional): Title of the plot. Defaults to "Portfolio Summary".
            compounded (bool, optional): Whether to use compounded returns. Defaults to True.
            log_scale (bool, optional): Whether to use logarithmic scale for cumulative returns.
                Defaults to False.

        Returns:
            go.Figure: A Plotly figure object containing the dashboard.

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
        fig = _plot_performance_dashboard(returns=self._data.all, log_scale=log_scale)
        return fig
