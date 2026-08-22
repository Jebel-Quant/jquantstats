"""Cumulative-return and equity-curve line charts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from .._render import render_plotly
from .._specs import compare_spec, cumulative_returns_spec, earnings_spec, log_returns_spec

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike


class _CumulativePlotsMixin:
    """Cumulative-return and equity-curve plots for :class:`DataPlots`."""

    __slots__ = ()

    _data: DataLike

    def returns(self, title: str = "Cumulative Returns", log_scale: bool = False) -> go.Figure:
        """Cumulative compounded returns over time.

        Plots ``(1 + r).cumprod()`` for every column in the dataset (including
        benchmark when present).

        Args:
            title: Chart title. Defaults to ``"Cumulative Returns"``.
            log_scale: Use a logarithmic y-axis. Defaults to False.

        Returns:
            go.Figure: Interactive Plotly line chart.

        """
        return render_plotly(cumulative_returns_spec(self._data, title=title, log_scale=log_scale))

    def compare(self, title: str = "Comparison vs Benchmark", figsize: tuple[int, int] | None = None) -> go.Figure:
        """Compare cumulative returns of each asset against the benchmark.

        Args:
            title: Chart title. Defaults to ``"Comparison vs Benchmark"``.
            figsize: Optional ``(width, height)`` in pixels.

        Returns:
            go.Figure: Interactive Plotly line chart.

        Raises:
            AttributeError: If no benchmark data is available.

        """
        return render_plotly(compare_spec(self._data, title=title, figsize=figsize))

    def log_returns(self, title: str = "Log Returns", figsize: tuple[int, int] | None = None) -> go.Figure:
        """Cumulative log returns over time.

        Plots ``log((1 + r).cumprod())`` — the natural log of the compounded
        growth factor — which linearises exponential growth and makes
        multi-asset comparisons on a common scale.

        Args:
            title: Chart title. Defaults to ``"Log Returns"``.
            figsize: Optional ``(width, height)`` in pixels.

        Returns:
            go.Figure: Interactive Plotly line chart.

        """
        return render_plotly(log_returns_spec(self._data, title=title, figsize=figsize))

    def earnings(
        self,
        start_balance: float = 1e5,
        title: str = "Portfolio Earnings",
        compounded: bool = True,
    ) -> go.Figure:
        """Dollar equity curve showing portfolio value over time.

        Scales cumulative returns by *start_balance* so the y-axis reflects
        an absolute portfolio value rather than a dimensionless growth factor.

        Args:
            start_balance: Starting portfolio value in currency units.
                Defaults to 100 000.
            title: Chart title. Defaults to ``"Portfolio Earnings"``.
            compounded: Use compounded returns (``cumprod``). When False uses
                cumulative sum. Defaults to True.

        Returns:
            go.Figure: Interactive Plotly line chart.

        """
        return render_plotly(earnings_spec(self._data, start_balance=start_balance, title=title, compounded=compounded))
