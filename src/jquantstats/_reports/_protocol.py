"""Protocols describing the minimal interfaces required by the _reports subpackage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import plotly.graph_objects as go
import polars as pl

from jquantstats._protocol import DataLike, StatsLike

__all__ = ["DataLike", "PlotsLike", "PortfolioLike", "StatsLike"]


@runtime_checkable
class PlotsLike(Protocol):  # pragma: no cover
    """Structural interface for the portfolio plots facade used by `Report`."""

    def snapshot(self) -> go.Figure:
        """NAV + drawdown snapshot figure."""
        ...

    def rolling_sharpe_plot(self) -> go.Figure:
        """Rolling Sharpe figure."""
        ...

    def rolling_volatility_plot(self) -> go.Figure:
        """Rolling volatility figure."""
        ...

    def annual_sharpe_plot(self) -> go.Figure:
        """Annual Sharpe figure."""
        ...

    def monthly_returns_heatmap(self) -> go.Figure:
        """Monthly returns heatmap figure."""
        ...

    def correlation_heatmap(self) -> go.Figure:
        """Correlation heatmap figure."""
        ...

    def lead_lag_ir_plot(self) -> go.Figure:
        """Lead/lag IR figure."""
        ...

    def trading_cost_impact_plot(self) -> go.Figure:
        """Trading cost impact figure."""
        ...


@runtime_checkable
class PortfolioLike(Protocol):  # pragma: no cover
    """Structural interface required by the `Report` class.

    Any object satisfying this protocol can be passed as ``portfolio`` without a
    concrete dependency on `Portfolio`.
    """

    prices: pl.DataFrame
    aum: float

    @property
    def assets(self) -> list[str]:
        """Asset names."""
        ...

    @property
    def plots(self) -> PlotsLike:
        """Portfolio plots facade."""
        ...

    @property
    def stats(self) -> StatsLike:
        """Statistics facade."""
        ...

    def turnover_summary(self) -> pl.DataFrame:
        """Turnover summary DataFrame."""
        ...
