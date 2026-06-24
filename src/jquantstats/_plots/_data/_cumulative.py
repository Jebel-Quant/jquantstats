"""Cumulative-return and equity-curve line charts."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl

from ._styling import _apply_base_layout, _apply_figsize, _ticker_colors

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
        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        prices = df.with_columns([(1.0 + pl.col(t)).cum_prod().alias(t) for t in tickers])

        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(
                go.Scatter(
                    x=prices[date_col],
                    y=prices[ticker],
                    mode="lines",
                    name=ticker,
                    line={"color": colors[ticker], "width": 2},
                    hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{ticker}: %{{y:.2f}}x",
                )
            )

        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text="Cumulative Return", tickformat=".2f")
        if log_scale:
            fig.update_yaxes(type="log")
        return fig

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
        benchmark_df = getattr(self._data, "benchmark", None)
        if benchmark_df is None:
            raise AttributeError("compare() requires benchmark data to be set")  # noqa: TRY003

        df = self._data.all
        date_col = df.columns[0]
        assets = list(self._data.returns.columns)
        benchmarks = list(benchmark_df.columns)

        series = assets + benchmarks
        colors = _ticker_colors(series)
        prices = df.with_columns([(1.0 + pl.col(col)).cum_prod().alias(col) for col in series])

        fig = go.Figure()
        for asset in assets:
            fig.add_trace(
                go.Scatter(
                    x=prices[date_col],
                    y=prices[asset],
                    mode="lines",
                    name=asset,
                    line={"color": colors[asset], "width": 2},
                    hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{asset}: %{{y:.2f}}x",
                )
            )
        for benchmark in benchmarks:
            fig.add_trace(
                go.Scatter(
                    x=prices[date_col],
                    y=prices[benchmark],
                    mode="lines",
                    name=benchmark,
                    line={"color": colors[benchmark], "width": 2.5, "dash": "dash"},
                    hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{benchmark}: %{{y:.2f}}x",
                )
            )

        _apply_base_layout(fig, title)
        _apply_figsize(fig, figsize)
        fig.update_yaxes(title_text="Cumulative Return", tickformat=".2f")
        return fig

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
        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        log_prices = df.with_columns([(1.0 + pl.col(t)).cum_prod().log(math.e).alias(t) for t in tickers])

        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(
                go.Scatter(
                    x=log_prices[date_col],
                    y=log_prices[ticker],
                    mode="lines",
                    name=ticker,
                    line={"color": colors[ticker], "width": 2},
                    hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{ticker}: %{{y:.4f}}",
                )
            )

        _apply_base_layout(fig, title)
        _apply_figsize(fig, figsize)
        fig.update_yaxes(title_text="Log Return")
        return fig

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
        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        if compounded:
            equity = df.with_columns([(start_balance * (1.0 + pl.col(t)).cum_prod()).alias(t) for t in tickers])
        else:
            equity = df.with_columns([(start_balance * (1.0 + pl.col(t).cum_sum())).alias(t) for t in tickers])

        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(
                go.Scatter(
                    x=equity[date_col],
                    y=equity[ticker],
                    mode="lines",
                    name=ticker,
                    line={"color": colors[ticker], "width": 2},
                    hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{ticker}: $%{{y:,.0f}}",
                )
            )

        _apply_base_layout(fig, title)
        fig.update_yaxes(
            title_text=f"Portfolio Value (starting ${start_balance:,.0f})",
            tickprefix="$",
            tickformat=",.0f",
        )
        return fig
