"""Rolling risk/return metric line charts (Sharpe, Sortino, volatility, beta)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl

from jquantstats.exceptions import NoBenchmarkError

from ._styling import _apply_base_layout, _apply_figsize, _ticker_colors

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike


class _RollingPlotsMixin:
    """Rolling-window metric plots for :class:`DataPlots`."""

    __slots__ = ()

    _data: DataLike

    def rolling_sharpe(
        self,
        rolling_period: int = 126,
        periods_per_year: int = 252,
        title: str = "Rolling Sharpe Ratio",
    ) -> go.Figure:
        """Rolling annualised Sharpe ratio over time.

        Computes ``rolling_mean / rolling_std * sqrt(periods_per_year)`` with a
        trailing window of *rolling_period* observations for every column in the
        dataset (assets and benchmark when present).

        Args:
            rolling_period: Trailing window size. Defaults to 126 (6 months).
            periods_per_year: Annualisation factor. Defaults to 252.
            title: Chart title. Defaults to ``"Rolling Sharpe Ratio"``.

        Returns:
            go.Figure: Interactive Plotly line chart.

        """
        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)
        scale = math.sqrt(periods_per_year)

        rolling = df.with_columns(
            [
                (
                    pl.col(t).rolling_mean(window_size=rolling_period)
                    / pl.col(t).rolling_std(window_size=rolling_period)
                    * scale
                ).alias(t)
                for t in tickers
            ]
        )

        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(
                go.Scatter(
                    x=rolling[date_col],
                    y=rolling[ticker],
                    mode="lines",
                    name=ticker,
                    line={"color": colors[ticker], "width": 1.5},
                    hovertemplate=f"{ticker}: %{{y:.2f}}",
                )
            )

        fig.add_hline(y=0, line_width=1, line_color="gray", line_dash="dash")
        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text=f"Sharpe ({rolling_period}-period rolling)")
        return fig

    def rolling_sortino(
        self,
        rolling_period: int = 126,
        periods_per_year: int = 252,
        title: str = "Rolling Sortino Ratio",
    ) -> go.Figure:
        """Rolling annualised Sortino ratio over time.

        Computes ``rolling_mean / rolling_downside_std * sqrt(periods_per_year)``
        where downside deviation considers only negative returns.

        Args:
            rolling_period: Trailing window size. Defaults to 126 (6 months).
            periods_per_year: Annualisation factor. Defaults to 252.
            title: Chart title. Defaults to ``"Rolling Sortino Ratio"``.

        Returns:
            go.Figure: Interactive Plotly line chart.

        """
        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)
        scale = math.sqrt(periods_per_year)

        exprs = []
        for t in tickers:
            mean_r = pl.col(t).rolling_mean(window_size=rolling_period)
            downside = (
                pl.when(pl.col(t) < 0)
                .then(pl.col(t) ** 2)
                .otherwise(0.0)
                .rolling_mean(window_size=rolling_period)
                .sqrt()
            )
            exprs.append((mean_r / downside * scale).alias(t))

        rolling = df.with_columns(exprs)

        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(
                go.Scatter(
                    x=rolling[date_col],
                    y=rolling[ticker],
                    mode="lines",
                    name=ticker,
                    line={"color": colors[ticker], "width": 1.5},
                    hovertemplate=f"{ticker}: %{{y:.2f}}",
                )
            )

        fig.add_hline(y=0, line_width=1, line_color="gray", line_dash="dash")
        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text=f"Sortino ({rolling_period}-period rolling)")
        return fig

    def rolling_volatility(
        self,
        rolling_period: int = 126,
        periods_per_year: int = 252,
        title: str = "Rolling Volatility",
    ) -> go.Figure:
        """Rolling annualised volatility over time.

        Computes ``rolling_std * sqrt(periods_per_year)`` for every column in
        the dataset.

        Args:
            rolling_period: Trailing window size. Defaults to 126 (6 months).
            periods_per_year: Annualisation factor. Defaults to 252.
            title: Chart title. Defaults to ``"Rolling Volatility"``.

        Returns:
            go.Figure: Interactive Plotly line chart.

        """
        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)
        scale = math.sqrt(periods_per_year)

        rolling = df.with_columns(
            [(pl.col(t).rolling_std(window_size=rolling_period) * scale).alias(t) for t in tickers]
        )

        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(
                go.Scatter(
                    x=rolling[date_col],
                    y=rolling[ticker],
                    mode="lines",
                    name=ticker,
                    line={"color": colors[ticker], "width": 1.5},
                    hovertemplate=f"{ticker}: %{{y:.2%}}",
                )
            )

        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text=f"Volatility ({rolling_period}-period rolling)", tickformat=".0%")
        return fig

    def rolling_beta(
        self,
        rolling_period: int = 126,
        rolling_period2: int | None = 252,
        title: str = "Rolling Beta",
        figsize: tuple[int, int] | None = None,
    ) -> go.Figure:
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

        Returns:
            go.Figure: Interactive Plotly line chart.

        Raises:
            AttributeError: If no benchmark columns are present in the data.

        """
        df = self._data.all
        date_col = df.columns[0]

        benchmark_df = getattr(self._data, "benchmark", None)
        if benchmark_df is None:
            raise NoBenchmarkError

        bench_col = benchmark_df.columns[0]
        returns_df = getattr(self._data, "returns", None)
        assets = (
            list(returns_df.columns)
            if returns_df is not None
            else [c for c in df.columns if c != date_col and c != bench_col]
        )
        colors = _ticker_colors(assets)
        windows = [w for w in (rolling_period, rolling_period2) if w is not None]
        line_styles = ["solid", "dash"]

        fig = go.Figure()
        for asset in assets:
            for w, dash in zip(windows, line_styles, strict=False):
                mean_x = pl.col(asset).rolling_mean(window_size=w)
                mean_y = pl.col(bench_col).rolling_mean(window_size=w)
                mean_xy = (pl.col(asset) * pl.col(bench_col)).rolling_mean(window_size=w)
                mean_y2 = (pl.col(bench_col) ** 2).rolling_mean(window_size=w)
                beta_expr = ((mean_xy - mean_x * mean_y) / (mean_y2 - mean_y**2)).alias("beta")

                beta_df = df.with_columns(beta_expr)
                label = f"{asset} ({w}d)"
                fig.add_trace(
                    go.Scatter(
                        x=beta_df[date_col],
                        y=beta_df["beta"],
                        mode="lines",
                        name=label,
                        line={"color": colors[asset], "width": 1.5, "dash": dash},
                        hovertemplate=f"{label}: %{{y:.2f}}",
                    )
                )

        fig.add_hline(y=1, line_width=1, line_color="gray", line_dash="dash")
        _apply_base_layout(fig, title)
        _apply_figsize(fig, figsize)
        fig.update_yaxes(title_text="Beta")
        return fig
