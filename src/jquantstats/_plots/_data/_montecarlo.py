"""Monte Carlo simulation charts (fan chart and metric distribution)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import polars as pl

from ._styling import _apply_base_layout, _apply_figsize, _hex_to_rgba, _ticker_colors

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike


class _MonteCarloPlotsMixin:
    """Monte Carlo simulation plots for :class:`DataPlots`."""

    __slots__ = ()

    _data: DataLike

    def montecarlo(
        self,
        n: int = 100,
        period: int = 252,
        title: str = "Monte Carlo Simulation",
        figsize: tuple[int, int] | None = None,
    ) -> go.Figure:
        """Fan chart of Monte Carlo simulated cumulative return paths.

        For each asset column, draws ``n`` bootstrapped paths sampled with
        replacement from historical returns and overlays the observed path for
        the trailing *period* observations.

        Args:
            n: Number of simulated paths per asset. Defaults to 100.
            period: Number of observations per path. Defaults to 252.
            title: Chart title. Defaults to ``"Monte Carlo Simulation"``.
            figsize: Optional figure ``(width, height)`` in pixels.

        Returns:
            go.Figure: Interactive Plotly fan chart.

        """
        if n <= 0:
            raise ValueError("n must be a positive integer")  # noqa: TRY003
        if period <= 0:
            raise ValueError("period must be a positive integer")  # noqa: TRY003

        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        sample_len = min(period, df.height)
        dates = df[date_col].tail(sample_len).to_list()
        rng = np.random.default_rng(seed=42)

        fig = go.Figure()
        for ticker in tickers:
            trailing_returns = (
                df[ticker].tail(sample_len - 1).fill_null(0.0).cast(pl.Float64).to_numpy()
                if sample_len > 1
                else np.array([], dtype=np.float64)
            )

            for i in range(n):
                draws = (
                    rng.choice(trailing_returns, size=sample_len - 1, replace=True)
                    if sample_len > 1
                    else np.array([], dtype=np.float64)
                )
                sim_path = np.cumprod(np.concatenate(([1.0], 1.0 + draws)))
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=sim_path,
                        mode="lines",
                        name=f"{ticker} Sim",
                        legendgroup=f"{ticker}_sim",
                        showlegend=(i == 0),
                        line={"color": _hex_to_rgba(colors[ticker], alpha=0.12), "width": 1},
                        hovertemplate=f"{ticker} Sim: %{{y:.2f}}x<extra></extra>",
                    )
                )

            observed_path = np.cumprod(np.concatenate(([1.0], 1.0 + trailing_returns)))
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=observed_path,
                    mode="lines",
                    name=f"{ticker} Observed",
                    legendgroup=f"{ticker}_obs",
                    line={"color": colors[ticker], "width": 2.5},
                    hovertemplate=f"{ticker} Observed: %{{y:.2f}}x<extra></extra>",
                )
            )

        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text="Cumulative Return", tickformat=".2f")
        _apply_figsize(fig, figsize)
        return fig

    def montecarlo_distribution(
        self,
        n: int = 1000,
        period: int = 252,
        metric: str = "sharpe",
        title: str = "Monte Carlo Distribution",
        figsize: tuple[int, int] | None = None,
    ) -> go.Figure:
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

        Returns:
            go.Figure: Interactive Plotly histogram figure.

        """
        if n <= 0:
            raise ValueError("n must be a positive integer")  # noqa: TRY003
        if period <= 0:
            raise ValueError("period must be a positive integer")  # noqa: TRY003

        metric_key = metric.strip().lower()
        if metric_key not in {"sharpe", "drawdown", "cagr"}:
            raise ValueError("metric must be one of: sharpe, drawdown, cagr")  # noqa: TRY003
        periods_per_year = 252.0

        def _metric_value(returns: np.ndarray) -> float:
            """Compute the selected metric for a simulated return path."""
            if metric_key == "sharpe":
                std = returns.std(ddof=1)
                return float(math.sqrt(periods_per_year) * returns.mean() / std) if std > 0 else 0.0
            if metric_key == "drawdown":
                path = np.cumprod(1.0 + returns)
                hwm = np.maximum.accumulate(path)
                dd = (path - hwm) / hwm
                return float(dd.min()) if dd.size else 0.0
            total_return = float(np.prod(1.0 + returns))
            return float(total_return ** (periods_per_year / len(returns)) - 1.0) if len(returns) > 0 else 0.0

        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)
        sample_len = min(period, df.height)
        rng = np.random.default_rng(seed=42)

        fig = go.Figure()
        for ticker in tickers:
            hist_returns = df[ticker].tail(sample_len).fill_null(0.0).cast(pl.Float64).to_numpy()
            if hist_returns.size == 0:  # pragma: no cover
                continue

            simulated_metrics = [
                _metric_value(rng.choice(hist_returns, size=sample_len, replace=True)) for _ in range(n)
            ]
            observed_metric = _metric_value(hist_returns)

            fig.add_trace(
                go.Histogram(
                    x=simulated_metrics,
                    name=ticker,
                    marker_color=colors[ticker],
                    opacity=0.6,
                    hovertemplate=f"{ticker}: %{{x:.4f}}<extra></extra>",
                )
            )
            fig.add_vline(
                x=observed_metric,
                line={"color": colors[ticker], "width": 2, "dash": "dash"},
                annotation_text=f"{ticker} observed",
                annotation_position="top right",
                annotation_font_size=10,
            )

        metric_title = {"sharpe": "Sharpe Ratio", "drawdown": "Max Drawdown", "cagr": "CAGR"}[metric_key]
        _apply_base_layout(fig, title, with_range_selector=False)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text=metric_title)
        fig.update_yaxes(title_text="Count")
        _apply_figsize(fig, figsize)
        return fig
