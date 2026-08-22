"""Spec builders for the Monte Carlo charts.

Both bootstrap from the trailing return history: `montecarlo_spec` draws the
simulated paths themselves, `montecarlo_distribution_spec` reduces each path to
one metric and shows where the observed value falls among them.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .._spec import Axis, FigureSpec, HistogramSeries, HoverSpec, LineSeries, Panel, RefLine
from .._style import hex_to_rgba, ticker_colors

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike

__all__ = ["METRICS", "montecarlo_distribution_spec", "montecarlo_spec"]

#: Metrics a simulated path can be reduced to, and how each is labelled.
METRICS = {"sharpe": "Sharpe Ratio", "drawdown": "Max Drawdown", "cagr": "CAGR"}

# Simulated paths are drawn very faintly: the point is the shape of the bundle,
# not any individual path. The observed path is heavier so it stands clear.
_SIM_ALPHA = 0.12
_SIM_WIDTH = 1
_OBSERVED_WIDTH = 2.5

_OVERLAY_OPACITY = 0.6
_PERIODS_PER_YEAR = 252.0

# Fixed so a chart is reproducible: the same data always yields the same fan.
_SEED = 42


def _split_columns(frame: pl.DataFrame) -> tuple[str, list[str]]:
    """Separate the date column from the value columns.

    Args:
        frame: A frame whose first column is the date axis.

    Returns:
        tuple[str, list[str]]: The date column name and every other column.

    """
    date_col = frame.columns[0]
    return date_col, [c for c in frame.columns if c != date_col]


def _require_positive(n: int, period: int) -> None:
    """Reject non-positive simulation counts or path lengths.

    Args:
        n: Number of simulations.
        period: Observations per simulation.

    Raises:
        ValueError: If either is not positive.

    """
    if n <= 0:
        raise ValueError("n must be a positive integer")  # noqa: TRY003
    if period <= 0:
        raise ValueError("period must be a positive integer")  # noqa: TRY003


def montecarlo_spec(
    data: DataLike,
    n: int,
    period: int,
    title: str,
    figsize: tuple[int, int] | None,
) -> FigureSpec:
    """Describe the Monte Carlo fan chart.

    Args:
        data: The dataset to plot.
        n: Number of simulated paths per asset.
        period: Observations per path.
        title: Chart title.
        figsize: Optional ``(width, height)`` in pixels.

    Returns:
        FigureSpec: *n* faint simulated paths per asset, with the observed
        path drawn over them.

    Raises:
        ValueError: If *n* or *period* is not positive.

    """
    _require_positive(n, period)

    df = data.all
    date_col, tickers = _split_columns(df)
    colors = ticker_colors(tickers)

    sample_len = min(period, df.height)
    dates = df[date_col].tail(sample_len).to_list()
    rng = np.random.default_rng(seed=_SEED)

    lines = []
    for ticker in tickers:
        trailing = (
            df[ticker].tail(sample_len - 1).fill_null(0.0).cast(pl.Float64).to_numpy()
            if sample_len > 1
            else np.array([], dtype=np.float64)
        )

        for i in range(n):
            draws = (
                rng.choice(trailing, size=sample_len - 1, replace=True)
                if sample_len > 1
                else np.array([], dtype=np.float64)
            )
            lines.append(
                LineSeries(
                    name=f"{ticker} Sim",
                    x=dates,
                    y=np.cumprod(np.concatenate(([1.0], 1.0 + draws))),
                    color=hex_to_rgba(colors[ticker], alpha=_SIM_ALPHA),
                    width=_SIM_WIDTH,
                    # One legend entry for the whole bundle, not n of them.
                    legend_group=f"{ticker}_sim",
                    show_legend=(i == 0),
                    hover=HoverSpec(
                        label=f"{ticker} Sim",
                        value_format="float2",
                        suffix="x",
                        date_header=False,
                        hide_extra=True,
                    ),
                )
            )

        lines.append(
            LineSeries(
                name=f"{ticker} Observed",
                x=dates,
                y=np.cumprod(np.concatenate(([1.0], 1.0 + trailing))),
                color=colors[ticker],
                width=_OBSERVED_WIDTH,
                legend_group=f"{ticker}_obs",
                hover=HoverSpec(
                    label=f"{ticker} Observed",
                    value_format="float2",
                    suffix="x",
                    date_header=False,
                    hide_extra=True,
                ),
            )
        )

    panel = Panel(lines=tuple(lines), yaxis=Axis(title="Cumulative Return", tick_format="float2"))
    return FigureSpec(title=title, panels=(panel,), figsize=figsize)


def _metric_value(returns: np.ndarray, metric_key: str) -> float:
    """Reduce one simulated return path to a single number.

    Args:
        returns: The path's per-period returns.
        metric_key: One of ``"sharpe"``, ``"drawdown"`` or ``"cagr"``.

    Returns:
        float: The metric's value for this path.

    """
    if metric_key == "sharpe":
        std = returns.std(ddof=1)
        return float(math.sqrt(_PERIODS_PER_YEAR) * returns.mean() / std) if std > 0 else 0.0
    if metric_key == "drawdown":
        path = np.cumprod(1.0 + returns)
        hwm = np.maximum.accumulate(path)
        dd = (path - hwm) / hwm
        return float(dd.min()) if dd.size else 0.0
    total_return = float(np.prod(1.0 + returns))
    return float(total_return ** (_PERIODS_PER_YEAR / len(returns)) - 1.0) if len(returns) > 0 else 0.0


def montecarlo_distribution_spec(
    data: DataLike,
    n: int,
    period: int,
    metric: str,
    title: str,
    figsize: tuple[int, int] | None,
) -> FigureSpec:
    """Describe the distribution of a metric across simulated paths.

    Args:
        data: The dataset to plot.
        n: Number of simulations per asset.
        period: Observations per simulation.
        metric: One of ``"sharpe"``, ``"drawdown"`` or ``"cagr"``.
        title: Chart title.
        figsize: Optional ``(width, height)`` in pixels.

    Returns:
        FigureSpec: One histogram per asset, each with a marker at the
        observed value.

    Raises:
        ValueError: If *n* or *period* is not positive, or *metric* is not one
            of the supported names.

    """
    _require_positive(n, period)

    metric_key = metric.strip().lower()
    if metric_key not in METRICS:
        raise ValueError("metric must be one of: sharpe, drawdown, cagr")  # noqa: TRY003

    df = data.all
    _, tickers = _split_columns(df)
    colors = ticker_colors(tickers)
    sample_len = min(period, df.height)
    rng = np.random.default_rng(seed=_SEED)

    histograms = []
    ref_lines = []
    for ticker in tickers:
        history = df[ticker].tail(sample_len).fill_null(0.0).cast(pl.Float64).to_numpy()
        if history.size == 0:  # pragma: no cover - a frame with no rows cannot reach a plot method
            continue

        histograms.append(
            HistogramSeries(
                name=ticker,
                values=[
                    _metric_value(rng.choice(history, size=sample_len, replace=True), metric_key) for _ in range(n)
                ],
                color=colors[ticker],
                opacity=_OVERLAY_OPACITY,
                hover=HoverSpec(
                    label=ticker,
                    value_format="float4",
                    date_header=False,
                    axis="x",
                    hide_extra=True,
                ),
            )
        )
        ref_lines.append(
            RefLine(
                value=_metric_value(history, metric_key),
                orientation="v",
                color=colors[ticker],
                width=2,
                dash="dash",
                label=f"{ticker} observed",
            )
        )

    panel = Panel(
        histograms=tuple(histograms),
        ref_lines=tuple(ref_lines),
        xaxis=Axis(title=METRICS[metric_key]),
        yaxis=Axis(title="Count"),
    )
    # No range selector: the x-axis is the metric's value, not time.
    return FigureSpec(
        title=title,
        panels=(panel,),
        figsize=figsize,
        date_range_selector=False,
        bar_mode="overlay",
    )
