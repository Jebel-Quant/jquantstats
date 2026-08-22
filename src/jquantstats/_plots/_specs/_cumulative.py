"""Spec builders for the cumulative-return and equity-curve charts.

Each builder takes a dataset and returns a `FigureSpec`. The arithmetic lives
here once; `jquantstats._plots._render` decides how it is drawn.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

from .._spec import Axis, Dash, FigureSpec, HoverSpec, LineSeries, Panel, TickFormat
from .._style import ticker_colors

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike

__all__ = [
    "compare_spec",
    "cumulative_returns_spec",
    "earnings_spec",
    "log_returns_spec",
]


def _split_columns(frame: pl.DataFrame) -> tuple[str, list[str]]:
    """Separate the date column from the value columns.

    Args:
        frame: A frame whose first column is the date axis.

    Returns:
        tuple[str, list[str]]: The date column name and every other column.

    """
    date_col = frame.columns[0]
    return date_col, [c for c in frame.columns if c != date_col]


def _lines(
    frame: pl.DataFrame,
    date_col: str,
    tickers: list[str],
    colors: dict[str, str],
    value_format: TickFormat,
    *,
    prefix: str = "",
    suffix: str = "",
    width: float = 2,
    dash: Dash = "solid",
) -> list[LineSeries]:
    """Build one line series per ticker, all sharing a style.

    Args:
        frame: The prepared frame holding the plotted values.
        date_col: Name of the date column.
        tickers: Columns to draw, in order.
        colors: Ticker to hex colour mapping.
        value_format: How tooltip values are rendered.
        prefix: Written before each tooltip value.
        suffix: Written after each tooltip value.
        width: Stroke width.
        dash: Stroke pattern.

    Returns:
        list[LineSeries]: One series per ticker, in the given order.

    """
    return [
        LineSeries(
            name=ticker,
            x=frame[date_col],
            y=frame[ticker],
            color=colors[ticker],
            width=width,
            dash=dash,
            hover=HoverSpec(label=ticker, value_format=value_format, prefix=prefix, suffix=suffix),
        )
        for ticker in tickers
    ]


def cumulative_returns_spec(data: DataLike, title: str, log_scale: bool) -> FigureSpec:
    """Describe the cumulative compounded-returns chart.

    Args:
        data: The dataset to plot.
        title: Chart title.
        log_scale: Use a logarithmic y-axis.

    Returns:
        FigureSpec: One line per column of ``data.all``.

    """
    df = data.all
    date_col, tickers = _split_columns(df)
    prices = df.with_columns([(1.0 + pl.col(t)).cum_prod().alias(t) for t in tickers])

    panel = Panel(
        # The "x" suffix reads the value as a growth multiple: "1.42x".
        lines=tuple(_lines(prices, date_col, tickers, ticker_colors(tickers), "float2", suffix="x")),
        yaxis=Axis(title="Cumulative Return", tick_format="float2", log=log_scale),
    )
    return FigureSpec(title=title, panels=(panel,))


def compare_spec(data: DataLike, title: str, figsize: tuple[int, int] | None) -> FigureSpec:
    """Describe the asset-versus-benchmark comparison chart.

    Benchmarks are drawn after the assets, slightly heavier and dashed, so they
    read as the reference rather than as another asset.

    Args:
        data: The dataset to plot. Must carry benchmark columns.
        title: Chart title.
        figsize: Optional ``(width, height)`` in pixels.

    Returns:
        FigureSpec: Asset lines followed by benchmark lines.

    Raises:
        AttributeError: If no benchmark data is available.

    """
    benchmark_df = getattr(data, "benchmark", None)
    if benchmark_df is None:
        raise AttributeError("compare() requires benchmark data to be set")  # noqa: TRY003

    df = data.all
    date_col, _ = _split_columns(df)
    assets = list(data.returns.columns)
    benchmarks = list(benchmark_df.columns)

    colors = ticker_colors(assets + benchmarks)
    prices = df.with_columns([(1.0 + pl.col(col)).cum_prod().alias(col) for col in assets + benchmarks])

    panel = Panel(
        lines=(
            *_lines(prices, date_col, assets, colors, "float2", suffix="x"),
            *_lines(prices, date_col, benchmarks, colors, "float2", suffix="x", width=2.5, dash="dash"),
        ),
        yaxis=Axis(title="Cumulative Return", tick_format="float2"),
    )
    return FigureSpec(title=title, panels=(panel,), figsize=figsize)


def log_returns_spec(data: DataLike, title: str, figsize: tuple[int, int] | None) -> FigureSpec:
    """Describe the cumulative log-returns chart.

    Args:
        data: The dataset to plot.
        title: Chart title.
        figsize: Optional ``(width, height)`` in pixels.

    Returns:
        FigureSpec: One line per column, on a linear axis of log values.

    """
    df = data.all
    date_col, tickers = _split_columns(df)
    log_prices = df.with_columns([(1.0 + pl.col(t)).cum_prod().log(math.e).alias(t) for t in tickers])

    panel = Panel(
        lines=tuple(_lines(log_prices, date_col, tickers, ticker_colors(tickers), "float4")),
        yaxis=Axis(title="Log Return"),
    )
    return FigureSpec(title=title, panels=(panel,), figsize=figsize)


def earnings_spec(data: DataLike, start_balance: float, title: str, compounded: bool) -> FigureSpec:
    """Describe the dollar equity curve.

    Args:
        data: The dataset to plot.
        start_balance: Starting portfolio value in currency units.
        title: Chart title.
        compounded: Compound the returns; when False they are summed.

    Returns:
        FigureSpec: One line per column, scaled to *start_balance*.

    """
    df = data.all
    date_col, tickers = _split_columns(df)

    if compounded:
        equity = df.with_columns([(start_balance * (1.0 + pl.col(t)).cum_prod()).alias(t) for t in tickers])
    else:
        equity = df.with_columns([(start_balance * (1.0 + pl.col(t).cum_sum())).alias(t) for t in tickers])

    panel = Panel(
        lines=tuple(_lines(equity, date_col, tickers, ticker_colors(tickers), "currency0", prefix="$")),
        yaxis=Axis(
            title=f"Portfolio Value (starting ${start_balance:,.0f})",
            tick_format="currency0",
            tick_prefix="$",
        ),
    )
    return FigureSpec(title=title, panels=(panel,))
