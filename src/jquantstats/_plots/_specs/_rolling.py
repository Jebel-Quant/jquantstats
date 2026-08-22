"""Spec builders for the rolling-window and per-year risk charts.

Both facades are served from here. `Data.plots` computes its own rolling
metrics from the returns frame; `Portfolio.plots` asks its stats facade for
them. The marks are the same either way, which is why they share a module.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

from jquantstats.exceptions import NoBenchmarkError

from .._spec import Axis, BarSeries, Dash, FigureSpec, HoverSpec, LineSeries, Panel, RefLine, TickFormat
from .._style import ticker_colors

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike

    from .._protocol import PortfolioLike

__all__ = [
    "annual_sharpe_spec",
    "portfolio_rolling_sharpe_spec",
    "portfolio_rolling_volatility_spec",
    "rolling_beta_expr",
    "rolling_beta_spec",
    "rolling_sharpe_spec",
    "rolling_sortino_spec",
    "rolling_volatility_spec",
    "validate_window",
]

# Rolling metrics are noisier than the cumulative curves, so they are drawn
# finer to keep several overlapping series legible.
_ROLLING_WIDTH = 1.5

# The portfolio facade's rolling charts are finer still and take the backend's
# palette rather than naming colours.
_PORTFOLIO_WIDTH = 1


def _split_columns(frame: pl.DataFrame) -> tuple[str, list[str]]:
    """Separate the date column from the value columns.

    Args:
        frame: A frame whose first column is the date axis.

    Returns:
        tuple[str, list[str]]: The date column name and every other column.

    """
    date_col = frame.columns[0]
    return date_col, [c for c in frame.columns if c != date_col]


def validate_window(window: int) -> None:
    """Reject a non-positive or non-integer rolling window.

    Lives in the builder rather than a renderer so both backends reject the
    same inputs with the same message.

    Args:
        window: The candidate rolling-window size.

    Raises:
        ValueError: If *window* is not a positive integer.

    """
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be a positive integer, got {window!r}")  # noqa: TRY003


def rolling_beta_expr(asset: str, bench_col: str, window: int) -> pl.Expr:
    """Trailing-window OLS beta of *asset* against *bench_col*.

    Beta is ``cov(asset, bench) / var(bench)``, expanded into rolling means so
    the whole estimate is a single Polars expression.

    Args:
        asset: Asset column name.
        bench_col: Benchmark column name.
        window: Trailing window size in rows.

    Returns:
        pl.Expr: An expression aliased ``beta``.

    """
    mean_x = pl.col(asset).rolling_mean(window_size=window)
    mean_y = pl.col(bench_col).rolling_mean(window_size=window)
    mean_xy = (pl.col(asset) * pl.col(bench_col)).rolling_mean(window_size=window)
    mean_y2 = (pl.col(bench_col) ** 2).rolling_mean(window_size=window)
    return ((mean_xy - mean_x * mean_y) / (mean_y2 - mean_y**2)).alias("beta")


def _metric_lines(
    frame: pl.DataFrame,
    date_col: str,
    tickers: list[str],
    value_format: TickFormat,
) -> tuple[LineSeries, ...]:
    """Build one line per ticker from an already-computed metric frame.

    Args:
        frame: The frame holding the rolling metric, one column per ticker.
        date_col: Name of the date column.
        tickers: Columns to draw, in order.
        value_format: How tooltip values are rendered.

    Returns:
        tuple[LineSeries, ...]: One series per ticker.

    """
    colors = ticker_colors(tickers)
    return tuple(
        LineSeries(
            name=ticker,
            x=frame[date_col],
            y=frame[ticker],
            color=colors[ticker],
            width=_ROLLING_WIDTH,
            hover=HoverSpec(label=ticker, value_format=value_format, date_header=False),
        )
        for ticker in tickers
    )


def rolling_sharpe_spec(data: DataLike, rolling_period: int, periods_per_year: int, title: str) -> FigureSpec:
    """Describe the rolling Sharpe-ratio chart.

    Args:
        data: The dataset to plot.
        rolling_period: Trailing window size in rows.
        periods_per_year: Annualisation factor.
        title: Chart title.

    Returns:
        FigureSpec: One line per column, with a break-even marker.

    """
    df = data.all
    date_col, tickers = _split_columns(df)
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

    panel = Panel(
        lines=_metric_lines(rolling, date_col, tickers, "float2"),
        ref_lines=(RefLine(value=0, dash="dash"),),
        yaxis=Axis(title=f"Sharpe ({rolling_period}-period rolling)"),
    )
    return FigureSpec(title=title, panels=(panel,))


def rolling_sortino_spec(data: DataLike, rolling_period: int, periods_per_year: int, title: str) -> FigureSpec:
    """Describe the rolling Sortino-ratio chart.

    Sortino divides by downside deviation rather than total volatility, so only
    negative returns contribute to the denominator.

    Args:
        data: The dataset to plot.
        rolling_period: Trailing window size in rows.
        periods_per_year: Annualisation factor.
        title: Chart title.

    Returns:
        FigureSpec: One line per column, with a break-even marker.

    """
    df = data.all
    date_col, tickers = _split_columns(df)
    scale = math.sqrt(periods_per_year)

    exprs = []
    for t in tickers:
        mean_r = pl.col(t).rolling_mean(window_size=rolling_period)
        downside = (
            pl.when(pl.col(t) < 0).then(pl.col(t) ** 2).otherwise(0.0).rolling_mean(window_size=rolling_period).sqrt()
        )
        exprs.append((mean_r / downside * scale).alias(t))

    rolling = df.with_columns(exprs)

    panel = Panel(
        lines=_metric_lines(rolling, date_col, tickers, "float2"),
        ref_lines=(RefLine(value=0, dash="dash"),),
        yaxis=Axis(title=f"Sortino ({rolling_period}-period rolling)"),
    )
    return FigureSpec(title=title, panels=(panel,))


def rolling_volatility_spec(data: DataLike, rolling_period: int, periods_per_year: int, title: str) -> FigureSpec:
    """Describe the rolling-volatility chart.

    Args:
        data: The dataset to plot.
        rolling_period: Trailing window size in rows.
        periods_per_year: Annualisation factor.
        title: Chart title.

    Returns:
        FigureSpec: One line per column. Volatility cannot be negative, so
        there is no break-even marker.

    """
    df = data.all
    date_col, tickers = _split_columns(df)
    scale = math.sqrt(periods_per_year)

    rolling = df.with_columns([(pl.col(t).rolling_std(window_size=rolling_period) * scale).alias(t) for t in tickers])

    panel = Panel(
        lines=_metric_lines(rolling, date_col, tickers, "percent2"),
        yaxis=Axis(title=f"Volatility ({rolling_period}-period rolling)", tick_format="percent0"),
    )
    return FigureSpec(title=title, panels=(panel,))


def _beta_assets(data: DataLike, df: pl.DataFrame, date_col: str, bench_col: str) -> list[str]:
    """Asset columns to plot beta for.

    Prefers the explicit ``returns`` frame when the data exposes one, and
    otherwise falls back to every column that is neither the date nor the
    benchmark.

    Args:
        data: The dataset being plotted.
        df: The combined index/returns/benchmark frame.
        date_col: Name of the date column.
        bench_col: Name of the benchmark column.

    Returns:
        list[str]: The asset column names.

    """
    returns_df = getattr(data, "returns", None)
    if returns_df is not None:
        return list(returns_df.columns)
    return [c for c in df.columns if c != date_col and c != bench_col]


def rolling_beta_spec(
    data: DataLike,
    rolling_period: int,
    rolling_period2: int | None,
    title: str,
    figsize: tuple[int, int] | None,
) -> FigureSpec:
    """Describe the rolling-beta chart.

    Args:
        data: The dataset to plot. Must carry a benchmark.
        rolling_period: Primary trailing window size.
        rolling_period2: Optional second window, overlaid dashed, or None.
        title: Chart title.
        figsize: Optional ``(width, height)`` in pixels.

    Returns:
        FigureSpec: One line per asset per window, with a marker at beta = 1.

    Raises:
        NoBenchmarkError: If the data carries no benchmark columns.

    """
    df = data.all
    date_col, _ = _split_columns(df)

    benchmark_df = getattr(data, "benchmark", None)
    if benchmark_df is None:
        raise NoBenchmarkError

    bench_col = benchmark_df.columns[0]
    assets = _beta_assets(data, df, date_col, bench_col)
    colors = ticker_colors(assets)
    windows = [w for w in (rolling_period, rolling_period2) if w is not None]

    lines = []
    for asset in assets:
        # The shorter window is solid and the longer one dashed, so the pair
        # for one asset reads as the same colour at two horizons.
        dashes: tuple[Dash, ...] = ("solid", "dash")
        for window, dash in zip(windows, dashes, strict=False):
            beta_df = df.with_columns(rolling_beta_expr(asset, bench_col, window))
            label = f"{asset} ({window}d)"
            lines.append(
                LineSeries(
                    name=label,
                    x=beta_df[date_col],
                    y=beta_df["beta"],
                    color=colors[asset],
                    width=_ROLLING_WIDTH,
                    dash=dash,
                    hover=HoverSpec(label=label, value_format="float2", date_header=False),
                )
            )

    panel = Panel(
        lines=tuple(lines),
        # Beta of 1 means moving with the benchmark, which is the reference
        # worth marking rather than zero.
        ref_lines=(RefLine(value=1, dash="dash"),),
        yaxis=Axis(title="Beta"),
    )
    return FigureSpec(title=title, panels=(panel,), figsize=figsize)


def _portfolio_metric_lines(rolling: pl.DataFrame) -> tuple[LineSeries, ...]:
    """Build one line per non-date column of a portfolio metric frame.

    These charts name no colours, taking the backend's palette instead, and
    carry no tooltips.

    Args:
        rolling: A frame with an optional ``date`` column and one column per
            asset.

    Returns:
        tuple[LineSeries, ...]: One series per asset column.

    """
    dates = rolling["date"] if "date" in rolling.columns else None
    return tuple(
        LineSeries(name=col, x=dates, y=rolling[col], width=_PORTFOLIO_WIDTH)
        for col in rolling.columns
        if col != "date"
    )


def portfolio_rolling_sharpe_spec(portfolio: PortfolioLike, window: int) -> FigureSpec:
    """Describe the portfolio's rolling Sharpe-ratio chart.

    Args:
        portfolio: The portfolio to plot.
        window: Rolling-window size in periods.

    Returns:
        FigureSpec: One line per asset, with a break-even marker.

    Raises:
        ValueError: If *window* is not a positive integer.

    """
    validate_window(window)
    panel = Panel(
        lines=_portfolio_metric_lines(portfolio.stats.rolling_sharpe(rolling_period=window)),
        ref_lines=(RefLine(value=0, dash="dash"),),
        yaxis=Axis(title="Sharpe ratio"),
    )
    return FigureSpec(title=f"Rolling Sharpe Ratio ({window}-period window)", panels=(panel,))


def portfolio_rolling_volatility_spec(portfolio: PortfolioLike, window: int) -> FigureSpec:
    """Describe the portfolio's rolling-volatility chart.

    Args:
        portfolio: The portfolio to plot.
        window: Rolling-window size in periods.

    Returns:
        FigureSpec: One line per asset. Volatility cannot be negative, so there
        is no break-even marker.

    Raises:
        ValueError: If *window* is not a positive integer.

    """
    validate_window(window)
    panel = Panel(
        lines=_portfolio_metric_lines(portfolio.stats.rolling_volatility(rolling_period=window)),
        yaxis=Axis(title="Annualised volatility"),
    )
    return FigureSpec(title=f"Rolling Volatility ({window}-period window)", panels=(panel,))


def annual_sharpe_spec(portfolio: PortfolioLike) -> FigureSpec:
    """Describe the per-calendar-year Sharpe breakdown.

    Args:
        portfolio: The portfolio to plot.

    Returns:
        FigureSpec: A grouped bar chart, one bar per year per asset.

    """
    breakdown = portfolio.stats.annual_breakdown()
    sharpe_rows = breakdown.filter(pl.col("metric") == "sharpe")
    asset_cols = [c for c in sharpe_rows.columns if c not in ("year", "metric")]

    panel = Panel(
        bars=tuple(BarSeries(name=asset, x=sharpe_rows["year"], y=sharpe_rows[asset]) for asset in asset_cols),
        ref_lines=(RefLine(value=0),),
        xaxis=Axis(title="Year"),
        yaxis=Axis(title="Sharpe ratio"),
    )
    return FigureSpec(
        title="Annual Sharpe Ratio by Year",
        panels=(panel,),
        # Calendar years, so no range selector; and this chart never fixed a
        # height, letting the container size it.
        date_range_selector=False,
        height=None,
        bar_mode="group",
    )
