"""Spec builders for the drawdown charts.

Two views of the same thing: `drawdown_spec` plots the decline from the running
peak directly, while `drawdowns_periods_spec` leaves the equity curve alone and
shades the worst episodes on top of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from .._spec import Axis, Band, FigureSpec, HoverSpec, LineSeries, Panel, RefLine
from .._style import PALETTE, hex_to_rgba, ticker_colors

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike

__all__ = ["compute_drawdown_periods", "drawdown_spec", "drawdowns_periods_spec"]

# The underwater curve's fill, and the shading over a drawdown episode. Both are
# translucent so the line underneath stays readable.
_FILL_ALPHA = 0.3
_BAND_ALPHA = 0.2

# The equity curve in the periods chart is a single fixed colour rather than a
# palette entry: there is only ever one asset on it, so nothing to distinguish.
_EQUITY_COLOR = "#1f77b4"


def _split_columns(frame: pl.DataFrame) -> tuple[str, list[str]]:
    """Separate the date column from the value columns.

    Args:
        frame: A frame whose first column is the date axis.

    Returns:
        tuple[str, list[str]]: The date column name and every other column.

    """
    date_col = frame.columns[0]
    return date_col, [c for c in frame.columns if c != date_col]


def compute_drawdown_periods(prices: list[float], n: int) -> list[dict[str, Any]]:
    """Identify the *n* worst drawdown periods in a cumulative price series.

    A period runs from the point the series falls below its running peak until
    it recovers to it. Periods are ranked by depth, worst first.

    Args:
        prices: Cumulative price (NAV) values.
        n: Maximum number of periods to return.

    Returns:
        list[dict[str, Any]]: Dicts with ``start_idx``, ``end_idx``,
        ``valley_idx`` and ``max_drawdown`` (a fraction <= 0), worst first.

    """
    length = len(prices)
    hwm: list[float] = [0.0] * length
    hwm[0] = prices[0]
    for i in range(1, length):
        hwm[i] = max(hwm[i - 1], prices[i])

    in_dd = [prices[i] < hwm[i] for i in range(length)]
    periods: list[dict[str, Any]] = []
    i = 0
    while i < length:
        if not in_dd[i]:
            i += 1
            continue
        start = i
        while i < length and in_dd[i]:
            i += 1
        end = i - 1
        valley = start + min(range(end - start + 1), key=lambda k: prices[start + k])
        max_dd = (prices[valley] - hwm[valley]) / hwm[valley]
        periods.append({"start_idx": start, "end_idx": end, "valley_idx": valley, "max_drawdown": max_dd})

    periods.sort(key=lambda p: p["max_drawdown"])
    return periods[:n]


def drawdown_spec(data: DataLike, title: str) -> FigureSpec:
    """Describe the underwater equity curve.

    Args:
        data: The dataset to plot.
        title: Chart title.

    Returns:
        FigureSpec: One filled series per column, with a break-even line.

    """
    df = data.all
    date_col, tickers = _split_columns(df)
    colors = ticker_colors(tickers)
    prices = df.with_columns([(1.0 + pl.col(t)).cum_prod().alias(t) for t in tickers])

    lines = []
    for ticker in tickers:
        price_s = prices[ticker]
        hwm = price_s.cum_max()
        lines.append(
            LineSeries(
                name=ticker,
                x=prices[date_col],
                y=((price_s - hwm) / hwm).to_list(),
                color=colors[ticker],
                width=1.5,
                fill=True,
                fill_color=hex_to_rgba(colors[ticker], _FILL_ALPHA),
                hover=HoverSpec(label=ticker, value_format="percent2", date_header=False),
            )
        )

    panel = Panel(
        lines=tuple(lines),
        ref_lines=(RefLine(value=0),),
        yaxis=Axis(title="Drawdown", tick_format="percent0"),
    )
    return FigureSpec(title=title, panels=(panel,))


def drawdowns_periods_spec(data: DataLike, n: int, title: str, asset: str | None) -> FigureSpec:
    """Describe the equity curve with its worst drawdown episodes shaded.

    One asset per chart: overlapping shaded spans from several assets would be
    unreadable.

    Args:
        data: The dataset to plot.
        n: How many of the worst episodes to shade.
        title: Chart title, which gains the asset name.
        asset: Asset column to display, or None for the first.

    Returns:
        FigureSpec: The equity curve, with one band per episode.

    """
    df = data.all
    date_col, tickers = _split_columns(df)
    col = asset if asset in tickers else tickers[0]

    price_list = (1.0 + df[col].cast(pl.Float64)).cum_prod().to_list()
    dates = df[date_col].to_list()

    bands = []
    for i, period in enumerate(compute_drawdown_periods(price_list, n)):
        bands.append(
            Band(
                x0=dates[period["start_idx"]],
                # Extend to the next point so the span covers the final day of
                # the episode rather than stopping at its left edge.
                x1=dates[min(period["end_idx"] + 1, len(dates) - 1)],
                color=hex_to_rgba(PALETTE[i % len(PALETTE)], alpha=_BAND_ALPHA),
                label=f"#{i + 1} {period['max_drawdown']:.1%}",
            )
        )

    panel = Panel(
        lines=(
            LineSeries(
                name=col,
                x=dates,
                y=price_list,
                color=_EQUITY_COLOR,
                hover=HoverSpec(label=col, value_format="float2", suffix="x"),
            ),
        ),
        bands=tuple(bands),
        yaxis=Axis(title="Cumulative Return", tick_format="float2"),
    )
    return FigureSpec(title=f"{title} — {col}", panels=(panel,))
