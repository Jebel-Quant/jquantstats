"""Shared styling and figure-layout helpers for the data plots."""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go


def _hex_to_rgba(hex_color: str, alpha: float = 0.5) -> str:
    """Convert a hex colour string to an RGBA CSS string.

    Args:
        hex_color: A hex colour string (with or without a leading ``#``).
        alpha: Opacity in the range [0, 1]. Defaults to 0.5.

    Returns:
        An RGBA CSS string suitable for use in Plotly colour arguments.

    """
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _ticker_colors(tickers: list[str]) -> dict[str, str]:
    """Map ticker names to Plotly qualitative palette colours.

    Args:
        tickers: Ordered list of ticker / column names.

    Returns:
        dict mapping each ticker to a hex colour string.

    """
    palette = px.colors.qualitative.Plotly
    return {ticker: palette[i % len(palette)] for i, ticker in enumerate(tickers)}


def _date_range_selector() -> dict[str, Any]:
    """Return a standard Plotly date range-selector configuration.

    Returns:
        A dict suitable for ``xaxis.rangeselector``.

    """
    return {
        "buttons": [
            {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
            {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
            {"count": 3, "label": "3y", "step": "year", "stepmode": "backward"},
            {"step": "year", "stepmode": "todate", "label": "YTD"},
            {"step": "all", "label": "All"},
        ]
    }


def _apply_base_layout(
    fig: go.Figure,
    title: str,
    height: int = 600,
    with_range_selector: bool = True,
) -> go.Figure:
    """Apply the standard jquantstats Plotly layout to a figure.

    Sets white background, light-grey grid, horizontal legend, and an
    optional date range-selector on the primary x-axis.

    Args:
        fig: The Plotly figure to style in-place.
        title: Chart title.
        height: Figure height in pixels. Defaults to 600.
        with_range_selector: Attach a date range-selector to ``xaxis``.
            Defaults to True.

    Returns:
        The same figure, mutated in-place and returned for chaining.

    """
    layout_kw: dict[str, Any] = {
        "title": title,
        "height": height,
        "hovermode": "x unified",
        "plot_bgcolor": "white",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    }
    if with_range_selector:
        layout_kw["xaxis"] = {
            "rangeselector": _date_range_selector(),
            "rangeslider": {"visible": False},
            "type": "date",
        }
    fig.update_layout(**layout_kw)
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
    return fig


def _apply_figsize(fig: go.Figure, figsize: tuple[int, int] | None) -> go.Figure:
    """Apply optional ``(width, height)`` figure size to Plotly layout."""
    if figsize is not None:
        fig.update_layout(width=figsize[0], height=figsize[1])
    return fig


def _bar_colors(values: list[float | None], positive_color: str, single_asset: bool = False) -> list[str]:
    """Return the shared positive/negative bar colors for a series of values."""
    if single_asset:
        return ["#2ca02c" if v is not None and v > 0 else "#d62728" for v in values]
    negative_color = _hex_to_rgba(positive_color, alpha=0.4)
    return [positive_color if v is not None and v > 0 else negative_color for v in values]


def _compute_drawdown_periods(prices: list[float], n: int) -> list[dict[str, Any]]:
    """Identify the top *n* drawdown periods from a cumulative price series.

    Args:
        prices: Cumulative price (NAV) values as a plain Python list.
        n: Maximum number of drawdown periods to return.

    Returns:
        List of dicts with keys ``start_idx``, ``end_idx``, ``valley_idx``,
        ``max_drawdown`` (fraction ≤ 0), sorted by severity (worst first).

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
