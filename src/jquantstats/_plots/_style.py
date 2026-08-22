"""Colour helpers shared by every chart, independent of any drawing library.

These decide *which* colours a chart uses. How those colours are applied to a
figure is a renderer's job, so nothing here imports a drawing library.

The palette is Plotly's qualitative sequence, kept because changing it would
alter every existing chart. `plotly.express` is imported only to read that list
of hex strings; no figure is built and no renderer is selected.
"""

from __future__ import annotations

import plotly.express as px

__all__ = ["hex_to_rgba", "ticker_colors"]


def hex_to_rgba(hex_color: str, alpha: float = 0.5) -> str:
    """Convert a hex colour to an RGBA CSS string.

    Args:
        hex_color: A hex colour, with or without a leading ``#``.
        alpha: Opacity in the range [0, 1]. Defaults to 0.5.

    Returns:
        str: An ``rgba(r, g, b, a)`` string. Both backends accept this
        spelling, so it needs no per-renderer translation.

    Examples:
        >>> hex_to_rgba("#636efa", 0.4)
        'rgba(99, 110, 250, 0.4)'

    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def ticker_colors(tickers: list[str]) -> dict[str, str]:
    """Assign a stable colour to each ticker.

    Colours are taken from the palette in order and wrap around once it is
    exhausted, so the same ticker list always yields the same mapping.

    Args:
        tickers: Ordered ticker / column names.

    Returns:
        dict[str, str]: One hex colour per ticker.

    Examples:
        >>> ticker_colors(["AAPL", "META"])["AAPL"]
        '#636EFA'

    """
    palette = px.colors.qualitative.Plotly
    return {ticker: palette[i % len(palette)] for i, ticker in enumerate(tickers)}
