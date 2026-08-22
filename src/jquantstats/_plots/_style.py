"""Colour helpers shared by every chart, independent of any drawing library.

These decide *which* colours a chart uses. How those colours are applied to a
figure is a renderer's job, so nothing here imports a drawing library.

The palette is Plotly's qualitative sequence, copied out rather than imported.
Reading it from `plotly.express` would make this module — and everything that
picks a colour, which is every spec builder — depend on a drawing library, and
would silently restyle every chart if Plotly ever revised the sequence. The
fidelity snapshots pin these exact values, so a copy is the honest form.
"""

from __future__ import annotations

__all__ = ["PALETTE", "bar_colors", "hex_to_rgba", "ticker_colors", "yearly_bar_colors"]

#: Plotly's ``qualitative.Plotly`` sequence, as of plotly 6.
PALETTE = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
)

#: Green and red for a single-asset chart, where a bar's colour can carry the
#: sign outright rather than having to stay identifiable as one asset among several.
_POSITIVE = "#2ca02c"
_NEGATIVE = "#d62728"


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
    return {ticker: PALETTE[i % len(PALETTE)] for i, ticker in enumerate(tickers)}


def bar_colors(values: list[float | None], positive_color: str, single_asset: bool = False) -> list[str]:
    """Colour each bar by the sign of its value.

    With one asset the bars are plain green and red. With several, each keeps
    its own palette colour and negatives are faded instead, so a bar stays
    identifiable as belonging to its asset.

    Args:
        values: The plotted values; None counts as negative.
        positive_color: The asset's base colour.
        single_asset: Use the plain green/red palette.

    Returns:
        list[str]: One colour per value.

    Examples:
        >>> bar_colors([0.1, -0.1], "#636EFA", single_asset=True)
        ['#2ca02c', '#d62728']

    """
    if single_asset:
        return [_POSITIVE if v is not None and v > 0 else _NEGATIVE for v in values]
    negative_color = hex_to_rgba(positive_color, alpha=0.4)
    return [positive_color if v is not None and v > 0 else negative_color for v in values]


def yearly_bar_colors(values: list[float | None], positive_color: str) -> list[str]:
    """Colour each annual bar by the sign of its value.

    Deliberately distinct from `bar_colors`: a flat zero year counts as
    positive here (``>= 0``), and negatives fade to alpha 0.5 rather than 0.4.
    The two cannot share an implementation without changing what is rendered.

    Args:
        values: The per-year return values; None counts as negative.
        positive_color: The asset's base colour.

    Returns:
        list[str]: One colour per value.

    Examples:
        >>> yearly_bar_colors([0.0], "#636EFA")
        ['#636EFA']

    """
    negative_color = hex_to_rgba(positive_color, 0.5)
    return [positive_color if v is not None and v >= 0 else negative_color for v in values]
