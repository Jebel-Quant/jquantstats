"""Shared styling and figure-layout helpers for the data plots.

The colour helpers here are re-exported from `jquantstats._plots._style`, which
is where they now live: choosing a colour is backend-neutral, whereas applying a
Plotly layout is not. The aliases keep the plot families that have not yet moved
to the spec/renderer split working against a single definition.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from .._style import bar_colors as _bar_colors
from .._style import hex_to_rgba as _hex_to_rgba
from .._style import ticker_colors as _ticker_colors
from .._style import yearly_bar_colors as _yearly_bar_colors

__all__ = [
    "_apply_base_layout",
    "_apply_figsize",
    "_bar_colors",
    "_date_range_selector",
    "_hex_to_rgba",
    "_ticker_colors",
    "_yearly_bar_colors",
]


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
    height: int | None = 600,
    with_range_selector: bool = True,
) -> go.Figure:
    """Apply the standard jquantstats Plotly layout to a figure.

    Sets white background, light-grey grid, horizontal legend, and an
    optional date range-selector on the primary x-axis.

    Args:
        fig: The Plotly figure to style in-place.
        title: Chart title.
        height: Figure height in pixels. Defaults to 600. None leaves the
            height unset, which Plotly treats as "size to the container".
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
