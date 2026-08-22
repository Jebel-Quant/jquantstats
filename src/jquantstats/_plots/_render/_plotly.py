"""Render a `FigureSpec` with Plotly.

This module owns every Plotly-specific decision: how a semantic
`~jquantstats._plots._spec.TickFormat` becomes a d3 format string, how a
`~jquantstats._plots._spec.HoverSpec` becomes a hovertemplate, and how the
shared layout is applied.

It deliberately reuses `_apply_base_layout` and `_apply_figsize` rather than
restating what they do. Those helpers already produce the layout every chart in
the package has, so routing through them is what lets a chart move onto the
spec/renderer split without altering a single byte of its rendered output — the
property the full-fidelity snapshot tests exist to pin.
"""

from __future__ import annotations

import plotly.graph_objects as go

from .._data._styling import _apply_base_layout, _apply_figsize
from .._spec import Axis, BarSeries, ColorScale, FigureSpec, HeatmapGrid, HoverSpec, LineSeries, TickFormat

__all__ = ["render_plotly"]

# Semantic tick format -> d3 format string, the vocabulary Plotly speaks.
_D3_FORMATS: dict[TickFormat, str] = {
    "float2": ".2f",
    "float4": ".4f",
    "percent1": ".1%",
    "percent2": ".2%",
    "currency0": ",.0f",
}

# Semantic dash style -> Plotly's `line.dash` vocabulary.
_DASHES = {"solid": "solid", "dash": "dash"}

# Named colour ramp -> Plotly's colorscale stops.
_COLORSCALES: dict[ColorScale, list[list[float | str]]] = {
    "red_white_green": [[0, "#d62728"], [0.5, "#ffffff"], [1, "#2ca02c"]],
}

# Bars are drawn without an outline throughout.
_BAR_OUTLINE = {"width": 0}


def _hovertemplate(hover: HoverSpec) -> str:
    """Build a Plotly hovertemplate from a structural hover description.

    Args:
        hover: What the tooltip should say.

    Returns:
        str: A hovertemplate string.

    """
    header = "<b>%{x|%b %Y}</b><br>" if hover.date_header else ""
    value = f"%{{y:{_D3_FORMATS[hover.value_format]}}}"
    return f"{header}{hover.label}: {hover.prefix}{value}{hover.suffix}"


def _scatter(line: LineSeries) -> go.Scatter:
    """Convert one line series into a Plotly scatter trace.

    Args:
        line: The series to draw.

    Returns:
        go.Scatter: The trace, styled per the series.

    """
    style: dict[str, object] = {"color": line.color, "width": line.width}
    if line.dash != "solid":
        style["dash"] = _DASHES[line.dash]

    trace = go.Scatter(x=line.x, y=line.y, mode="lines", name=line.name, line=style)
    if line.hover is not None:
        trace.update(hovertemplate=_hovertemplate(line.hover))
    return trace


def _bar(bar: BarSeries) -> go.Bar:
    """Convert one bar series into a Plotly bar trace.

    Args:
        bar: The series to draw.

    Returns:
        go.Bar: The trace, with one colour per bar.

    """
    trace = go.Bar(
        x=bar.x,
        y=bar.y,
        name=bar.name,
        marker={"color": list(bar.colors), "line": _BAR_OUTLINE},
        opacity=bar.opacity,
    )
    if bar.hover is not None:
        trace.update(hovertemplate=_hovertemplate(bar.hover))
    return trace


def _heatmap(grid: HeatmapGrid) -> go.Heatmap:
    """Convert a value matrix into a Plotly heatmap trace.

    Args:
        grid: The matrix to draw.

    Returns:
        go.Heatmap: The trace, labelled cell by cell.

    """
    trace = go.Heatmap(
        x=list(grid.x_labels),
        y=list(grid.y_labels),
        z=[list(row) for row in grid.z],
        text=[list(row) for row in grid.text],
        texttemplate="%{text}",
        colorscale=_COLORSCALES[grid.colorscale],
        zmid=grid.zmid,
        showscale=True,
        colorbar={"title": grid.colorbar_title},
    )
    if grid.hover_label is not None:
        trace.update(hovertemplate=f"<b>%{{y}} %{{x}}</b><br>{grid.hover_label}: %{{text}}<extra></extra>")
    return trace


def _axis_kwargs(axis: Axis, *, vertical: bool = False) -> dict[str, object]:
    """Collect the axis properties a spec actually asked for.

    Only requested properties are returned. Emitting a default for an untouched
    property would change the serialised figure, which the fidelity snapshots
    would (correctly) flag.

    Args:
        axis: The axis configuration.
        vertical: Whether this is the y-axis, which decides what
            `~jquantstats._plots._spec.Axis.opposite_side` resolves to.

    Returns:
        dict[str, object]: Keyword arguments for ``update_xaxes`` /
        ``update_yaxes``.

    """
    kwargs: dict[str, object] = {}
    if axis.title is not None:
        kwargs["title_text"] = axis.title
    if axis.tick_prefix:
        kwargs["tickprefix"] = axis.tick_prefix
    if axis.tick_format is not None:
        kwargs["tickformat"] = _D3_FORMATS[axis.tick_format]
    if axis.log:
        kwargs["type"] = "log"
    if axis.opposite_side:
        kwargs["side"] = "right" if vertical else "top"
    return kwargs


def render_plotly(spec: FigureSpec) -> go.Figure:
    """Render *spec* as an interactive Plotly figure.

    Args:
        spec: The chart to draw. Must describe exactly one panel; multi-panel
            charts arrive with the dashboards.

    Returns:
        go.Figure: The rendered figure.

    Raises:
        ValueError: If *spec* does not contain exactly one panel.

    """
    (panel,) = spec.panels

    fig = go.Figure()
    for line in panel.lines:
        fig.add_trace(_scatter(line))
    for bar in panel.bars:
        fig.add_trace(_bar(bar))
    if panel.heatmap is not None:
        fig.add_trace(_heatmap(panel.heatmap))

    if spec.chrome == "timeseries":
        _apply_base_layout(fig, spec.title, height=spec.height, with_range_selector=spec.date_range_selector)
    else:
        # A matrix chart: colour encodes the value, so a legend would name
        # nothing and a shared-x hover has nothing to align. Only the title,
        # height and background are set.
        fig.update_layout(title=spec.title, height=spec.height, plot_bgcolor="white")
    _apply_figsize(fig, spec.figsize)
    if spec.bar_mode is not None:
        fig.update_layout(barmode=spec.bar_mode)

    x_kwargs = _axis_kwargs(panel.xaxis)
    if x_kwargs:
        fig.update_xaxes(**x_kwargs)
    y_kwargs = _axis_kwargs(panel.yaxis, vertical=True)
    if y_kwargs:
        fig.update_yaxes(**y_kwargs)
    return fig
