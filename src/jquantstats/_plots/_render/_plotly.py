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
from plotly.subplots import make_subplots

from .._data._styling import _apply_base_layout, _apply_figsize
from .._spec import (
    Axis,
    Band,
    BarSeries,
    BoxSeries,
    ColorScale,
    FigureSpec,
    HeatmapGrid,
    HistogramSeries,
    HoverSpec,
    LineSeries,
    Panel,
    RefLine,
    TickFormat,
)

__all__ = ["render_plotly"]

# Semantic tick format -> d3 format string, the vocabulary Plotly speaks.
_D3_FORMATS: dict[TickFormat, str] = {
    "float2": ".2f",
    "float4": ".4f",
    "percent0": ".0%",
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
    value = f"%{{{hover.axis}:{_D3_FORMATS[hover.value_format]}}}"
    extra = "<extra></extra>" if hover.hide_extra else ""
    return f"{header}{hover.label}: {hover.prefix}{value}{hover.suffix}{extra}"


def _scatter(line: LineSeries) -> go.Scatter:
    """Convert one line series into a Plotly scatter trace.

    Args:
        line: The series to draw.

    Returns:
        go.Scatter: The trace, styled per the series.

    """
    style: dict[str, object] = {}
    if line.color is not None:
        style["color"] = line.color
    style["width"] = line.width
    if line.dash is not None:
        style["dash"] = _DASHES[line.dash]

    fill: dict[str, object] = {}
    if line.fill_color is not None:
        fill = {"fill": "tozeroy", "fillcolor": line.fill_color}

    grouping: dict[str, object] = {}
    if line.legend_group is not None:
        grouping["legendgroup"] = line.legend_group
    if line.show_legend is not None:
        grouping["showlegend"] = line.show_legend

    trace = go.Scatter(x=line.x, y=line.y, mode="lines", name=line.name, line=style, **fill, **grouping)
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
    # A chart that names no colours takes Plotly's palette, and says nothing
    # about markers or opacity at all.
    styling: dict[str, object] = {}
    if bar.colors is not None:
        styling["marker"] = {"color": list(bar.colors), "line": _BAR_OUTLINE}
    if bar.opacity is not None:
        styling["opacity"] = bar.opacity

    trace = go.Bar(x=bar.x, y=bar.y, name=bar.name, **styling)
    if bar.hover is not None:
        trace.update(hovertemplate=_hovertemplate(bar.hover))
    return trace


def _histogram(hist: HistogramSeries) -> go.Histogram:
    """Convert one histogram series into a Plotly histogram trace.

    Args:
        hist: The series to bin and draw.

    Returns:
        go.Histogram: The trace.

    """
    binning: dict[str, object] = {} if hist.bins is None else {"nbinsx": hist.bins}
    trace = go.Histogram(
        x=hist.values,
        name=hist.name,
        marker_color=hist.color,
        opacity=hist.opacity,
        **binning,
    )
    if hist.hover is not None:
        trace.update(hovertemplate=_hovertemplate(hist.hover))
    return trace


def _box(box: BoxSeries) -> go.Box:
    """Convert one box-and-whisker series into a Plotly box trace.

    Args:
        box: The series to summarise.

    Returns:
        go.Box: The trace, showing outliers individually.

    """
    grouping: dict[str, object] = {"showlegend": box.show_legend}
    if box.legend_group is not None:
        grouping["legendgroup"] = box.legend_group

    trace = go.Box(y=box.values, name=box.name, marker_color=box.color, boxpoints="outliers", **grouping)
    if box.hover is not None:
        trace.update(hovertemplate=_hovertemplate(box.hover))
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


def _add_ref_line(fig: go.Figure, ref: RefLine) -> None:
    """Draw a fixed-value marker line onto *fig*.

    Args:
        fig: The figure to draw on.
        ref: The line to draw.

    """
    # An unset dash is left unstated rather than sent as "solid": that would
    # add a property to the serialised shape which the chart never had.
    style: dict[str, object] = {"line_width": ref.width, "line_color": ref.color}
    if ref.dash is not None:
        style["line_dash"] = _DASHES[ref.dash]
    if ref.label is not None:
        style |= {
            "annotation_text": ref.label,
            "annotation_position": "top right",
            "annotation_font_size": ref.label_size,
        }

    if ref.orientation == "h":
        fig.add_hline(y=ref.value, **style)
    else:
        fig.add_vline(x=ref.value, **style)


def _add_band(fig: go.Figure, band: Band) -> None:
    """Draw a shaded vertical span onto *fig*.

    Args:
        fig: The figure to draw on.
        band: The span to draw.

    """
    kwargs: dict[str, object] = {}
    if band.label is not None:
        kwargs = {
            "annotation_text": band.label,
            "annotation_position": "top left",
            "annotation_font_size": band.label_size,
        }
    fig.add_vrect(x0=band.x0, x1=band.x1, fillcolor=band.color, line_width=0, **kwargs)


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

    """
    fig = _blank_figure(spec)
    for index, panel in enumerate(spec.panels, start=1):
        # A single-panel figure has no subplot grid, so the traces carry no
        # position; a side-by-side one places each panel in its own column.
        at = {} if spec.arrangement == "single" else {"row": 1, "col": index}
        _draw_panel(fig, panel, at)

    _apply_layout(fig, spec)
    _apply_axes(fig, spec)
    return fig


def _blank_figure(spec: FigureSpec) -> go.Figure:
    """Create the figure a spec's panels will be drawn onto.

    Args:
        spec: The chart being rendered.

    Returns:
        go.Figure: An empty figure, with a subplot grid when the spec has one.

    """
    if spec.arrangement == "single":
        return go.Figure()
    return make_subplots(
        rows=1,
        cols=len(spec.panels),
        subplot_titles=[panel.title for panel in spec.panels],
        shared_yaxes=spec.shared_y,
    )


def _draw_panel(fig: go.Figure, panel: Panel, at: dict[str, int]) -> None:
    """Draw one panel's marks onto *fig*.

    Args:
        fig: The figure to draw on.
        panel: The panel to draw.
        at: Subplot position, empty for a single-panel figure.

    """
    for line in panel.lines:
        fig.add_trace(_scatter(line), **at)
    for bar in panel.bars:
        fig.add_trace(_bar(bar), **at)
    for hist in panel.histograms:
        fig.add_trace(_histogram(hist), **at)
    for box in panel.boxes:
        fig.add_trace(_box(box), **at)
    if panel.heatmap is not None:
        fig.add_trace(_heatmap(panel.heatmap), **at)
    for ref in panel.ref_lines:
        _add_ref_line(fig, ref)
    for band in panel.bands:
        _add_band(fig, band)


def _apply_layout(fig: go.Figure, spec: FigureSpec) -> None:
    """Apply the figure-wide layout for *spec*'s chrome.

    Args:
        fig: The figure to lay out.
        spec: The chart being rendered.

    """
    if spec.chrome == "timeseries":
        _apply_base_layout(fig, spec.title, height=spec.height, with_range_selector=spec.date_range_selector)
    elif spec.chrome == "panels":
        # Small multiples: the legend names the categories repeated in every
        # panel, so it sits a little clear of the subplot titles.
        fig.update_layout(
            title=spec.title,
            height=spec.height,
            plot_bgcolor="white",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "xanchor": "right", "x": 1},
        )
        # Horizontal rules help compare box heights across panels; vertical
        # ones would only clutter what is a categorical axis.
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
        fig.update_xaxes(showgrid=False)
    else:
        # A matrix chart: colour encodes the value, so a legend would name
        # nothing and a shared-x hover has nothing to align. Only the title,
        # height and background are set.
        fig.update_layout(title=spec.title, height=spec.height, plot_bgcolor="white")

    _apply_figsize(fig, spec.figsize)
    if spec.bar_mode is not None:
        fig.update_layout(barmode=spec.bar_mode)


def _apply_axes(fig: go.Figure, spec: FigureSpec) -> None:
    """Apply axis configuration across *spec*'s panels.

    Panels of a side-by-side chart share their axis settings, so the first
    panel's configuration is applied to all of them.

    Args:
        fig: The figure to configure.
        spec: The chart being rendered.

    """
    panel = spec.panels[0]
    x_kwargs = _axis_kwargs(panel.xaxis)
    if x_kwargs:
        fig.update_xaxes(**x_kwargs)
    y_kwargs = _axis_kwargs(panel.yaxis, vertical=True)
    if y_kwargs:
        fig.update_yaxes(**y_kwargs)
