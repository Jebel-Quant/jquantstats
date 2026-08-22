"""A backend-agnostic description of a chart.

Every plot method in jquantstats is two separable jobs: prepare the numbers with
polars, then hand them to a drawing library. The types here are the seam between
those halves. A *spec builder* under `jquantstats._plots._specs` turns a dataset
into a `FigureSpec`; a *renderer* under `jquantstats._plots._render` turns that
`FigureSpec` into a Plotly or matplotlib figure.

The point of the seam is arithmetic. Without it a second backend means a second
copy of every rolling beta, drawdown scan and Monte Carlo path, and the two
copies drift. With it there is one builder per chart and one renderer per
backend.

Nothing here may import a drawing library, and no field may hold a
backend-specific value — no Plotly format strings, no matplotlib linestyle
tuples. Where a visual property has to be named, it is named semantically
(`"float2"`, `"dash"`) and each renderer maps it to its own vocabulary.

**Series stay as `polars.Series`.** Plotly serialises a Series to a compact
binary buffer and a Python list to a plain JSON array, so converting here would
silently change every rendered figure. Renderers that need other containers
convert at the point of use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
import polars as pl

__all__ = [
    "Arrangement",
    "Axis",
    "Band",
    "BarSeries",
    "BoxSeries",
    "Chrome",
    "ColorScale",
    "Dash",
    "FigureSpec",
    "HeatmapGrid",
    "HistogramSeries",
    "HoverSpec",
    "LineSeries",
    "Panel",
    "RefLine",
    "TickFormat",
    "Values",
]

#: Plotted values: a polars Series, a numpy array, or a plain Python list.
#:
#: All three are accepted deliberately. Plotly serialises a Series or an array
#: to a compact binary buffer and a list to a plain JSON array, and each chart's
#: existing wire format is pinned by the fidelity snapshots. Builders therefore
#: pass through whichever container the chart already used rather than
#: normalising, so moving a chart onto this seam changes nothing. Renderers must
#: accept any of them.
Values: TypeAlias = "pl.Series | np.ndarray[Any, Any] | list[Any]"

#: How to render a number, named by intent rather than by any backend's syntax.
#:
#: ``float2``/``float4`` are fixed-point to that many decimals; ``percent1``/
#: ``percent2`` scale to a percentage with that many decimals; ``currency0`` is
#: a thousands-separated integer. Each renderer owns the mapping — Plotly wants
#: ``".2f"``, matplotlib wants a ``Formatter`` — so neither vocabulary leaks in
#: here.
TickFormat = Literal["float2", "float4", "percent0", "percent1", "percent2", "currency0"]

#: Line styles, kept to the set the charts actually use.
#:
#: A field typed ``Dash | None`` treats None as *say nothing* and leave the
#: backend's default, which is distinct from asking for ``"solid"``
#: explicitly. Some charts state it and some do not, and Plotly records the
#: difference in its serialised output.
Dash = Literal["solid", "dash"]

#: Named colour ramps for matrix charts, resolved per backend.
ColorScale = Literal["red_white_green"]

#: How much furniture a chart carries.
#:
#: ``timeseries`` is the standard treatment shared by most charts: a legend,
#: unified hover, a light grid and optionally the date range-selector.
#: ``bare`` is for matrix charts, where colour encodes the value rather than
#: the series — a legend would name nothing and a shared-x hover has no
#: meaning, so only the title, height and background are set.
#: ``panels`` is the small-multiples treatment: one panel per asset sharing a
#: scale, with the legend naming the categories repeated in each panel rather
#: than the panels themselves.
Chrome = Literal["timeseries", "bare", "panels"]

#: How a chart's panels are laid out.
#:
#: ``side_by_side`` places them in a row sharing the vertical scale, so the
#: same measurement can be compared across assets. Stacked panels arrive with
#: the dashboards.
Arrangement = Literal["single", "side_by_side"]


@dataclass(frozen=True, slots=True)
class HoverSpec:
    """The tooltip shown when a pointer rests on a series.

    Interactive-only, and therefore Plotly-only: matplotlib has no equivalent
    and its renderer ignores this entirely. It is described structurally rather
    than as a template string so that spec builders never write Plotly syntax.

    Attributes:
        label: Text naming the series, shown before the value.
        value_format: How to render the value.
        prefix: Written immediately before the value, e.g. a currency sign.
        suffix: Written immediately after the value, e.g. ``"x"`` to mark a
            growth multiple.
        date_header: Show the x-value as a bold ``Mon YYYY`` heading above.
        axis: Which coordinate carries the value. Histograms bin along x, so
            their tooltip reads the x-value; everything else reads y.
        hide_extra: Suppress the trace-name box Plotly appends beside the
            tooltip.

    """

    label: str
    value_format: TickFormat
    prefix: str = ""
    suffix: str = ""
    date_header: bool = True
    axis: Literal["x", "y"] = "y"
    hide_extra: bool = False


@dataclass(frozen=True, slots=True)
class LineSeries:
    """One line drawn across a panel.

    Attributes:
        name: Legend entry for the series.
        x: Horizontal positions, typically the date column, or None to let the
            backend number the points.
        y: Vertical positions.
        color: Line colour as ``#RRGGBB``, or None to take the backend's next
            palette colour.
        width: Stroke width.
        dash: Stroke pattern, or None to leave the backend's default.
        fill_color: Shade the area between the line and zero in this colour,
            or None to leave the line unfilled. Used by the underwater curve.
        hover: Tooltip description, or None to leave the backend's default.
        show_legend: Whether to give this series its own legend entry, or None
            to say nothing and leave the backend's default. The fan charts
            state it on every path — True for the first of a bundle, False for
            the rest — so that hundreds of lines produce one entry.
        legend_group: Name tying several series together, so toggling the
            legend shows or hides them as one.

    """

    name: str
    y: Values
    x: Values | None = None
    color: str | None = None
    # Left as an int so a whole-number width serialises as `2` rather than
    # `2.0`. Plotly preserves the distinction in its JSON, and the fidelity
    # snapshots compare that JSON exactly.
    width: float = 2
    dash: Dash | None = None
    fill_color: str | None = None
    hover: HoverSpec | None = None
    show_legend: bool | None = None
    legend_group: str | None = None


@dataclass(frozen=True, slots=True)
class HistogramSeries:
    """One set of binned values drawn as a histogram.

    Attributes:
        name: Legend entry for the series.
        values: The observations to bin.
        color: Bar colour.
        bins: Requested bin count, or None for the backend's own choice.
        opacity: Fill opacity. These charts overlay several series, so the
            default is translucent.
        hover: Tooltip description, or None to leave the backend's default.

    """

    name: str
    values: Values
    color: str
    bins: int | None = None
    opacity: float = 0.6
    hover: HoverSpec | None = None


@dataclass(frozen=True, slots=True)
class BoxSeries:
    """One box-and-whisker summary of a set of values.

    Attributes:
        name: Category label, shown on the axis and in the legend.
        values: The observations to summarise.
        color: Box colour.
        show_legend: Give this series its own legend entry. With one panel per
            asset the same categories repeat, so only the first panel does.
        legend_group: Name tying the same category across panels together.
        hover: Tooltip description, or None to leave the backend's default.

    """

    name: str
    values: Values
    color: str
    show_legend: bool = True
    legend_group: str | None = None
    hover: HoverSpec | None = None


@dataclass(frozen=True, slots=True)
class BarSeries:
    """One set of bars drawn across a panel.

    Attributes:
        name: Legend entry for the series.
        x: Bar positions — dates, or the category each bar sits at.
        y: Bar heights.
        colors: One colour per bar, or None to take the backend's palette.
            Per-bar rather than per-series because several of these charts
            colour a bar by the sign of its value.
        opacity: Fill opacity, or None for the backend's default.
        hover: Tooltip description, or None to leave the backend's default.

    """

    name: str
    x: Values
    y: Values
    colors: tuple[str, ...] | None = None
    opacity: float | None = None
    hover: HoverSpec | None = None


@dataclass(frozen=True, slots=True)
class RefLine:
    """A straight line marking a fixed value, such as break-even at zero.

    Attributes:
        value: Where to draw it, in data coordinates.
        orientation: ``"h"`` for a horizontal line at *value*, ``"v"`` for a
            vertical one.
        color: Line colour.
        width: Stroke width.
        dash: Stroke pattern, or None to leave the backend's default.
        label: Text naming the line, or None for no label.
        label_size: Point size for that text.

    """

    value: float
    orientation: Literal["h", "v"] = "h"
    color: str = "gray"
    width: float = 1
    dash: Dash | None = None
    label: str | None = None
    label_size: int = 10


@dataclass(frozen=True, slots=True)
class Band:
    """A shaded vertical span marking a stretch of the x-axis.

    Used to pick out the worst drawdown episodes on an equity curve.

    Attributes:
        x0: Where the span starts.
        x1: Where it ends.
        color: Fill colour, usually translucent so the line stays readable.
        label: Text drawn at the top of the span, or None for no label.
        label_size: Point size for that text.

    """

    x0: Any
    x1: Any
    color: str
    label: str | None = None
    label_size: int = 10


@dataclass(frozen=True, slots=True)
class HeatmapGrid:
    """A matrix of values drawn as a coloured grid.

    Attributes:
        x_labels: Column headings, left to right.
        y_labels: Row headings, top to bottom.
        z: The values, as ``z[row][column]``. None marks a missing cell.
        text: Per-cell labels drawn over the grid, parallel to *z*.
        colorscale: The colour ramp to map values through.
        zmid: Value anchored to the middle of a diverging ramp, or None to
            span the data range.
        colorbar_title: Heading for the colour scale legend.
        hover_label: Word naming the quantity in the tooltip, or None for no
            tooltip. Interactive-only, so matplotlib ignores it.

    """

    x_labels: tuple[str, ...]
    y_labels: tuple[str, ...]
    z: tuple[tuple[float | None, ...], ...]
    text: tuple[tuple[str, ...], ...]
    colorscale: ColorScale
    # An int, so a whole-number anchor serialises as `0` rather than `0.0`;
    # Plotly preserves the distinction and the fidelity snapshots compare it.
    zmid: float | None = 0
    colorbar_title: str = ""
    hover_label: str | None = None


@dataclass(frozen=True, slots=True)
class Axis:
    """Configuration for one axis of a panel.

    Every field defaults to "leave it alone", so a renderer sets only what a
    builder asked for. That matters for fidelity: emitting a property the
    original chart never set would change the rendered output.

    Attributes:
        title: Axis label, or None for no label.
        tick_format: How to render tick values, or None for the default.
        tick_prefix: Written before each tick value, e.g. a currency sign.
        log: Use a logarithmic scale.
        opposite_side: Draw the axis on the far edge — the top for a
            horizontal axis, the right for a vertical one. The monthly
            calendar puts its months along the top, where they read as column
            headings.

    """

    title: str | None = None
    tick_format: TickFormat | None = None
    tick_prefix: str = ""
    log: bool = False
    # Deliberately a flag rather than a compass point. Which edge counts as
    # "opposite" depends on the axis, and naming it absolutely would admit
    # nonsense a renderer would then have to police — an x-axis on the "left".
    opposite_side: bool = False


@dataclass(frozen=True, slots=True)
class Panel:
    """One set of axes and the marks drawn on it.

    A chart is a tuple of panels. Most are a single panel; the dashboards stack
    several sharing an x-axis.

    Attributes:
        lines: Line series to draw.
        bars: Bar series to draw.
        histograms: Histogram series to draw.
        boxes: Box-and-whisker series to draw.
        heatmap: A value matrix to draw, or None.
        ref_lines: Fixed-value marker lines.
        bands: Shaded vertical spans, drawn behind the series.
        xaxis: Horizontal axis configuration.
        yaxis: Vertical axis configuration.
        title: Heading for this panel, used when a chart has several.

    """

    lines: tuple[LineSeries, ...] = ()
    bars: tuple[BarSeries, ...] = ()
    histograms: tuple[HistogramSeries, ...] = ()
    boxes: tuple[BoxSeries, ...] = ()
    heatmap: HeatmapGrid | None = None
    ref_lines: tuple[RefLine, ...] = ()
    bands: tuple[Band, ...] = ()
    xaxis: Axis = field(default_factory=Axis)
    yaxis: Axis = field(default_factory=Axis)
    title: str | None = None


@dataclass(frozen=True, slots=True)
class FigureSpec:
    """A complete chart, ready for any renderer.

    Attributes:
        title: Chart title.
        panels: The panels to draw, top to bottom.
        height: Figure height in pixels, or None to let the backend size it.
        figsize: Optional ``(width, height)`` in pixels, overriding *height*.
            Pixels for both backends; the matplotlib renderer converts to
            inches so one public signature means the same thing either way.
        date_range_selector: Offer the Plotly range-selector buttons. Ignored
            by matplotlib, which has no interactive widgets.
        chrome: How much surrounding furniture the chart carries.
        arrangement: How the panels are laid out.
        shared_y: Give side-by-side panels one vertical scale, so their
            distributions are directly comparable.
        bar_mode: How bars from different series share an x position, or None
            for the backend's default.

    """

    title: str
    panels: tuple[Panel, ...]
    height: int | None = 600
    figsize: tuple[int, int] | None = None
    date_range_selector: bool = True
    chrome: Chrome = "timeseries"
    arrangement: Arrangement = "single"
    shared_y: bool = False
    bar_mode: Literal["group", "overlay", "relative"] | None = None
