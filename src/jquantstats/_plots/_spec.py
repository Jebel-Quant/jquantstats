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
from typing import Literal

import polars as pl

__all__ = [
    "Axis",
    "BarSeries",
    "Chrome",
    "ColorScale",
    "Dash",
    "FigureSpec",
    "HeatmapGrid",
    "HoverSpec",
    "LineSeries",
    "Panel",
    "TickFormat",
]

#: How to render a number, named by intent rather than by any backend's syntax.
#:
#: ``float2``/``float4`` are fixed-point to that many decimals; ``percent1``/
#: ``percent2`` scale to a percentage with that many decimals; ``currency0`` is
#: a thousands-separated integer. Each renderer owns the mapping — Plotly wants
#: ``".2f"``, matplotlib wants a ``Formatter`` — so neither vocabulary leaks in
#: here.
TickFormat = Literal["float2", "float4", "percent1", "percent2", "currency0"]

#: Line styles, kept to the set the charts actually use.
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
Chrome = Literal["timeseries", "bare"]


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

    """

    label: str
    value_format: TickFormat
    prefix: str = ""
    suffix: str = ""
    date_header: bool = True


@dataclass(frozen=True, slots=True)
class LineSeries:
    """One line drawn across a panel.

    Attributes:
        name: Legend entry for the series.
        x: Horizontal positions, typically the date column.
        y: Vertical positions.
        color: Line colour as ``#RRGGBB``; both backends accept that spelling.
        width: Stroke width.
        dash: Stroke pattern.
        hover: Tooltip description, or None to leave the backend's default.

    """

    name: str
    x: pl.Series
    y: pl.Series
    color: str
    # Left as an int so a whole-number width serialises as `2` rather than
    # `2.0`. Plotly preserves the distinction in its JSON, and the fidelity
    # snapshots compare that JSON exactly.
    width: float = 2
    dash: Dash = "solid"
    hover: HoverSpec | None = None


@dataclass(frozen=True, slots=True)
class BarSeries:
    """One set of bars drawn across a panel.

    Attributes:
        name: Legend entry for the series.
        x: Bar positions — dates, or the category each bar sits at.
        y: Bar heights.
        colors: One colour per bar. Per-bar rather than per-series because
            these charts colour a bar by the sign of its value.
        opacity: Fill opacity.
        hover: Tooltip description, or None to leave the backend's default.

    """

    name: str
    x: pl.Series
    y: pl.Series
    colors: tuple[str, ...]
    opacity: float = 0.85
    hover: HoverSpec | None = None


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
        heatmap: A value matrix to draw, or None.
        xaxis: Horizontal axis configuration.
        yaxis: Vertical axis configuration.
        title: Heading for this panel, used when a chart has several.

    """

    lines: tuple[LineSeries, ...] = ()
    bars: tuple[BarSeries, ...] = ()
    heatmap: HeatmapGrid | None = None
    xaxis: Axis = field(default_factory=Axis)
    yaxis: Axis = field(default_factory=Axis)
    title: str | None = None


@dataclass(frozen=True, slots=True)
class FigureSpec:
    """A complete chart, ready for any renderer.

    Attributes:
        title: Chart title.
        panels: The panels to draw, top to bottom.
        height: Figure height in pixels.
        figsize: Optional ``(width, height)`` in pixels, overriding *height*.
            Pixels for both backends; the matplotlib renderer converts to
            inches so one public signature means the same thing either way.
        date_range_selector: Offer the Plotly range-selector buttons. Ignored
            by matplotlib, which has no interactive widgets.
        chrome: How much surrounding furniture the chart carries.
        bar_mode: How bars from different series share an x position, or None
            for the backend's default.

    """

    title: str
    panels: tuple[Panel, ...]
    height: int = 600
    figsize: tuple[int, int] | None = None
    date_range_selector: bool = True
    chrome: Chrome = "timeseries"
    bar_mode: Literal["group", "overlay", "relative"] | None = None
