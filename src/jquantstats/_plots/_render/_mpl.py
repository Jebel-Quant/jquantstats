"""Render a `FigureSpec` with matplotlib.

Figures are built by constructing `matplotlib.figure.Figure` directly rather
than through ``pyplot``. pyplot keeps every figure it creates alive in a global
registry, so a script looping over hundreds of portfolios accumulates figures
until matplotlib warns and memory grows without bound — the resource cost behind
issue #628. A directly constructed Figure is owned by its caller, garbage
collected normally, and still supports ``fig.savefig(...)``. Importing this
module also never selects a GUI backend or mutates ``rcParams``.

This backend reproduces the same data and the same static design as the Plotly
one. It does not emulate interactivity: hover tooltips and the date
range-selector buttons have no matplotlib equivalent and are silently absent.
Everything determining *what* is plotted — series, colours, axis scales and tick
formats — is reproduced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter, MultipleLocator, StrMethodFormatter

from .._spec import (
    Axis,
    Band,
    BarSeries,
    BoxSeries,
    ColorScale,
    FigureSpec,
    HeatmapGrid,
    HistogramSeries,
    LineSeries,
    Panel,
    RefLine,
    TickFormat,
    Values,
)

if TYPE_CHECKING:
    from matplotlib.ticker import Formatter

__all__ = ["render_mpl"]

# Pixels per inch used to reinterpret the pixel sizes the specs carry, so
# `figsize=(920, 420)` frames the same chart on either backend.
_DPI = 100

# Sizes for a spec that names none. Plotly lets a figure size itself to its
# container; matplotlib needs numbers up front, so charts that never fix a
# dimension get these.
_DEFAULT_WIDTH_PX = 1000
_DEFAULT_HEIGHT_PX = 600

# Semantic tick format -> a str.format field spec. Python's format mini-language
# covers percentages too, so `StrMethodFormatter` serves every case and no
# separate `PercentFormatter` is needed.
_FORMATS = {
    "float2": "{x:.2f}",
    "float4": "{x:.4f}",
    "percent0": "{x:.0%}",
    "percent1": "{x:.1%}",
    "percent2": "{x:.2%}",
    "currency0": "{x:,.0f}",
}

# Named colour ramp -> how matplotlib names it. The custom one is built from
# its stops; the other two are matplotlib built-ins that Plotly borrowed the
# names from, so they need no translation beyond case.
_COLORSCALES: dict[ColorScale, tuple[str, ...]] = {
    "red_white_green": ("#d62728", "#ffffff", "#2ca02c"),
}
_BUILTIN_COLORMAPS: dict[ColorScale, str] = {"rdylgn": "RdYlGn", "rdbu_r": "RdBu_r"}

# Semantic dash style -> matplotlib's linestyle vocabulary.
_LINESTYLES = {"solid": "-", "dash": "--", None: "-"}

# The Plotly charts draw on white with a light grey grid; mirror that here so a
# figure is recognisably the same chart whichever backend drew it.
_GRID_COLOR = "lightgrey"
_GRID_WIDTH = 0.5


def _tick_formatter(tick_format: TickFormat, prefix: str) -> Formatter:
    """Build a tick formatter for one axis.

    Args:
        tick_format: How to render the number.
        prefix: Written before each tick value, e.g. a currency sign.

    Returns:
        Formatter: A formatter applying the requested format and prefix.

    """
    if tick_format == "si2":
        # Plotly's ".2s" abbreviates with an SI suffix — 1500000 reads as
        # "1.5M". `EngFormatter` is matplotlib's equivalent; `sep=""` keeps the
        # suffix tight against the number as d3 writes it.
        return EngFormatter(places=1, sep="")
    return StrMethodFormatter(f"{prefix}{_FORMATS[tick_format]}")


def _as_array(values: Values) -> Any:
    """Convert plotted values to a numpy array.

    Specs carry either a polars Series or a plain list, whichever the chart
    already used (see `~jquantstats._plots._spec.Values`). Going via numpy
    rather than a list also turns a polars null into NaN, which matplotlib
    draws as a gap; a list of ``None`` would instead force a slow object-dtype
    array it cannot interpolate across.

    Args:
        values: A polars Series or a Python list.

    Returns:
        The values as a numpy array — floats where they convert, so ``None``
        becomes NaN, and otherwise left as-is for x-axes holding dates.

    """
    if isinstance(values, pl.Series):
        return values.to_numpy()
    try:
        return np.asarray(values, dtype=float)
    except TypeError:
        # A date axis: leave the objects alone, matplotlib understands them.
        return np.asarray(values)


def _draw_line(ax: Axes, line: LineSeries) -> None:
    """Draw one series onto *ax*.

    The series' `~jquantstats._spec.HoverSpec` is ignored: tooltips are
    interactive, and this backend is static.

    Args:
        ax: The axes to draw on.
        line: The series to draw.

    """
    y = _as_array(line.y)
    # A series with no x is drawn against its own index, which is what
    # Plotly does with an unset `x` too.
    x = np.arange(len(y)) if line.x is None else _as_array(line.x)
    # A line that names no colour is left to matplotlib's own colour cycle,
    # matching how Plotly treats an unset `line.color`.
    styling: dict[str, Any] = {} if line.color is None else {"color": mpl_color(line.color)}
    if line.markers:
        styling["marker"] = "o"
    if line.marker_size is not None:
        # matplotlib sizes a marker by diameter in points, Plotly by area-ish
        # "size"; the square root keeps the two visually comparable.
        styling["markersize"] = line.marker_size**0.5 * 2
    ax.plot(
        x,
        y,
        linewidth=line.width,
        linestyle=_LINESTYLES[line.dash],
        label=line.name,
        **styling,
    )
    if line.fill:
        # `where` keeps the fill out of the gaps a NaN leaves in the line,
        # which otherwise get shaded as though the value were zero. An unnamed
        # fill colour is left to matplotlib, which matches it to the line.
        shading: dict[str, Any] = {} if line.fill_color is None else {"color": mpl_color(line.fill_color)}
        ax.fill_between(x, y, 0, where=~np.isnan(y), linewidth=0, **shading)


def mpl_color(color: str) -> str | tuple[float, float, float, float]:
    """Translate a spec colour into something matplotlib accepts.

    Specs spell translucent colours the CSS way — ``rgba(99, 110, 250, 0.4)``
    — because that is what Plotly emits and the fidelity snapshots pin.
    matplotlib does not parse that form, so it is converted here; plain hex
    passes straight through.

    Args:
        color: A hex string or a CSS ``rgba()`` string.

    Returns:
        The colour as matplotlib understands it.

    Examples:
        >>> mpl_color("#636EFA")
        '#636EFA'
        >>> mpl_color("rgba(99, 110, 250, 0.4)")
        (0.38823529411764707, 0.43137254901960786, 0.9803921568627451, 0.4)

    """
    if not color.startswith("rgba("):
        return color
    r, g, b, a = (part.strip() for part in color[len("rgba(") : -1].split(","))
    return (int(r) / 255, int(g) / 255, int(b) / 255, float(a))


def _faded(color: str, opacity: float) -> tuple[float, float, float, float]:
    """Scale a colour's alpha by *opacity*.

    Matches Plotly, which multiplies a trace's opacity by the alpha already in
    its colour, rather than matplotlib's ``alpha=`` argument, which replaces it.

    Args:
        color: A hex or CSS ``rgba()`` colour.
        opacity: Factor to scale the alpha channel by.

    Returns:
        The colour as RGBA floats, alpha scaled.

    Examples:
        >>> _faded("rgba(0, 0, 0, 0.4)", 0.5)
        (0.0, 0.0, 0.0, 0.2)

    """
    r, g, b, a = to_rgba(mpl_color(color))
    return (r, g, b, a * opacity)


def _draw_bars(ax: Axes, bar: BarSeries) -> None:
    """Draw one bar series onto *ax*.

    Args:
        ax: The axes to draw on.
        bar: The series to draw.

    """
    styling: dict[str, Any] = {}
    if bar.colors is not None:
        # Opacity is folded into each colour rather than passed as `alpha=`.
        # matplotlib's alpha argument *replaces* a colour's own alpha channel,
        # whereas Plotly multiplies its trace opacity by it — so passing it
        # separately would render the faded negative bars at the wrong strength.
        styling["color"] = [_faded(c, bar.opacity if bar.opacity is not None else 1.0) for c in bar.colors]
    elif bar.opacity is not None:
        # No colours named, so there is no alpha to fold into and nothing to
        # get wrong: matplotlib's own argument is the right tool.
        styling["alpha"] = bar.opacity

    ax.bar(_as_array(bar.x), _as_array(bar.y), linewidth=0, label=bar.name, **styling)


def _draw_ref_line(ax: Axes, ref: RefLine) -> None:
    """Draw a fixed-value marker line onto *ax*.

    Args:
        ax: The axes to draw on.
        ref: The line to draw.

    """
    draw = ax.axhline if ref.orientation == "h" else ax.axvline
    draw(ref.value, color=mpl_color(ref.color), linewidth=ref.width, linestyle=_LINESTYLES[ref.dash])
    if ref.label is not None:
        # Anchored to the line in data coordinates and to the top of the axes,
        # matching where Plotly places its annotation.
        anchor = (0.0, ref.value) if ref.orientation == "h" else (ref.value, 1.0)
        coords = ("axes fraction", "data") if ref.orientation == "h" else ("data", "axes fraction")
        ax.annotate(
            ref.label,
            xy=anchor,
            xycoords=coords,
            xytext=(-2, -2),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=ref.label_size,
        )


def _draw_band(ax: Axes, band: Band) -> None:
    """Draw a shaded vertical span onto *ax*.

    Args:
        ax: The axes to draw on.
        band: The span to draw.

    """
    ax.axvspan(band.x0, band.x1, color=mpl_color(band.color), linewidth=0)
    if band.label is not None:
        # Placed at the top of the span in axes coordinates, matching where
        # Plotly puts its annotation.
        ax.annotate(
            band.label,
            xy=(band.x0, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=band.label_size,
        )


def _draw_histogram(ax: Axes, hist: HistogramSeries) -> None:
    """Draw one histogram series onto *ax*.

    Args:
        ax: The axes to draw on.
        hist: The series to bin and draw.

    """
    # `bins=None` is matplotlib's own "use the default count", so an unset bin
    # count passes straight through rather than needing a separate call.
    ax.hist(
        _as_array(hist.values),
        bins=hist.bins,
        color=mpl_color(hist.color),
        alpha=hist.opacity,
        label=hist.name,
    )


def _draw_boxes(ax: Axes, boxes: tuple[BoxSeries, ...]) -> None:
    """Draw a panel's box-and-whisker series onto *ax*.

    Drawn together rather than one at a time: matplotlib positions boxes by
    index within a single call, and wants the category labels alongside.

    Args:
        ax: The axes to draw on.
        boxes: The series to summarise, left to right.

    """
    if not boxes:
        return
    artists = ax.boxplot(
        [_as_array(box.values) for box in boxes],
        tick_labels=[box.name for box in boxes],
        showfliers=True,
        patch_artist=True,
    )
    for patch, box in zip(artists["boxes"], boxes, strict=True):
        patch.set_facecolor(mpl_color(box.color))


def _colormap(scale: ColorScale) -> Any:
    """Resolve a named colour ramp to a matplotlib colormap.

    Args:
        scale: The ramp's semantic name.

    Returns:
        The colormap, built from explicit stops or looked up by name.

    """
    builtin = _BUILTIN_COLORMAPS.get(scale)
    if builtin is not None:
        return colormaps[builtin]
    return LinearSegmentedColormap.from_list(scale, _COLORSCALES[scale])


def _draw_heatmap(fig: Figure, ax: Axes, grid: HeatmapGrid) -> None:
    """Draw a value matrix onto *ax*, with per-cell labels and a colour bar.

    Args:
        fig: The figure owning *ax*, needed to attach the colour bar.
        ax: The axes to draw on.
        grid: The matrix to draw.

    """
    # None marks a month the data does not cover. NaN carries that through
    # numpy, and a masked array keeps those cells unpainted rather than
    # colouring them as if they were zero.
    values = np.array([[float("nan") if v is None else v for v in row] for row in grid.z], dtype=float)
    masked = np.ma.masked_invalid(values)

    cmap = _colormap(grid.colorscale)
    norm = None
    if grid.zmid is not None and masked.count():
        # Pinned bounds win over the data's own range: a correlation runs -1 to
        # 1 whatever this particular matrix happens to span.
        low = float(masked.min()) if grid.zmin is None else grid.zmin
        high = float(masked.max()) if grid.zmax is None else grid.zmax
        # TwoSlopeNorm needs the centre strictly inside the range; widen a
        # one-sided or degenerate span so an all-positive year still renders.
        low = min(low, grid.zmid - 1e-9)
        high = max(high, grid.zmid + 1e-9)
        norm = TwoSlopeNorm(vmin=low, vcenter=grid.zmid, vmax=high)

    image = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(grid.x_labels)), labels=list(grid.x_labels))
    ax.set_yticks(range(len(grid.y_labels)), labels=list(grid.y_labels))
    for row, labels in enumerate(grid.text):
        for col, label in enumerate(labels):
            if label:
                ax.text(col, row, label, ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label=grid.colorbar_title)


def _apply_axis(ax: Axes, axis: Axis, *, vertical: bool) -> None:
    """Apply one axis configuration to *ax*.

    Only properties the spec actually set are touched, mirroring the Plotly
    renderer so the two backends agree on what "unset" means.

    Args:
        ax: The axes to configure.
        axis: The requested configuration.
        vertical: Configure the y-axis rather than the x-axis.

    """
    target = ax.yaxis if vertical else ax.xaxis
    if axis.title is not None:
        target.set_label_text(axis.title)
    if axis.tick_format is not None:
        target.set_major_formatter(_tick_formatter(axis.tick_format, axis.tick_prefix))
    if axis.log:
        set_scale = ax.set_yscale if vertical else ax.set_xscale
        set_scale("log")
    if axis.dtick is not None:
        target.set_major_locator(MultipleLocator(axis.dtick))
    if axis.opposite_side:
        # Each axis has its own vocabulary for "the far edge", which is why
        # the spec says `opposite_side` rather than naming a compass point.
        if vertical:
            ax.yaxis.set_ticks_position("right")
            ax.yaxis.set_label_position("right")
        else:
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")


def render_mpl(spec: FigureSpec) -> Figure:
    """Render *spec* as a static matplotlib figure.

    Args:
        spec: The chart to draw.

    Returns:
        Figure: The rendered figure. It is not registered with pyplot, so the
        caller owns it and nothing accumulates between calls.

    """
    width, height = spec.figsize if spec.figsize is not None else (_DEFAULT_WIDTH_PX, spec.height)
    fig = Figure(figsize=(width / _DPI, (height or _DEFAULT_HEIGHT_PX) / _DPI), dpi=_DPI)
    axes = _make_axes(fig, spec)

    for ax, panel in zip(axes, spec.panels, strict=True):
        _draw_panel(fig, ax, panel, spec)
    fig.suptitle(spec.title)
    return fig


def _make_axes(fig: Figure, spec: FigureSpec) -> Any:
    """Create one Axes per panel, laid out as the spec asks.

    Args:
        fig: The figure to add axes to.
        spec: The chart being rendered.

    Returns:
        The axes, flattened to one per panel in spec order.

    """
    if spec.arrangement == "stacked":
        # Stacked panels share the time axis and split the height in the
        # proportions the panels ask for, so the headline view gets the room.
        return fig.subplots(
            nrows=len(spec.panels),
            sharex=spec.shared_x,
            height_ratios=[panel.height_ratio for panel in spec.panels],
            squeeze=False,
        )[:, 0]
    return fig.subplots(ncols=len(spec.panels), sharey=spec.shared_y, squeeze=False)[0]


def _draw_panel(fig: Figure, ax: Axes, panel: Panel, spec: FigureSpec) -> None:
    """Draw one panel's marks and configure its axes.

    Args:
        fig: The figure owning *ax*, needed to attach a colour bar.
        ax: The axes to draw on.
        panel: The panel to draw.
        spec: The chart being rendered, for its figure-wide chrome.

    """
    for line in panel.lines:
        _draw_line(ax, line)
    for bar in panel.bars:
        _draw_bars(ax, bar)
    for hist in panel.histograms:
        _draw_histogram(ax, hist)
    _draw_boxes(ax, panel.boxes)
    if panel.heatmap is not None:
        _draw_heatmap(fig, ax, panel.heatmap)
    for ref in panel.ref_lines:
        _draw_ref_line(ax, ref)
    for band in panel.bands:
        _draw_band(ax, band)

    ax.set_facecolor("white")
    # Stated either way rather than leaning on ``rcParams["axes.grid"]``: that
    # default is global and third-party libraries flip it on import (quantstats
    # does), which would otherwise put a grid behind a matrix chart depending on
    # what else the caller happened to import.
    if spec.chrome == "bare":
        ax.grid(visible=False)
    elif spec.chrome == "panels":
        # Horizontal rules help compare box heights across panels; vertical
        # ones would only clutter what is a categorical axis.
        ax.grid(visible=True, axis="y", color=_GRID_COLOR, linewidth=_GRID_WIDTH)
    else:
        ax.grid(visible=True, color=_GRID_COLOR, linewidth=_GRID_WIDTH)

    if panel.title is not None:
        ax.set_title(panel.title)
    _apply_axis(ax, panel.xaxis, vertical=False)
    _apply_axis(ax, panel.yaxis, vertical=True)

    # Colour carries the value on a matrix chart, so its series names would
    # label nothing; only series charts get a legend.
    series_count = len(panel.lines) + len(panel.bars) + len(panel.histograms)
    if series_count and spec.chrome != "bare":
        ax.legend(loc="upper left", frameon=False, ncols=series_count)
