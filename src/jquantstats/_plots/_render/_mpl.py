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

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter

from .._spec import Axis, BarSeries, ColorScale, FigureSpec, HeatmapGrid, LineSeries, TickFormat

if TYPE_CHECKING:
    from matplotlib.ticker import Formatter

__all__ = ["render_mpl"]

# Pixels per inch used to reinterpret the pixel sizes the specs carry, so
# `figsize=(920, 420)` frames the same chart on either backend.
_DPI = 100

# Width for a spec that names none. Plotly lets a figure size itself to its
# container; matplotlib needs a number up front, so charts that never set a
# width get this one.
_DEFAULT_WIDTH_PX = 1000

# Semantic tick format -> a str.format field spec. Python's format mini-language
# covers percentages too, so `StrMethodFormatter` serves every case and no
# separate `PercentFormatter` is needed.
_FORMATS = {
    "float2": "{x:.2f}",
    "float4": "{x:.4f}",
    "percent1": "{x:.1%}",
    "percent2": "{x:.2%}",
    "currency0": "{x:,.0f}",
}

# Named colour ramp -> the colours it interpolates between.
_COLORSCALES: dict[ColorScale, tuple[str, ...]] = {
    "red_white_green": ("#d62728", "#ffffff", "#2ca02c"),
}

# Semantic dash style -> matplotlib's linestyle vocabulary.
_LINESTYLES = {"solid": "-", "dash": "--"}

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
    return StrMethodFormatter(f"{prefix}{_FORMATS[tick_format]}")


def _draw_line(ax: Axes, line: LineSeries) -> None:
    """Draw one series onto *ax*.

    The series' `~jquantstats._spec.HoverSpec` is ignored: tooltips are
    interactive, and this backend is static.

    Args:
        ax: The axes to draw on.
        line: The series to draw.

    """
    # `to_numpy`, not `to_list`: a polars null becomes NaN in a float array but
    # `None` in a list, and matplotlib draws a gap for the former while the
    # latter forces a slow object-dtype array it cannot interpolate across.
    ax.plot(
        line.x.to_numpy(),
        line.y.to_numpy(),
        color=mpl_color(line.color),
        linewidth=line.width,
        linestyle=_LINESTYLES[line.dash],
        label=line.name,
    )


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
    # Opacity is folded into each colour rather than passed as `alpha=`.
    # matplotlib's alpha argument *replaces* a colour's own alpha channel,
    # whereas Plotly multiplies its trace opacity by it — so passing it
    # separately would render the faded negative bars at the wrong strength.
    ax.bar(
        bar.x.to_numpy(),
        bar.y.to_numpy(),
        color=[_faded(c, bar.opacity) for c in bar.colors],
        linewidth=0,
        label=bar.name,
    )


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

    cmap = LinearSegmentedColormap.from_list(grid.colorscale, _COLORSCALES[grid.colorscale])
    norm = None
    if grid.zmid is not None and masked.count():
        low, high = float(masked.min()), float(masked.max())
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
        spec: The chart to draw. Must describe exactly one panel; multi-panel
            charts arrive with the dashboards.

    Returns:
        Figure: The rendered figure. It is not registered with pyplot, so the
        caller owns it and nothing accumulates between calls.

    Raises:
        ValueError: If *spec* does not contain exactly one panel.

    """
    (panel,) = spec.panels

    width, height = spec.figsize if spec.figsize is not None else (_DEFAULT_WIDTH_PX, spec.height)
    fig = Figure(figsize=(width / _DPI, height / _DPI), dpi=_DPI)
    ax = fig.subplots()

    for line in panel.lines:
        _draw_line(ax, line)
    for bar in panel.bars:
        _draw_bars(ax, bar)
    if panel.heatmap is not None:
        _draw_heatmap(fig, ax, panel.heatmap)

    ax.set_facecolor("white")
    # Stated either way rather than leaning on ``rcParams["axes.grid"]``: that
    # default is global and third-party libraries flip it on import (quantstats
    # does), which would otherwise put a grid behind a matrix chart depending on
    # what else the caller happened to import.
    if spec.chrome == "timeseries":
        ax.grid(visible=True, color=_GRID_COLOR, linewidth=_GRID_WIDTH)
    else:
        ax.grid(visible=False)
    _apply_axis(ax, panel.xaxis, vertical=False)
    _apply_axis(ax, panel.yaxis, vertical=True)

    # Colour carries the value on a matrix chart, so its series names would
    # label nothing; only series charts get a legend.
    series_count = len(panel.lines) + len(panel.bars)
    if series_count and spec.chrome == "timeseries":
        ax.legend(loc="upper left", frameon=False, ncols=series_count)
    fig.suptitle(spec.title)
    return fig
