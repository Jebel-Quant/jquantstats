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

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter

from .._spec import Axis, FigureSpec, LineSeries, TickFormat

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

# Semantic tick format -> a str.format field spec.
_FORMATS = {
    "float2": "{x:.2f}",
    "float4": "{x:.4f}",
    "currency0": "{x:,.0f}",
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
        color=line.color,
        linewidth=line.width,
        linestyle=_LINESTYLES[line.dash],
        label=line.name,
    )


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

    ax.set_facecolor("white")
    ax.grid(visible=True, color=_GRID_COLOR, linewidth=_GRID_WIDTH)
    _apply_axis(ax, panel.xaxis, vertical=False)
    _apply_axis(ax, panel.yaxis, vertical=True)

    if panel.lines:
        ax.legend(loc="upper left", frameon=False, ncols=len(panel.lines))
    fig.suptitle(spec.title)
    return fig
