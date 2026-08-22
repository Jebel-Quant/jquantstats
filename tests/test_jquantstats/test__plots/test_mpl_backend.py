"""The matplotlib rendering backend, and its parity with Plotly.

Three concerns, in order:

* **dispatch** — that a backend is selected from the right place;
* **parity** — that both backends plot the same numbers, which is the property
  the spec/renderer split exists to guarantee;
* **translation** — that each semantic spec property reaches the right
  matplotlib artist, including the ones no cumulative chart happens to use.
"""

from __future__ import annotations

import matplotlib.figure as mfigure
import plotly.graph_objects as go
import polars as pl
import pytest
from matplotlib.ticker import StrMethodFormatter

import jquantstats
from jquantstats._plots import _backend
from jquantstats._plots._render import render, render_plotly
from jquantstats._plots._render._mpl import _DEFAULT_WIDTH_PX, _DPI, render_mpl
from jquantstats._plots._spec import Axis, FigureSpec, LineSeries, Panel
from jquantstats.exceptions import MissingBackendError

# Every cumulative chart, with arguments exercising the interesting branches.
CUMULATIVE_CHARTS = [
    ("returns", {}),
    ("returns", {"log_scale": True}),
    ("compare", {}),
    ("compare", {"figsize": (920, 420)}),
    ("log_returns", {}),
    ("log_returns", {"figsize": (800, 400)}),
    ("earnings", {}),
    ("earnings", {"compounded": False}),
    ("earnings", {"start_balance": 250_000}),
]
_IDS = [f"{name}-{sorted(kwargs)}" for name, kwargs in CUMULATIVE_CHARTS]


def _line(**overrides) -> LineSeries:
    """Build a minimal line series, overriding selected fields."""
    defaults = {
        "name": "AAPL",
        "x": pl.Series("date", [1, 2, 3]),
        "y": pl.Series("AAPL", [1.0, 1.1, 1.2]),
        "color": "#636EFA",
    }
    return LineSeries(**{**defaults, **overrides})


def _spec(panel: Panel, **overrides) -> FigureSpec:
    """Build a single-panel figure spec, overriding selected fields."""
    return FigureSpec(**{"title": "T", "panels": (panel,), **overrides})


# ── dispatch ─────────────────────────────────────────────────────────────────


def test_default_backend_still_returns_plotly(data) -> None:
    """Existing callers are untouched: no argument still means Plotly."""
    assert isinstance(data.plots.returns(), go.Figure)


def test_explicit_backend_selects_matplotlib(data) -> None:
    """A per-call argument selects the renderer."""
    assert isinstance(data.plots.returns(backend="matplotlib"), mfigure.Figure)


def test_global_default_reaches_the_plot_methods(data) -> None:
    """`set_plot_backend` changes what a bare call returns."""
    jquantstats.set_plot_backend("matplotlib")
    assert isinstance(data.plots.returns(), mfigure.Figure)


def test_context_manager_reaches_the_plot_methods(data) -> None:
    """A scoped override applies inside the block and not outside it."""
    with jquantstats.plot_backend("matplotlib"):
        assert isinstance(data.plots.returns(), mfigure.Figure)
    assert isinstance(data.plots.returns(), go.Figure)


def test_explicit_argument_outranks_the_global_default(data) -> None:
    """The most specific selection wins."""
    jquantstats.set_plot_backend("matplotlib")
    assert isinstance(data.plots.returns(backend="plotly"), go.Figure)


def test_missing_matplotlib_names_the_extra(data, monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting an uninstalled backend explains how to install it."""
    monkeypatch.setattr(_backend, "_find_spec", lambda name: None)
    with pytest.raises(MissingBackendError, match=r"jquantstats\[mpl\]"):
        data.plots.returns(backend="matplotlib")


def test_render_dispatches_to_both_backends() -> None:
    """`render` is the single seam both backends are reached through."""
    spec = _spec(Panel(lines=(_line(),)))
    assert isinstance(render(spec, "plotly"), go.Figure)
    assert isinstance(render(spec, "matplotlib"), mfigure.Figure)


# ── parity ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("method", "kwargs"), CUMULATIVE_CHARTS, ids=_IDS)
def test_backends_plot_the_same_numbers(data, method: str, kwargs: dict) -> None:
    """Both backends draw identical series for every cumulative chart.

    This is the anti-drift ratchet. It only holds because both figures are
    rendered from one spec, so there is a single copy of the arithmetic; if the
    two renderers are ever fed from separate code paths, this fails.
    """
    plotly_fig = getattr(data.plots, method)(**kwargs, backend="plotly")
    mpl_fig = getattr(data.plots, method)(**kwargs, backend="matplotlib")

    mpl_lines = mpl_fig.axes[0].get_lines()
    assert len(mpl_lines) == len(plotly_fig.data)

    for trace, line in zip(plotly_fig.data, mpl_lines, strict=True):
        assert trace.name == line.get_label()
        assert list(trace.y) == pytest.approx(list(line.get_ydata()), nan_ok=True)


@pytest.mark.parametrize(("method", "kwargs"), CUMULATIVE_CHARTS, ids=_IDS)
def test_backends_agree_on_colours(data, method: str, kwargs: dict) -> None:
    """Colour is chosen in the spec, so both backends use the same one."""
    plotly_fig = getattr(data.plots, method)(**kwargs, backend="plotly")
    mpl_fig = getattr(data.plots, method)(**kwargs, backend="matplotlib")

    plotly_colors = [trace.line.color.lower() for trace in plotly_fig.data]
    mpl_colors = [line.get_color().lower() for line in mpl_fig.axes[0].get_lines()]
    assert plotly_colors == mpl_colors


def test_validation_is_backend_independent(data_no_benchmark) -> None:
    """Validation sits in the builder, so every backend raises alike."""
    for backend in ("plotly", "matplotlib"):
        with pytest.raises(AttributeError, match=r"^compare\(\) requires benchmark data to be set$"):
            data_no_benchmark.plots.compare(backend=backend)


# ── translation ──────────────────────────────────────────────────────────────


def test_figures_are_not_registered_with_pyplot(data) -> None:
    """Rendering must not add to pyplot's global registry.

    A figure that pyplot owns is never garbage collected, which is the leak
    behind issue #628. The autouse leak fixture guards the whole suite; this
    states the requirement where it is easy to find.
    """
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    for _ in range(25):
        data.plots.returns(backend="matplotlib")
    assert set(plt.get_fignums()) == before


def test_figsize_is_pixels_on_both_backends(data) -> None:
    """`figsize=(920, 420)` means the same thing either way.

    Plotly measures in pixels and matplotlib in inches, so the renderer
    converts. Without that, one public signature would frame two different
    charts.
    """
    plotly_fig = data.plots.compare(figsize=(920, 420), backend="plotly")
    mpl_fig = data.plots.compare(figsize=(920, 420), backend="matplotlib")

    assert (plotly_fig.layout.width, plotly_fig.layout.height) == (920, 420)
    assert mpl_fig.get_size_inches().tolist() == [9.2, 4.2]
    assert mpl_fig.dpi == _DPI


def test_default_width_applies_when_no_figsize_given(data) -> None:
    """A chart that names no width still needs a concrete one."""
    fig = data.plots.returns(backend="matplotlib")
    assert fig.get_size_inches().tolist() == [_DEFAULT_WIDTH_PX / _DPI, 600 / _DPI]


def test_title_becomes_the_figure_suptitle(data) -> None:
    """The chart title survives the translation."""
    assert data.plots.returns(title="Growth", backend="matplotlib").get_suptitle() == "Growth"


def test_dashed_benchmarks_are_dashed(data) -> None:
    """Benchmarks read as the reference on this backend too."""
    fig = data.plots.compare(backend="matplotlib")
    styles = [line.get_linestyle() for line in fig.axes[0].get_lines()]
    assert "--" in styles, f"expected a dashed benchmark, got {styles}"


def test_log_scale_reaches_the_axis(data) -> None:
    """`log_scale` sets the axis scale rather than transforming the data."""
    assert data.plots.returns(log_scale=True, backend="matplotlib").axes[0].get_yscale() == "log"
    assert data.plots.returns(log_scale=False, backend="matplotlib").axes[0].get_yscale() == "linear"


def test_currency_ticks_carry_the_prefix(data) -> None:
    """The earnings axis renders as money."""
    fig = data.plots.earnings(backend="matplotlib")
    assert fig.axes[0].yaxis.get_major_formatter()(12345.0) == "$12,345"


def test_axis_without_tick_format_keeps_the_default_formatter(data) -> None:
    """Log returns set no tick format, so matplotlib's default stands."""
    fig = data.plots.log_returns(backend="matplotlib")
    assert not isinstance(fig.axes[0].yaxis.get_major_formatter(), StrMethodFormatter)


def test_both_axes_are_honoured() -> None:
    """A configured x-axis reaches the figure, not just the y-axis.

    No cumulative chart configures its x-axis, so without this the x-axis path
    would be an untested no-op on this backend.
    """
    panel = Panel(
        lines=(_line(),),
        xaxis=Axis(title="When", tick_format="float2", log=True),
        yaxis=Axis(title="How much"),
    )
    ax = render_mpl(_spec(panel)).axes[0]
    assert ax.get_xlabel() == "When"
    assert ax.get_ylabel() == "How much"
    assert ax.get_xscale() == "log"


def test_lineless_spec_draws_no_legend() -> None:
    """An empty chart must not ask matplotlib for a legend with no entries."""
    fig = render_mpl(_spec(Panel()))
    assert fig.axes[0].get_legend() is None


def test_render_rejects_multi_panel_specs() -> None:
    """Multi-panel rendering arrives with the dashboards, not before."""
    panel = Panel(lines=(_line(),))
    with pytest.raises(ValueError, match="too many values"):
        render_mpl(FigureSpec(title="T", panels=(panel, panel)))


def test_hover_is_dropped_rather_than_emulated(data) -> None:
    """Tooltips are interactive; the static backend simply omits them.

    Recorded as a deliberate degradation rather than left as an unstated gap.
    """
    plotly_fig = data.plots.returns(backend="plotly")
    assert plotly_fig.data[0].hovertemplate

    mpl_fig = data.plots.returns(backend="matplotlib")
    assert not mpl_fig.axes[0].texts


def test_renderers_are_independent_of_one_another() -> None:
    """Rendering one backend must not disturb the other's output."""
    spec = _spec(Panel(lines=(_line(),)))
    before = render_plotly(spec).to_json()
    render_mpl(spec)
    assert render_plotly(spec).to_json() == before
