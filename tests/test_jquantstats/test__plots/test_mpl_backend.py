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
import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest
from matplotlib.colors import TwoSlopeNorm, to_hex, to_rgba
from matplotlib.ticker import StrMethodFormatter

import jquantstats
from jquantstats._plots import _backend
from jquantstats._plots._render import render, render_plotly
from jquantstats._plots._render._mpl import _DEFAULT_WIDTH_PX, _DPI, mpl_color, render_mpl
from jquantstats._plots._spec import Axis, Band, BarSeries, FigureSpec, HeatmapGrid, LineSeries, Panel, RefLine
from jquantstats.exceptions import MissingBackendError, NoBenchmarkError


def _normalise_colour(colour: str) -> tuple[float, ...]:
    """Reduce a colour to comparable RGBA floats.

    The two backends spell the same colour differently — a spec carries
    Plotly's CSS ``rgba(99, 110, 250, 0.4)``, while matplotlib reports a hex
    string — so comparison goes through a common representation. It routes
    through the renderer's own `mpl_color`, so this checks the real translation
    rather than a second copy of it. Values are rounded because ``rgba()``
    carries 8-bit channels while matplotlib keeps floats.
    """
    return tuple(round(channel, 2) for channel in to_rgba(mpl_color(colour)))


def _scaled_colour(colour: str, opacity: float) -> tuple[float, ...]:
    """Reduce a colour to comparable RGBA floats with its alpha scaled."""
    r, g, b, a = to_rgba(mpl_color(colour))
    return tuple(round(channel, 2) for channel in (r, g, b, a * opacity))


def _floats(values) -> list[float]:
    """Normalise plotted values for comparison, missing points becoming NaN.

    Plotly reports a missing point as ``None`` when the spec carried a plain
    list and as NaN when it carried a Series, while matplotlib always reports
    NaN. That is a difference in representation, not in what is drawn.
    """
    return [float("nan") if value is None else float(value) for value in values]


# Every line chart migrated so far, with arguments exercising the interesting
# branches. Parity is asserted per line series, so bar and matrix charts are
# checked separately below.
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

# The periodic bar charts.
BAR_CHARTS = [
    ("daily_returns", {}),
    ("yearly_returns", {}),
    ("yearly_returns", {"compounded": False}),
    ("monthly_returns", {}),
    ("monthly_returns", {"compounded": False}),
]
_BAR_IDS = [f"{name}-{sorted(kwargs)}" for name, kwargs in BAR_CHARTS]


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
        assert _floats(trace.y) == pytest.approx(_floats(line.get_ydata()), nan_ok=True)


@pytest.mark.parametrize(("method", "kwargs"), CUMULATIVE_CHARTS, ids=_IDS)
def test_backends_agree_on_colours(data, method: str, kwargs: dict) -> None:
    """Colour is chosen in the spec, so both backends use the same one."""
    plotly_fig = getattr(data.plots, method)(**kwargs, backend="plotly")
    mpl_fig = getattr(data.plots, method)(**kwargs, backend="matplotlib")

    plotly_colors = [trace.line.color.lower() for trace in plotly_fig.data]
    mpl_colors = [line.get_color().lower() for line in mpl_fig.axes[0].get_lines()]
    assert plotly_colors == mpl_colors


@pytest.mark.parametrize(("method", "kwargs"), BAR_CHARTS, ids=_BAR_IDS)
def test_bar_backends_plot_the_same_numbers(data, method: str, kwargs: dict) -> None:
    """Both backends draw identical bars for every periodic chart."""
    plotly_fig = getattr(data.plots, method)(**kwargs, backend="plotly")
    mpl_fig = getattr(data.plots, method)(**kwargs, backend="matplotlib")

    containers = mpl_fig.axes[0].containers
    assert len(containers) == len(plotly_fig.data)

    for trace, container in zip(plotly_fig.data, containers, strict=True):
        assert trace.name == container.get_label()
        heights = [patch.get_height() for patch in container]
        assert _floats(trace.y) == pytest.approx(_floats(heights), nan_ok=True)


@pytest.mark.parametrize(("method", "kwargs"), BAR_CHARTS, ids=_BAR_IDS)
def test_bar_backends_agree_on_per_bar_colours(data, method: str, kwargs: dict) -> None:
    """Sign colouring is decided in the spec, so both backends match.

    These charts colour each bar by the sign of its value rather than by
    series, which is the reason `BarSeries` carries a colour per bar.

    The two backends apply opacity differently: Plotly keeps it on the trace
    and multiplies, while matplotlib has no equivalent, so the renderer folds
    it into each colour's alpha. Comparison therefore scales Plotly's colours
    by its trace opacity before matching.
    """
    plotly_fig = getattr(data.plots, method)(**kwargs, backend="plotly")
    mpl_fig = getattr(data.plots, method)(**kwargs, backend="matplotlib")

    for trace, container in zip(plotly_fig.data, mpl_fig.axes[0].containers, strict=True):
        expected = [_scaled_colour(c, trace.opacity) for c in trace.marker.color]
        actual = [_normalise_colour(to_hex(p.get_facecolor(), keep_alpha=True)) for p in container]
        assert expected == actual


def test_heatmap_backends_plot_the_same_matrix(data) -> None:
    """The calendar grid holds the same values on both backends."""
    plotly_fig = data.plots.monthly_heatmap(backend="plotly")
    mpl_fig = data.plots.monthly_heatmap(backend="matplotlib")

    expected = [list(row) for row in plotly_fig.data[0].z]
    actual = mpl_fig.axes[0].images[0].get_array()

    assert actual.shape == (len(expected), 12)
    for row_index, row in enumerate(expected):
        for col_index, value in enumerate(row):
            cell = actual[row_index, col_index]
            if value is None:
                assert cell is np.ma.masked, "an uncovered month must stay unpainted"
            else:
                assert float(cell) == pytest.approx(value)


def test_heatmap_titles_name_the_asset(data) -> None:
    """One asset per calendar, so the title says which."""
    asset = data.assets[0]
    plotly_fig = data.plots.monthly_heatmap(asset=asset, backend="plotly")
    mpl_fig = data.plots.monthly_heatmap(asset=asset, backend="matplotlib")

    assert plotly_fig.layout.title.text.endswith(f"— {asset}")
    assert mpl_fig.get_suptitle().endswith(f"— {asset}")


def test_heatmap_puts_months_along_the_top(data) -> None:
    """Months read as column headings on both backends."""
    assert data.plots.monthly_heatmap(backend="plotly").layout.xaxis.side == "top"

    mpl_fig = data.plots.monthly_heatmap(backend="matplotlib")
    labels = [t.get_text() for t in mpl_fig.axes[0].get_xticklabels()]
    assert labels[:3] == ["Jan", "Feb", "Mar"]
    assert mpl_fig.axes[0].xaxis.get_ticks_position() == "top"


def test_matrix_charts_carry_no_legend(data) -> None:
    """Colour encodes the value, so a legend would name nothing."""
    assert data.plots.monthly_heatmap(backend="matplotlib").axes[0].get_legend() is None


def _grid_is_on(ax) -> bool:
    """Whether *ax* draws grid lines.

    Read from the ticks rather than ``xaxis.get_gridlines()``: those Line2D
    artists exist and report themselves visible whether or not the grid is
    actually enabled.
    """
    ticks = ax.xaxis.get_major_ticks()
    return bool(ticks) and ticks[0].gridline.get_visible()


def test_matrix_charts_carry_no_grid(data) -> None:
    """A grid behind an opaque matrix would be hidden anyway."""
    assert not _grid_is_on(data.plots.monthly_heatmap(backend="matplotlib").axes[0])


def test_series_charts_keep_their_grid(data) -> None:
    """The counterpart: a line or bar chart still gets its grid."""
    assert _grid_is_on(data.plots.daily_returns(backend="matplotlib").axes[0])


def test_heatmap_without_a_midpoint_spans_the_data_range() -> None:
    """A grid may decline to anchor its colour ramp.

    No calendar does — they all centre on zero so red and green read
    symmetrically — but the spec can express it, so it must work.
    """
    grid = HeatmapGrid(
        x_labels=("Jan", "Feb"),
        y_labels=("2024",),
        z=((1.0, 3.0),),
        text=(("", ""),),
        colorscale="red_white_green",
        zmid=None,
    )
    fig = render_mpl(FigureSpec(title="T", panels=(Panel(heatmap=grid),), chrome="bare"))
    assert not isinstance(fig.axes[0].images[0].norm, TwoSlopeNorm)


def test_heatmap_with_a_midpoint_centres_the_ramp(data) -> None:
    """Anchoring at zero is what makes red and green read symmetrically."""
    norm = data.plots.monthly_heatmap(backend="matplotlib").axes[0].images[0].norm
    assert isinstance(norm, TwoSlopeNorm)
    assert norm.vcenter == 0


def test_line_without_x_is_drawn_against_its_index() -> None:
    """A series may omit its x-axis.

    The portfolio rolling charts pass whatever their metric frame carries, so
    a frame without a date column yields `x=None`. Plotly numbers the points
    itself; matplotlib is given the same numbering explicitly.
    """
    ax = render_mpl(_spec(Panel(lines=(_line(x=None),)))).axes[0]
    assert list(ax.get_lines()[0].get_xdata()) == [0, 1, 2]


def test_palette_bars_can_still_be_translucent() -> None:
    """Opacity works without named colours.

    The bar charts always name their colours and the annual breakdown names
    none *and* sets no opacity, so this combination has no chart behind it —
    but the spec can express it, and folding opacity into a colour is
    impossible when the colour comes from the backend's palette. matplotlib's
    own ``alpha`` is the right tool there.
    """
    bar = BarSeries(name="A", x=pl.Series("x", [1, 2]), y=pl.Series("y", [1.0, 2.0]), opacity=0.5)
    ax = render_mpl(_spec(Panel(bars=(bar,)))).axes[0]
    assert ax.containers[0][0].get_alpha() == 0.5


def test_band_without_a_label_draws_no_text() -> None:
    """A span may shade without naming itself."""
    panel = Panel(lines=(_line(),), bands=(Band(x0=1, x1=2, color="rgba(0, 0, 0, 0.2)"),))
    ax = render_mpl(_spec(panel)).axes[0]
    assert len(ax.patches) == 1
    assert not ax.texts


def test_vertical_and_dashed_reference_lines_render() -> None:
    """Both marker orientations and stroke patterns reach the axes.

    The drawdown charts only use a solid horizontal line, so the other
    variants are covered here rather than left untested.
    """
    refs = (RefLine(value=0), RefLine(value=2, orientation="v", dash="dash"))
    ax = render_mpl(_spec(Panel(lines=(_line(),), ref_lines=refs))).axes[0]

    horizontals = [ln for ln in ax.get_lines() if list(ln.get_ydata()) == [0, 0]]
    verticals = [ln for ln in ax.get_lines() if list(ln.get_xdata()) == [2, 2]]
    assert horizontals
    assert verticals
    assert verticals[0].get_linestyle() == "--"


def test_opposite_side_resolves_per_axis() -> None:
    """The far edge is the top for an x-axis and the right for a y-axis.

    Only the calendar's x-axis uses this in practice, so the vertical case is
    covered here rather than left as an untested path.
    """
    panel = Panel(lines=(_line(),), xaxis=Axis(opposite_side=True), yaxis=Axis(opposite_side=True))
    ax = render_mpl(_spec(panel)).axes[0]
    assert ax.xaxis.get_ticks_position() == "top"
    assert ax.yaxis.get_ticks_position() == "right"


def test_heatmap_with_no_values_still_renders() -> None:
    """An entirely empty calendar must not divide by an empty range."""
    grid = HeatmapGrid(
        x_labels=("Jan",),
        y_labels=("2024",),
        z=((None,),),
        text=(("",),),
        colorscale="red_white_green",
    )
    fig = render_mpl(FigureSpec(title="T", panels=(Panel(heatmap=grid),), chrome="bare"))
    assert fig.axes[0].images[0].get_array().count() == 0


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("rolling_sharpe", {}),
        ("rolling_sharpe", {"rolling_period": 63}),
        ("rolling_sortino", {}),
        ("rolling_volatility", {}),
        ("rolling_beta", {}),
        ("rolling_beta", {"rolling_period2": None}),
        ("rolling_beta", {"figsize": (900, 400)}),
    ],
    ids=lambda v: str(sorted(v)) if isinstance(v, dict) else v,
)
def test_rolling_backends_plot_the_same_numbers(data, method: str, kwargs: dict) -> None:
    """Both backends draw identical rolling metrics.

    Reference lines are Line2D artists on the matplotlib side, so the series
    are matched by name rather than by position.
    """
    plotly_fig = getattr(data.plots, method)(**kwargs, backend="plotly")
    mpl_fig = getattr(data.plots, method)(**kwargs, backend="matplotlib")

    names = [trace.name for trace in plotly_fig.data]
    series = {ln.get_label(): ln for ln in mpl_fig.axes[0].get_lines() if ln.get_label() in set(names)}
    assert sorted(series) == sorted(names)

    for trace in plotly_fig.data:
        assert _floats(trace.y) == pytest.approx(_floats(series[trace.name].get_ydata()), nan_ok=True)


@pytest.mark.parametrize("window", [21, 63])
def test_portfolio_rolling_backends_agree(pf, window: int) -> None:
    """The portfolio facade's rolling charts match across backends too."""
    for method in ("rolling_sharpe_plot", "rolling_volatility_plot"):
        plotly_fig = getattr(pf.plots, method)(window=window, backend="plotly")
        mpl_fig = getattr(pf.plots, method)(window=window, backend="matplotlib")

        names = {trace.name for trace in plotly_fig.data}
        series = {ln.get_label(): ln for ln in mpl_fig.axes[0].get_lines() if ln.get_label() in names}
        assert set(series) == names

        for trace in plotly_fig.data:
            assert _floats(trace.y) == pytest.approx(_floats(series[trace.name].get_ydata()), nan_ok=True)


def test_portfolio_rolling_charts_take_the_backend_palette(pf) -> None:
    """These charts name no colours, unlike the Data ones.

    The spec leaves `LineSeries.color` unset, so each backend assigns from its
    own palette rather than the shared one.
    """
    spec_lines = pf.plots.rolling_sharpe_plot(backend="plotly").data
    assert all(trace.line.color is None for trace in spec_lines)

    mpl_fig = pf.plots.rolling_sharpe_plot(backend="matplotlib")
    assert all(line.get_color() for line in mpl_fig.axes[0].get_lines())


def test_annual_sharpe_backends_plot_the_same_bars(pf) -> None:
    """The per-year breakdown matches, and takes the backend palette."""
    plotly_fig = pf.plots.annual_sharpe_plot(backend="plotly")
    mpl_fig = pf.plots.annual_sharpe_plot(backend="matplotlib")

    containers = mpl_fig.axes[0].containers
    assert len(containers) == len(plotly_fig.data)

    for trace, container in zip(plotly_fig.data, containers, strict=True):
        assert trace.name == container.get_label()
        assert trace.marker.color is None, "no colours named; the backend picks"
        heights = [patch.get_height() for patch in container]
        assert _floats(trace.y) == pytest.approx(_floats(heights), nan_ok=True)


def test_rolling_window_validation_is_backend_independent(pf) -> None:
    """A bad window is rejected in the builder, before any renderer runs."""
    for backend in ("plotly", "matplotlib"):
        for method in ("rolling_sharpe_plot", "rolling_volatility_plot"):
            with pytest.raises(ValueError, match="window must be a positive integer"):
                getattr(pf.plots, method)(window=0, backend=backend)


def test_rolling_beta_requires_a_benchmark_on_both_backends(data_no_benchmark) -> None:
    """Validation lives above the dispatch, so both backends raise alike."""
    for backend in ("plotly", "matplotlib"):
        with pytest.raises(NoBenchmarkError):
            data_no_benchmark.plots.rolling_beta(backend=backend)


def test_data_dashboard_stacks_three_panels(data) -> None:
    """The returns dashboard is three views over one shared time axis."""
    mpl_fig = data.plots.snapshot(backend="matplotlib")
    assert [ax.get_title() for ax in mpl_fig.axes] == ["Cumulative Returns", "Drawdowns", "Monthly Returns"]

    plotly_fig = data.plots.snapshot(backend="plotly")
    assert [a.text for a in plotly_fig.layout.annotations] == ["Cumulative Returns", "Drawdowns", "Monthly Returns"]


def test_portfolio_dashboard_stacks_two_panels(pf) -> None:
    """The portfolio dashboard pairs NAV with the drawdown it produced."""
    mpl_fig = pf.plots.snapshot(backend="matplotlib")
    assert [ax.get_title() for ax in mpl_fig.axes] == ["Accumulated Profit", "Drawdown"]


def test_stacked_panels_share_one_time_axis(data) -> None:
    """Sharing x is what lets a drawdown be read against its months."""
    axes = data.plots.snapshot(backend="matplotlib").axes
    first = axes[0].get_xlim()
    assert all(ax.get_xlim() == first for ax in axes[1:])


def test_dashboard_headline_panel_is_the_tallest(data) -> None:
    """Cumulative returns get half the height; the rest split the remainder."""
    heights = [ax.get_position().height for ax in data.plots.snapshot(backend="matplotlib").axes]
    assert heights[0] > heights[1]
    assert heights[1] == pytest.approx(heights[2], rel=0.05)


@pytest.mark.parametrize("log_scale", [False, True])
def test_dashboard_log_scale_applies_only_to_the_top_panel(data, log_scale: bool) -> None:
    """A log NAV axis must not turn the drawdown panel logarithmic too."""
    axes = data.plots.snapshot(log_scale=log_scale, backend="matplotlib").axes
    assert axes[0].get_yscale() == ("log" if log_scale else "linear")
    assert axes[1].get_yscale() == "linear"


def test_dashboard_backends_plot_the_same_panels(data) -> None:
    """Every panel's series match across backends.

    Traces are matched to panels by their subplot axis on the Plotly side,
    since a stacked figure flattens them into one trace list.
    """
    plotly_fig = data.plots.snapshot(backend="plotly")
    mpl_axes = data.plots.snapshot(backend="matplotlib").axes

    for index, ax in enumerate(mpl_axes, start=1):
        axis = "y" if index == 1 else f"y{index}"
        traces = [t for t in plotly_fig.data if (t.yaxis or "y") == axis]
        drawn = ax.get_lines() or ax.containers
        # Reference lines are Line2D artists too, so compare by name.
        named = {t.name for t in traces}
        assert {a.get_label() for a in drawn} >= named


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("lagged_performance_plot", {}),
        ("lagged_performance_plot", {"lags": [0, 2]}),
        ("lagged_performance_plot", {"log_scale": True}),
        ("smoothed_holdings_performance_plot", {}),
        ("smoothed_holdings_performance_plot", {"windows": [0, 3]}),
    ],
    ids=lambda v: str(sorted(v)) if isinstance(v, dict) else v,
)
def test_nav_comparison_backends_agree(pf, method: str, kwargs: dict) -> None:
    """The lag and smoothing sweeps match across backends."""
    plotly_fig = getattr(pf.plots, method)(**kwargs, backend="plotly")
    mpl_fig = getattr(pf.plots, method)(**kwargs, backend="matplotlib")

    lines = mpl_fig.axes[0].get_lines()
    assert [line.get_label() for line in lines] == [t.name for t in plotly_fig.data]
    for trace, line in zip(plotly_fig.data, lines, strict=True):
        assert _floats(trace.y) == pytest.approx(_floats(line.get_ydata()), nan_ok=True)


@pytest.mark.parametrize(
    ("method", "bad"),
    [
        ("lagged_performance_plot", {"lags": (0, 1)}),
        ("lagged_performance_plot", {"lags": [0, "1"]}),
        ("smoothed_holdings_performance_plot", {"windows": [-1]}),
    ],
    ids=["tuple", "non-int", "negative"],
)
def test_nav_comparison_validation_is_backend_independent(pf, method: str, bad: dict) -> None:
    """Validation sits in the builder, so both backends reject alike."""
    for backend in ("plotly", "matplotlib"):
        with pytest.raises(TypeError):
            getattr(pf.plots, method)(**bad, backend=backend)


def test_histogram_backends_bin_the_same_values(data) -> None:
    """Both backends receive the same observations to bin.

    Bin *edges* are each backend's own business — Plotly's `nbinsx` is a hint
    and matplotlib's `bins` is exact — so the counts are what is compared.
    """
    plotly_fig = data.plots.histogram(bins=20, backend="plotly")
    mpl_fig = data.plots.histogram(bins=20, backend="matplotlib")

    assert len(mpl_fig.axes[0].containers) == len(plotly_fig.data)
    for trace, container in zip(plotly_fig.data, mpl_fig.axes[0].containers, strict=True):
        assert sum(patch.get_height() for patch in container) == pytest.approx(len(trace.x))


def test_distribution_puts_one_panel_per_asset(data) -> None:
    """The by-period chart is small multiples, one panel per asset."""
    plotly_fig = data.plots.distribution(backend="plotly")
    mpl_fig = data.plots.distribution(backend="matplotlib")

    assert len(mpl_fig.axes) == len(data.assets)
    assert [ax.get_title() for ax in mpl_fig.axes] == data.assets
    assert [a.text for a in plotly_fig.layout.annotations] == data.assets


def test_distribution_boxes_summarise_the_same_periods(data) -> None:
    """Each panel holds one box per holding period, in the same order."""
    periods = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
    mpl_fig = data.plots.distribution(backend="matplotlib")

    for ax in mpl_fig.axes:
        assert [t.get_text() for t in ax.get_xticklabels()] == periods


def test_distribution_panels_share_a_vertical_scale(data) -> None:
    """Sharing the y-axis is what makes the panels comparable."""
    axes = data.plots.distribution(backend="matplotlib").axes
    first = axes[0].get_ylim()
    assert all(ax.get_ylim() == first for ax in axes[1:])


def test_montecarlo_backends_draw_the_same_paths(data) -> None:
    """The fan and the observed path match across backends.

    The simulation is seeded, so the same data yields the same bundle every
    time and the two backends can be compared point for point.
    """
    plotly_fig = data.plots.montecarlo(n=4, period=30, backend="plotly")
    mpl_fig = data.plots.montecarlo(n=4, period=30, backend="matplotlib")

    lines = mpl_fig.axes[0].get_lines()
    assert len(lines) == len(plotly_fig.data)
    for trace, line in zip(plotly_fig.data, lines, strict=True):
        assert _floats(trace.y) == pytest.approx(_floats(line.get_ydata()), nan_ok=True)


def test_montecarlo_names_each_bundle_once(data) -> None:
    """Hundreds of paths must not become hundreds of legend entries."""
    plotly_fig = data.plots.montecarlo(n=6, period=30, backend="plotly")
    shown = [trace for trace in plotly_fig.data if trace.showlegend is not False]
    # One "Sim" entry and one "Observed" entry per asset.
    assert len(shown) == 2 * len(data.assets)


def test_montecarlo_distribution_marks_the_observed_value(data) -> None:
    """The observed metric is marked against the simulated distribution."""
    plotly_fig = data.plots.montecarlo_distribution(n=40, period=30, backend="plotly")
    mpl_fig = data.plots.montecarlo_distribution(n=40, period=30, backend="matplotlib")

    plotly_labels = [a.text for a in plotly_fig.layout.annotations]
    mpl_labels = [t.get_text() for t in mpl_fig.axes[0].texts]
    assert plotly_labels == mpl_labels
    assert all(label.endswith(" observed") for label in mpl_labels)


@pytest.mark.parametrize("metric", ["sharpe", "drawdown", "cagr"])
def test_montecarlo_distribution_supports_every_metric(data, metric: str) -> None:
    """Each metric renders on both backends and labels its own axis."""
    titles = {"sharpe": "Sharpe Ratio", "drawdown": "Max Drawdown", "cagr": "CAGR"}
    plotly_fig = data.plots.montecarlo_distribution(n=20, period=30, metric=metric, backend="plotly")
    mpl_fig = data.plots.montecarlo_distribution(n=20, period=30, metric=metric, backend="matplotlib")

    assert plotly_fig.layout.xaxis.title.text == titles[metric]
    assert mpl_fig.axes[0].get_xlabel() == titles[metric]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n": 0}, "n must be a positive integer"),
        ({"period": 0}, "period must be a positive integer"),
        ({"metric": "not-a-metric"}, "metric must be one of"),
    ],
    ids=["n", "period", "metric"],
)
def test_montecarlo_validation_is_backend_independent(data, kwargs: dict, message: str) -> None:
    """Validation sits in the builder, so both backends reject alike."""
    for backend in ("plotly", "matplotlib"):
        with pytest.raises(ValueError, match=message):
            data.plots.montecarlo_distribution(**kwargs, backend=backend)


def test_drawdown_backends_plot_the_same_curves(data) -> None:
    """The underwater curve matches, reference line excluded.

    matplotlib's ``axhline`` is itself a Line2D, so it has to be filtered out
    before the series line up with Plotly's traces.
    """
    plotly_fig = data.plots.drawdown(backend="plotly")
    mpl_fig = data.plots.drawdown(backend="matplotlib")

    series = [line for line in mpl_fig.axes[0].get_lines() if line.get_label() in {t.name for t in plotly_fig.data}]
    assert len(series) == len(plotly_fig.data)

    for trace, line in zip(plotly_fig.data, series, strict=True):
        assert _floats(trace.y) == pytest.approx(_floats(line.get_ydata()), nan_ok=True)


def test_drawdown_is_filled_on_both_backends(data) -> None:
    """The area between the curve and zero is shaded, not just outlined."""
    plotly_fig = data.plots.drawdown(backend="plotly")
    assert all(trace.fill == "tozeroy" for trace in plotly_fig.data)

    mpl_fig = data.plots.drawdown(backend="matplotlib")
    assert len(mpl_fig.axes[0].collections) == len(plotly_fig.data)


def test_drawdown_marks_break_even_on_both_backends(data) -> None:
    """A line at zero separates the underwater region from the surface."""
    plotly_fig = data.plots.drawdown(backend="plotly")
    assert any(shape.y0 == 0 for shape in plotly_fig.layout.shapes)

    mpl_fig = data.plots.drawdown(backend="matplotlib")
    flat = [ln for ln in mpl_fig.axes[0].get_lines() if list(ln.get_ydata()) == [0, 0]]
    assert flat, "expected a horizontal reference line at zero"


@pytest.mark.parametrize("n", [1, 3, 5])
def test_drawdown_periods_shade_the_same_episodes(data, n: int) -> None:
    """Both backends shade the same number of episodes, and label them."""
    plotly_fig = data.plots.drawdowns_periods(n=n, backend="plotly")
    mpl_fig = data.plots.drawdowns_periods(n=n, backend="matplotlib")

    plotly_bands = [s for s in plotly_fig.layout.shapes if s.type == "rect"]
    mpl_bands = mpl_fig.axes[0].patches

    assert len(plotly_bands) == len(mpl_bands) <= n
    assert len(mpl_fig.axes[0].texts) == len(mpl_bands)


def test_drawdown_period_labels_match(data) -> None:
    """The rank-and-depth labels read the same on both backends."""
    plotly_fig = data.plots.drawdowns_periods(n=3, backend="plotly")
    mpl_fig = data.plots.drawdowns_periods(n=3, backend="matplotlib")

    plotly_labels = [a.text for a in plotly_fig.layout.annotations]
    mpl_labels = [t.get_text() for t in mpl_fig.axes[0].texts]
    assert plotly_labels == mpl_labels
    assert mpl_labels[0].startswith("#1 ")


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


def test_multi_panel_specs_render_side_by_side() -> None:
    """Panels are laid out in a row, one Axes each.

    This replaces an earlier assertion that multi-panel specs were rejected;
    the by-period distribution chart is the first that needs them.
    """
    panel = Panel(lines=(_line(),))
    fig = render_mpl(FigureSpec(title="T", panels=(panel, panel), arrangement="side_by_side"))
    assert len(fig.axes) == 2


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
