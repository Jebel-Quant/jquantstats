"""Unit tests for the figure spec and the Plotly renderer.

The chart-level behaviour of the cumulative family is already pinned by
``test_plots.py`` and the two snapshot suites. These tests cover the seam
itself: that a spec builder describes what it should, and that the renderer
translates each semantic property into the right Plotly vocabulary — including
the properties it must *not* emit when a builder left them unset.
"""

from __future__ import annotations

import json

import plotly.graph_objects as go
import polars as pl
import pytest

from jquantstats._plots._render import render_plotly
from jquantstats._plots._render._plotly import _axis_kwargs, _hovertemplate
from jquantstats._plots._spec import (
    Axis,
    Band,
    BarSeries,
    FigureSpec,
    HeatmapGrid,
    HoverSpec,
    LineSeries,
    Panel,
    RefLine,
)
from jquantstats._plots._specs import (
    compare_spec,
    cumulative_returns_spec,
    earnings_spec,
    log_returns_spec,
)
from jquantstats._plots._style import hex_to_rgba, ticker_colors


def _line(**overrides) -> LineSeries:
    """Build a minimal line series, overriding selected fields."""
    defaults = {
        "name": "AAPL",
        "x": pl.Series("date", [1, 2, 3]),
        "y": pl.Series("AAPL", [1.0, 1.1, 1.2]),
        "color": "#636efa",
    }
    return LineSeries(**{**defaults, **overrides})


def _spec(panel: Panel, **overrides) -> FigureSpec:
    """Build a single-panel figure spec, overriding selected fields."""
    return FigureSpec(**{"title": "T", "panels": (panel,), **overrides})


def _grid(**overrides) -> HeatmapGrid:
    """Build a minimal 1x2 heatmap grid, overriding selected fields."""
    defaults = {
        "x_labels": ("Jan", "Feb"),
        "y_labels": ("2024",),
        "z": ((1.0, -1.0),),
        "text": (("1.0%", "-1.0%"),),
        "colorscale": "red_white_green",
    }
    return HeatmapGrid(**{**defaults, **overrides})


# ── the spec is backend-neutral ───────────────────────────────────────────────


def test_spec_holds_polars_series_not_lists(data) -> None:
    """Series must stay as polars Series through the spec.

    Plotly serialises a Series to a compact binary buffer and a Python list to
    a plain JSON array, so a conversion anywhere in the spec layer would
    silently change every rendered figure.
    """
    spec = cumulative_returns_spec(data, title="T", log_scale=False)
    line = spec.panels[0].lines[0]
    assert isinstance(line.x, pl.Series)
    assert isinstance(line.y, pl.Series)


def test_spec_dataclasses_are_frozen() -> None:
    """A spec is a description, not a mutable builder."""
    with pytest.raises(AttributeError):
        _line().name = "other"


# ── style helpers ────────────────────────────────────────────────────────────


def test_ticker_colors_are_stable_and_wrap() -> None:
    """Colours are positional and wrap once the palette is exhausted."""
    many = [f"T{i}" for i in range(12)]
    colors = ticker_colors(many)
    assert colors["T0"] == ticker_colors(["T0"])["T0"]
    assert colors["T10"] == colors["T0"], "palette of 10 should wrap"


def test_hex_to_rgba_accepts_both_spellings() -> None:
    """A leading '#' is optional."""
    assert hex_to_rgba("#636efa", 0.4) == hex_to_rgba("636efa", 0.4)


# ── hovertemplate translation ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("hover", "expected"),
    [
        (
            HoverSpec(label="AAPL", value_format="float2", suffix="x"),
            "<b>%{x|%b %Y}</b><br>AAPL: %{y:.2f}x",
        ),
        (
            HoverSpec(label="AAPL", value_format="float4"),
            "<b>%{x|%b %Y}</b><br>AAPL: %{y:.4f}",
        ),
        (
            HoverSpec(label="AAPL", value_format="currency0", prefix="$"),
            "<b>%{x|%b %Y}</b><br>AAPL: $%{y:,.0f}",
        ),
        (
            HoverSpec(label="AAPL", value_format="float2", date_header=False),
            "AAPL: %{y:.2f}",
        ),
    ],
)
def test_hovertemplate_translation(hover: HoverSpec, expected: str) -> None:
    """Each semantic hover field maps onto the expected Plotly template."""
    assert _hovertemplate(hover) == expected


# ── axis translation ─────────────────────────────────────────────────────────


def test_unset_axis_emits_nothing() -> None:
    """An untouched axis contributes no properties.

    Emitting a default for a property the chart never set would change the
    serialised figure.
    """
    assert _axis_kwargs(Axis()) == {}


def test_axis_translates_every_property() -> None:
    """Each semantic axis field maps onto the expected Plotly property."""
    kwargs = _axis_kwargs(Axis(title="V", tick_format="currency0", tick_prefix="$", log=True))
    assert kwargs == {"title_text": "V", "tickprefix": "$", "tickformat": ",.0f", "type": "log"}


def test_opposite_side_resolves_per_axis() -> None:
    """The far edge is the top for an x-axis and the right for a y-axis.

    The spec says `opposite_side` rather than naming a compass point precisely
    so that it cannot express an x-axis on the "left"; each renderer resolves
    it against the axis it is configuring.
    """
    assert _axis_kwargs(Axis(opposite_side=True))["side"] == "top"
    assert _axis_kwargs(Axis(opposite_side=True), vertical=True)["side"] == "right"
    assert "side" not in _axis_kwargs(Axis())


# ── renderer ─────────────────────────────────────────────────────────────────


def test_render_rejects_multi_panel_specs() -> None:
    """Multi-panel rendering arrives with the dashboards, not before."""
    panel = Panel(lines=(_line(),))
    with pytest.raises(ValueError, match="too many values"):
        render_plotly(FigureSpec(title="T", panels=(panel, panel)))


def test_unset_dash_omits_the_property() -> None:
    """A line that says nothing about its dash gets nothing emitted."""
    fig = render_plotly(_spec(Panel(lines=(_line(dash=None),))))
    assert "dash" not in json.loads(fig.to_json())["data"][0]["line"]


def test_explicitly_solid_dash_is_stated() -> None:
    """Asking for solid is not the same as saying nothing.

    Some charts write ``dash="solid"`` and some omit the property entirely.
    Plotly records the difference, so the spec has to be able to express both:
    None means "leave it alone", ``"solid"`` means "state it".
    """
    fig = render_plotly(_spec(Panel(lines=(_line(dash="solid"),))))
    assert json.loads(fig.to_json())["data"][0]["line"]["dash"] == "solid"


def test_dashed_lines_state_the_dash_property() -> None:
    """A non-default stroke pattern is emitted."""
    fig = render_plotly(_spec(Panel(lines=(_line(dash="dash"),))))
    assert json.loads(fig.to_json())["data"][0]["line"]["dash"] == "dash"


def test_line_without_hover_emits_no_hovertemplate() -> None:
    """A series may opt out of a tooltip entirely."""
    fig = render_plotly(_spec(Panel(lines=(_line(hover=None),))))
    assert "hovertemplate" not in json.loads(fig.to_json())["data"][0]


def test_bar_without_hover_emits_no_hovertemplate() -> None:
    """Bars may opt out of a tooltip too."""
    bar = BarSeries(
        name="AAPL",
        x=pl.Series("x", [1, 2]),
        y=pl.Series("y", [0.1, -0.2]),
        colors=("#2ca02c", "#d62728"),
    )
    fig = render_plotly(_spec(Panel(bars=(bar,))))
    assert "hovertemplate" not in json.loads(fig.to_json())["data"][0]


def test_heatmap_without_hover_label_emits_no_hovertemplate() -> None:
    """A matrix may opt out of a tooltip too."""
    fig = render_plotly(_spec(Panel(heatmap=_grid(hover_label=None)), chrome="bare"))
    assert "hovertemplate" not in json.loads(fig.to_json())["data"][0]


def test_bare_chrome_omits_the_timeseries_furniture() -> None:
    """A matrix chart gets no legend, unified hover or range selector."""
    layout = json.loads(render_plotly(_spec(Panel(heatmap=_grid()), chrome="bare")).to_json())["layout"]
    assert "hovermode" not in layout
    assert "legend" not in layout
    assert "rangeselector" not in layout.get("xaxis", {})
    assert layout["plot_bgcolor"] == "white"


def test_horizontal_reference_line_sits_at_its_value() -> None:
    """A break-even marker is a horizontal line at the given value."""
    fig = render_plotly(_spec(Panel(lines=(_line(),), ref_lines=(RefLine(value=0),))))
    (shape,) = fig.layout.shapes
    assert (shape.y0, shape.y1) == (0, 0)
    assert shape.line.dash is None, "solid is the default and is left unstated"


def test_vertical_reference_line_sits_at_its_value() -> None:
    """The same marker can run the other way."""
    fig = render_plotly(_spec(Panel(lines=(_line(),), ref_lines=(RefLine(value=2, orientation="v"),))))
    (shape,) = fig.layout.shapes
    assert (shape.x0, shape.x1) == (2, 2)


def test_dashed_reference_line_states_its_dash() -> None:
    """A non-default stroke pattern is emitted."""
    fig = render_plotly(_spec(Panel(lines=(_line(),), ref_lines=(RefLine(value=1, dash="dash"),))))
    assert fig.layout.shapes[0].line.dash == "dash"


def test_band_without_a_label_draws_no_annotation() -> None:
    """A span may shade without naming itself."""
    fig = render_plotly(_spec(Panel(lines=(_line(),), bands=(Band(x0=1, x1=2, color="rgba(0,0,0,0.2)"),))))
    assert not fig.layout.annotations
    assert len(fig.layout.shapes) == 1


def test_band_with_a_label_annotates_it() -> None:
    """A labelled span carries its text."""
    band = Band(x0=1, x1=2, color="rgba(0,0,0,0.2)", label="#1 -12.3%")
    fig = render_plotly(_spec(Panel(lines=(_line(),), bands=(band,))))
    assert [a.text for a in fig.layout.annotations] == ["#1 -12.3%"]


def test_unfilled_lines_omit_the_fill_properties() -> None:
    """Only the underwater curve is filled; the rest stay outlines."""
    payload = json.loads(render_plotly(_spec(Panel(lines=(_line(),)))).to_json())["data"][0]
    assert "fill" not in payload
    assert "fillcolor" not in payload


def test_filled_lines_shade_to_zero() -> None:
    """A fill colour shades the area between the line and zero."""
    fig = render_plotly(_spec(Panel(lines=(_line(fill_color="rgba(0,0,0,0.3)"),))))
    assert fig.data[0].fill == "tozeroy"
    assert fig.data[0].fillcolor == "rgba(0,0,0,0.3)"


def test_bar_mode_is_only_emitted_when_asked_for() -> None:
    """Plotly's default grouping stands unless a spec overrides it."""
    bar = BarSeries(name="A", x=pl.Series("x", [1]), y=pl.Series("y", [1.0]), colors=("#000000",))
    without = json.loads(render_plotly(_spec(Panel(bars=(bar,)))).to_json())["layout"]
    with_mode = json.loads(render_plotly(_spec(Panel(bars=(bar,)), bar_mode="group")).to_json())["layout"]
    assert "barmode" not in without
    assert with_mode["barmode"] == "group"


def test_figsize_sets_width_and_height() -> None:
    """`figsize` is in pixels and overrides the default height."""
    fig = render_plotly(_spec(Panel(lines=(_line(),)), figsize=(920, 420)))
    assert (fig.layout.width, fig.layout.height) == (920, 420)


def test_height_applies_when_no_figsize_given() -> None:
    """Without `figsize` the spec height stands and no width is set."""
    fig = render_plotly(_spec(Panel(lines=(_line(),)), height=480))
    assert fig.layout.height == 480
    assert fig.layout.width is None


def test_range_selector_can_be_suppressed() -> None:
    """Charts without a date axis opt out of the range selector."""
    fig = render_plotly(_spec(Panel(lines=(_line(),)), date_range_selector=False))
    assert "rangeselector" not in json.loads(fig.to_json())["layout"]["xaxis"]


def test_range_selector_is_attached_by_default() -> None:
    """Date-axis charts get the 6m/1y/3y/YTD/All buttons."""
    fig = render_plotly(_spec(Panel(lines=(_line(),))))
    assert json.loads(fig.to_json())["layout"]["xaxis"]["rangeselector"]["buttons"]


def test_render_returns_a_plotly_figure() -> None:
    """The Plotly renderer produces a Plotly figure."""
    assert isinstance(render_plotly(_spec(Panel(lines=(_line(),)))), go.Figure)


def test_both_axes_are_honoured() -> None:
    """A configured x-axis reaches the figure, not just the y-axis.

    No chart in the cumulative family configures its x-axis, so without this
    the x-axis path would be an untested no-op — and a builder that later set
    an x-axis title would find it silently dropped.
    """
    panel = Panel(lines=(_line(),), xaxis=Axis(title="When"), yaxis=Axis(title="How much"))
    fig = render_plotly(_spec(panel))
    assert fig.layout.xaxis.title.text == "When"
    assert fig.layout.yaxis.title.text == "How much"


def test_unconfigured_axes_stay_untouched() -> None:
    """Neither axis gains a title when the spec sets none."""
    fig = render_plotly(_spec(Panel(lines=(_line(),))))
    assert fig.layout.xaxis.title.text is None
    assert fig.layout.yaxis.title.text is None


# ── builders describe what the charts need ───────────────────────────────────


def test_cumulative_spec_marks_growth_multiples(data) -> None:
    """The 'x' suffix reads the value as a growth multiple."""
    spec = cumulative_returns_spec(data, title="T", log_scale=False)
    assert all(line.hover.suffix == "x" for line in spec.panels[0].lines)


def test_cumulative_spec_honours_log_scale(data) -> None:
    """`log_scale` reaches the y-axis rather than the data."""
    assert cumulative_returns_spec(data, title="T", log_scale=True).panels[0].yaxis.log is True
    assert cumulative_returns_spec(data, title="T", log_scale=False).panels[0].yaxis.log is False


def test_compare_spec_draws_benchmarks_last_and_dashed(data) -> None:
    """Benchmarks read as the reference, not as another asset."""
    spec = compare_spec(data, title="T", figsize=None)
    assets = list(data.returns.columns)
    lines = spec.panels[0].lines

    assert [line.name for line in lines[: len(assets)]] == assets
    for line in lines[len(assets) :]:
        assert line.dash == "dash"
        assert line.width == 2.5


def test_compare_spec_requires_a_benchmark(data_no_benchmark) -> None:
    """Validation lives in the builder, so every backend raises alike."""
    with pytest.raises(AttributeError, match=r"^compare\(\) requires benchmark data to be set$"):
        compare_spec(data_no_benchmark, title="T", figsize=None)


def test_log_returns_spec_leaves_the_tick_format_alone(data) -> None:
    """Log values carry no tick format, matching the original chart."""
    yaxis = log_returns_spec(data, title="T", figsize=None).panels[0].yaxis
    assert yaxis.title == "Log Return"
    assert yaxis.tick_format is None


def test_earnings_spec_labels_the_starting_balance(data) -> None:
    """The axis states the starting balance, formatted with separators."""
    yaxis = earnings_spec(data, start_balance=250_000, title="T", compounded=True).panels[0].yaxis
    assert yaxis.title == "Portfolio Value (starting $250,000)"
    assert yaxis.tick_prefix == "$"


def test_earnings_spec_compounding_changes_the_values(data) -> None:
    """`compounded` selects cumprod over cumsum, giving different curves."""
    compounded = earnings_spec(data, start_balance=1e5, title="T", compounded=True)
    summed = earnings_spec(data, start_balance=1e5, title="T", compounded=False)
    assert compounded.panels[0].lines[0].y.to_list() != summed.panels[0].lines[0].y.to_list()
