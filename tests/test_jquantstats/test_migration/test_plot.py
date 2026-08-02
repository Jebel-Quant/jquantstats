"""Parametrized smoke tests for DataPlots methods."""

import plotly.graph_objects as go
import pytest


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("snapshot", {}),
        ("snapshot", {"log_scale": True}),
        ("returns", {}),
        ("returns", {"log_scale": True}),
        ("compare", {}),
        ("log_returns", {}),
        ("daily_returns", {}),
        ("yearly_returns", {}),
        ("yearly_returns", {"compounded": False}),
        ("monthly_returns", {}),
        ("monthly_returns", {"compounded": False}),
        ("monthly_heatmap", {}),
        ("monthly_heatmap", {"asset": "AAPL"}),
        ("histogram", {}),
        ("histogram", {"bins": 20}),
        ("distribution", {}),
        ("distribution", {"compounded": False}),
        ("montecarlo", {}),
        ("montecarlo", {"n": 20, "period": 63}),
        ("montecarlo_distribution", {}),
        ("montecarlo_distribution", {"n": 100, "period": 63, "metric": "drawdown"}),
        ("drawdown", {}),
        ("drawdowns_periods", {}),
        ("drawdowns_periods", {"n": 3}),
        ("earnings", {}),
        ("earnings", {"start_balance": 1e6, "compounded": False}),
        ("rolling_sharpe", {}),
        ("rolling_sharpe", {"rolling_period": 63, "periods_per_year": 252}),
        ("rolling_sortino", {}),
        ("rolling_sortino", {"rolling_period": 63}),
        ("rolling_volatility", {}),
        ("rolling_volatility", {"rolling_period": 63}),
        ("rolling_beta", {}),
        ("rolling_beta", {"rolling_period2": None}),
    ],
)
def test_plot_returns_figure(data, method, kwargs):
    """Each plot method returns a non-empty go.Figure."""
    fig = getattr(data.plots, method)(**kwargs)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("returns", {}),
        ("compare", {}),
        ("log_returns", {}),
        ("daily_returns", {}),
        ("drawdown", {}),
    ],
)
def test_plot_includes_benchmark_trace(data, method, kwargs):
    """Plots that render all columns include a trace for the benchmark."""
    fig = getattr(data.plots, method)(**kwargs)
    trace_names = [t.name for t in fig.data]
    assert any("Benchmark" in name or "SPY" in name for name in trace_names)


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("snapshot", {}),
        ("returns", {}),
        ("compare", {}),
        ("log_returns", {}),
        ("daily_returns", {}),
        ("yearly_returns", {}),
        ("monthly_returns", {}),
        ("drawdown", {}),
        ("drawdowns_periods", {}),
    ],
)
def test_plot_has_title(data, method, kwargs):
    """Each plot has a non-empty title."""
    fig = getattr(data.plots, method)(**kwargs)
    assert fig.layout.title.text


def test_plot_monthly_heatmap_axes(data):
    """Monthly heatmap has 12 x-axis labels (Jan-Dec)."""
    fig = data.plots.monthly_heatmap()
    heatmap = fig.data[0]
    assert len(heatmap.x) == 12


def test_plot_drawdowns_periods_shading(data):
    """drawdowns_periods adds vrect shapes for the drawdown periods."""
    fig = data.plots.drawdowns_periods(n=3)
    assert len(fig.layout.shapes) > 0


def test_plot_distribution_one_subplot_per_asset(data):
    """Distribution produces 5 box traces per asset."""
    fig = data.plots.distribution()
    n_assets = len(data.assets)
    assert len(fig.data) == 5 * n_assets


def test_plot_earnings_scale(data):
    """Earnings y-values start near start_balance."""
    start = 50_000.0
    fig = data.plots.earnings(start_balance=start)
    first_y = fig.data[0].y[0]
    assert abs(first_y - start) / start < 0.05


def test_plot_rolling_beta_two_windows(data):
    """rolling_beta with two windows produces 2 traces per asset."""
    fig = data.plots.rolling_beta(rolling_period=63, rolling_period2=126)
    n_return_assets = len(data.returns.columns)
    assert len(fig.data) == 2 * n_return_assets


def test_plot_rolling_beta_no_benchmark_raises(data_no_benchmark):
    """rolling_beta raises AttributeError when no benchmark is attached."""
    with pytest.raises(AttributeError):
        data_no_benchmark.plots.rolling_beta()


def test_plot_rolling_beta_falls_back_to_frame_columns_without_returns(data):
    """rolling_beta derives assets from `all` when the data exposes no `returns`.

    `_beta_assets` prefers an explicit ``returns`` frame and otherwise takes
    every column of ``all`` that is neither the date nor the benchmark. That
    fallback is reachable through the `DataLike` protocol — an implementation
    need not carry a ``returns`` attribute — so it is exercised here with a
    proxy that hides it.
    """

    class _NoReturnsData:
        """DataLike proxy exposing everything except ``returns``."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, item):
            """Delegate every attribute except ``returns`` to the wrapped data."""
            if item == "returns":
                raise AttributeError(item)
            return getattr(self._inner, item)

    plots = type(data.plots)(_NoReturnsData(data))
    fig = plots.rolling_beta(rolling_period=63, rolling_period2=None)

    date_col = data.all.columns[0]
    bench_col = data.benchmark.columns[0]
    expected = [c for c in data.all.columns if c not in (date_col, bench_col)]
    assert len(fig.data) == len(expected)
    assert [trace.name for trace in fig.data] == [f"{asset} (63d)" for asset in expected]


def test_plot_compare_no_benchmark_raises(data_no_benchmark):
    """Compare raises AttributeError when no benchmark is attached."""
    with pytest.raises(AttributeError):
        data_no_benchmark.plots.compare()


def test_plot_figsize_applied(data):
    """log_returns/compare/rolling_beta honor explicit figsize."""
    figsize = (900, 600)
    for method in ("log_returns", "compare", "rolling_beta"):
        fig = getattr(data.plots, method)(figsize=figsize)
        assert fig.layout.width == figsize[0]
        assert fig.layout.height == figsize[1]


# ── Single-asset path (covers bar_colors = green/red branch) ─────────────────


@pytest.fixture
def data_single(resource_dir):
    """Data fixture with a single return column (Meta only, no benchmark).

    Args:
        resource_dir: Path to the test resources directory.

    Returns:
        Data: Single-asset Data object.

    """
    import polars as pl

    from jquantstats import Data

    df = pl.read_csv(resource_dir / "meta.csv", try_parse_dates=True).select(["Date", "Meta"])
    return Data.from_returns(returns=df)


def test_plot_yearly_returns_single_asset(data_single):
    """yearly_returns with one asset uses the green/red bar colour path."""
    fig = data_single.plots.yearly_returns()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_plot_monthly_returns_single_asset(data_single):
    """monthly_returns with one asset uses the green/red bar colour path."""
    fig = data_single.plots.monthly_returns()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    colors = set(fig.data[0].marker.color)
    assert colors <= {"#2ca02c", "#d62728"}


def test_plot_daily_returns_single_asset(data_single):
    """daily_returns with one asset uses the green/red bar colour path."""
    fig = data_single.plots.daily_returns()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    colors = set(fig.data[0].marker.color)
    assert colors <= {"#2ca02c", "#d62728"}
