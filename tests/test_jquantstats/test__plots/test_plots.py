"""Tests for the _plots subpackage (Data.plots and Portfolio.plots facades)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from jquantstats import Data, Portfolio

# ─── Data.plots tests ────────────────────────────────────────────────────────


@pytest.fixture
def plots(data):
    """Fixture that returns the plots property of the data fixture.

    Args:
        data: The data fixture containing a Data object.

    Returns:
        DataPlots: The plots property of the data fixture.

    """
    return data.plots


def test_plot_snapshot(plots):
    """Tests that the plot_snapshot method works correctly.

    Args:
        plots: The plots fixture.

    Verifies:
        1. The method returns a plotly Figure object.
        2. The method doesn't raise any exceptions.
        3. The method works with different parameters.

    """
    # Test with default parameters
    fig = plots.snapshot()
    assert fig is not None
    assert hasattr(fig, "show")

    # Test with custom parameters
    fig = plots.snapshot(title="Custom Title", log_scale=True)
    assert fig is not None
    assert hasattr(fig, "show")

    # causing sometimes problems
    # fig.show()


def test_plot_snapshot_one_symbol(returns):
    """Tests that the plot_snapshot method works correctly with a single symbol.

    Args:
        returns: The returns fixture containing a DataFrame with a single symbol.

    Verifies:
        1. The method returns a plotly Figure object.
        2. The method doesn't raise any exceptions when working with a single symbol.

    """
    fig = Data.from_returns(returns=returns).plots.snapshot()

    assert fig is not None
    assert hasattr(fig, "show")
    # causing sometimes problems
    # fig.show()


def test_repr(plots):
    """Tests that DataPlots.__repr__ returns an informative string."""
    r = repr(plots)
    assert r.startswith("DataPlots(assets=")
    for asset in plots.assets:
        assert asset in r


def test_montecarlo_plot_returns_figure_with_simulations(data):
    """Montecarlo returns a fan chart with one observed line per asset."""
    n = 8
    fig = data.plots.montecarlo(n=n, period=30, title="Monte Carlo Fan", figsize=(920, 420))

    assert isinstance(fig, go.Figure)
    n_assets = len(data.all.columns) - 1
    assert len(fig.data) == (n + 1) * n_assets
    assert sum(1 for trace in fig.data if "Observed" in (trace.name or "")) == n_assets
    assert fig.layout.title.text == "Monte Carlo Fan"
    assert fig.layout.width == 920
    assert fig.layout.height == 420
    assert fig.layout.xaxis.rangeselector is not None
    _ = fig.to_dict()


def test_montecarlo_distribution_returns_histograms_and_observed_lines(data):
    """montecarlo_distribution renders one histogram and observed marker per asset."""
    fig = data.plots.montecarlo_distribution(n=25, period=63, metric="sharpe")

    assert isinstance(fig, go.Figure)
    n_assets = len(data.all.columns) - 1
    assert len(fig.data) == n_assets
    assert all(isinstance(trace, go.Histogram) for trace in fig.data)
    assert len(fig.layout.shapes) >= n_assets
    assert fig.layout.xaxis.title.text == "Sharpe Ratio"
    _ = fig.to_dict()


def test_montecarlo_invalid_n_raises(data):
    """Montecarlo raises ValueError when n <= 0."""
    with pytest.raises(ValueError, match="n must be a positive integer"):
        data.plots.montecarlo(n=0, period=30)


def test_montecarlo_invalid_period_raises(data):
    """Montecarlo raises ValueError when period <= 0."""
    with pytest.raises(ValueError, match="period must be a positive integer"):
        data.plots.montecarlo(n=5, period=0)


def test_montecarlo_distribution_invalid_n_raises(data):
    """montecarlo_distribution raises ValueError when n <= 0."""
    with pytest.raises(ValueError, match="n must be a positive integer"):
        data.plots.montecarlo_distribution(n=0, period=30)


def test_montecarlo_distribution_invalid_period_raises(data):
    """montecarlo_distribution raises ValueError when period <= 0."""
    with pytest.raises(ValueError, match="period must be a positive integer"):
        data.plots.montecarlo_distribution(n=5, period=0)


def test_montecarlo_distribution_cagr_metric(data):
    """montecarlo_distribution renders histograms for the cagr metric."""
    fig = data.plots.montecarlo_distribution(n=10, period=30, metric="cagr")
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "CAGR"
    _ = fig.to_dict()


def test_montecarlo_distribution_invalid_metric_raises(data):
    """montecarlo_distribution rejects unsupported metric names."""
    with pytest.raises(ValueError, match="metric"):
        _ = data.plots.montecarlo_distribution(metric="not-a-metric")


# ─── Portfolio.plots tests ────────────────────────────────────────────────────


def test_snapshot_returns_figure_with_expected_traces_and_log_scale(pf: Portfolio):
    """Snapshot should return a 2-trace figure and honor log_scale on y-axis."""
    fig = pf.plots.snapshot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4
    names = {trace.name for trace in fig.data}
    assert {"NAV", "Drawdown"}.issubset(names)
    assert fig.layout.title.text
    assert "Performance" in fig.layout.title.text
    assert fig.layout.hovermode in ("x unified", "x", "x unified")

    fig_log = pf.plots.snapshot(log_scale=True)
    assert isinstance(fig_log, go.Figure)
    assert getattr(fig_log.layout.yaxis, "type", None) == "log"
    _ = fig_log.to_dict()


def test_snapshot_with_cost_per_unit_includes_net_nav_trace(pf: Portfolio):
    """snapshot() adds a Net-of-Cost NAV trace when cost_per_unit > 0."""
    from jquantstats import Portfolio as _Portfolio

    pf_with_cost = _Portfolio.from_cash_position(
        prices=pf.prices,
        cash_position=pf.cashposition,
        aum=pf.aum,
        cost_per_unit=0.01,
    )
    fig = pf_with_cost.plots.snapshot()
    assert isinstance(fig, go.Figure)
    names = {trace.name for trace in fig.data}
    assert "Net-of-Cost NAV" in names
    _ = fig.to_dict()


# ─── Lagged performance ───────────────────────────────────────────────────────


def test_lagged_performance_plot_traces_and_log_scale(pf):
    """lagged_performance_plot returns 5 traces by default and supports log scale."""
    fig = pf.plots.lagged_performance_plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 5
    assert [t.name for t in fig.data] == [f"lag {i}" for i in range(5)]
    for tr in fig.data:
        assert len(tr.x) == len(tr.y) > 0

    fig_log = pf.plots.lagged_performance_plot(log_scale=True)
    assert getattr(fig_log.layout.yaxis, "type", None) == "log"
    _ = fig_log.to_dict()


def test_lagged_performance_plot_type_validation_raises(pf):
    """Lags must be a list of ints; other types or contents should raise TypeError."""
    with pytest.raises(TypeError):
        _ = pf.plots.lagged_performance_plot(lags=(0, 1, 2))
    with pytest.raises(TypeError):
        _ = pf.plots.lagged_performance_plot(lags=[0, "1", 2])


# ─── Smoothed holdings ────────────────────────────────────────────────────────


def test_smoothed_holdings_performance_plot_traces_and_log_scale(pf):
    """smoothed_holdings_performance_plot returns 5 traces and supports log scale."""
    fig = pf.plots.smoothed_holdings_performance_plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 5
    assert [t.name for t in fig.data] == [f"smooth {i}" for i in range(5)]
    for tr in fig.data:
        assert len(tr.x) == len(tr.y) > 0

    fig_log = pf.plots.smoothed_holdings_performance_plot(log_scale=True)
    assert getattr(fig_log.layout.yaxis, "type", None) == "log"
    _ = fig_log.to_dict()


def test_smoothed_holdings_performance_plot_type_validation_raises(pf):
    """Windows must be a list of non-negative ints; invalid inputs raise TypeError."""
    with pytest.raises(TypeError):
        _ = pf.plots.smoothed_holdings_performance_plot(windows=(0, 1, 2))
    with pytest.raises(TypeError):
        _ = pf.plots.smoothed_holdings_performance_plot(windows=[0, -1, 2])
    with pytest.raises(TypeError):
        _ = pf.plots.smoothed_holdings_performance_plot(windows=[0, "2"])


# ─── Lead/lag IR ─────────────────────────────────────────────────────────────


def test_lead_lag_ir_plot_basic_structure_and_values(pf):
    """lead_lag_ir_plot returns a Figure with bars for lags -10..+19 and valid values."""
    fig = pf.plots.lead_lag_ir_plot()
    bar = fig.data[0]
    expected_lags = list(range(-10, 20))
    assert list(bar.x) == expected_lags
    assert len(list(bar.y)) == len(expected_lags)

    for lag in (-10, 0, 5, 19):
        pf_lagged = pf if lag == 0 else pf.lag(lag)
        sharpe_n = pf_lagged.stats.sharpe()["returns"]
        assert np.isclose(list(bar.y)[expected_lags.index(lag)], sharpe_n, rtol=1e-12, atol=1e-12)

    _ = fig.to_dict()


def test_lead_lag_ir_plot_swaps_when_start_greater_than_end(pf):
    """When start > end, the function swaps them and still plots inclusive range."""
    fig = pf.plots.lead_lag_ir_plot(start=5, end=-5)
    assert list(fig.data[0].x) == list(range(-5, 6))


def test_lead_lag_ir_plot_type_validation_raises(pf):
    """Non-integer start/end should raise TypeError in lead_lag_ir_plot."""
    with pytest.raises(TypeError):
        _ = pf.plots.lead_lag_ir_plot(start=-10.0, end=10)
    with pytest.raises(TypeError):
        _ = pf.plots.lead_lag_ir_plot(start=-10, end="19")


# ─── Correlation heatmap ──────────────────────────────────────────────────────


def test_correlation_heatmap_default_trace_and_serialize(pf):
    """Default call returns Heatmap trace and is serializable; axes align."""
    fig = pf.plots.correlation_heatmap()

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert isinstance(fig.data[0], go.Heatmap)

    x_labels = list(fig.data[0].x)
    y_labels = list(fig.data[0].y)
    assert x_labels == y_labels
    assert len(x_labels) >= 2

    flat = [v for row in fig.data[0].z for v in row]
    assert all(-1.000001 <= float(v) <= 1.000001 for v in flat)

    _ = fig.to_dict()


def test_correlation_heatmap_custom_args_title_and_name(pf):
    """Custom frame/name/title are respected and output remains a Heatmap."""
    custom_title = "My Correlations"
    fig = pf.plots.correlation_heatmap(frame=pf.prices, name="my_port", title=custom_title)

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == custom_title
    assert isinstance(fig.data[0], go.Heatmap)
    _ = fig.to_dict()


# ─── Fixtures for rolling / multi-year tests ─────────────────────────────────


@pytest.fixture
def long_portfolio() -> Portfolio:
    """Three-year pf used to test rolling and annual-breakdown plots."""
    n = 756  # ~3 years of trading days
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    a = pl.Series([100.0 * (1.001**i) for i in range(n)], dtype=pl.Float64)
    b = pl.Series([200.0 + 5.0 * np.sin(0.1 * i) for i in range(n)], dtype=pl.Float64)
    prices = pl.DataFrame({"date": dates, "A": a, "B": b})

    pos_a = pl.Series([1000.0 + float(i % 10) for i in range(n)], dtype=pl.Float64)
    pos_b = pl.Series([500.0 + float(i % 5) for i in range(n)], dtype=pl.Float64)
    positions = pl.DataFrame({"date": dates, "A": pos_a, "B": pos_b})

    return Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)


# ─── Rolling Sharpe plot ──────────────────────────────────────────────────────


def test_rolling_sharpe_plot_returns_figure_with_traces(long_portfolio):
    """rolling_sharpe_plot returns a Figure with one trace per asset."""
    fig = long_portfolio.plots.rolling_sharpe_plot(window=63)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    for trace in fig.data:
        assert isinstance(trace, go.Scatter)
    assert "Rolling Sharpe" in fig.layout.title.text
    assert fig.layout.xaxis.rangeselector is not None
    _ = fig.to_dict()


def test_rolling_sharpe_plot_invalid_window_raises(long_portfolio):
    """rolling_sharpe_plot should raise ValueError for non-positive window."""
    with pytest.raises(ValueError, match=r".*"):
        _ = long_portfolio.plots.rolling_sharpe_plot(window=0)
    with pytest.raises(ValueError, match=r".*"):
        _ = long_portfolio.plots.rolling_sharpe_plot(window=-1)


# ─── Rolling Volatility plot ──────────────────────────────────────────────────


def test_rolling_volatility_plot_returns_figure_with_traces(long_portfolio):
    """rolling_volatility_plot returns a Figure with one trace per asset."""
    fig = long_portfolio.plots.rolling_volatility_plot(window=63)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    for trace in fig.data:
        assert isinstance(trace, go.Scatter)
    assert "Rolling Volatility" in fig.layout.title.text
    assert fig.layout.xaxis.rangeselector is not None
    _ = fig.to_dict()


def test_rolling_volatility_plot_invalid_window_raises(long_portfolio):
    """rolling_volatility_plot should raise ValueError for non-positive window."""
    with pytest.raises(ValueError, match=r".*"):
        _ = long_portfolio.plots.rolling_volatility_plot(window=0)


# ─── Annual Sharpe plot ───────────────────────────────────────────────────────


def test_annual_sharpe_plot_returns_figure_with_bars(long_portfolio):
    """annual_sharpe_plot returns a bar Figure with year labels on x-axis."""
    fig = long_portfolio.plots.annual_sharpe_plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    for trace in fig.data:
        assert isinstance(trace, go.Bar)
    assert "Annual Sharpe" in fig.layout.title.text
    _ = fig.to_dict()
