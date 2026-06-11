"""Snapshot tests pinning the structure of reports and plots (syrupy).

Raw HTML/JSON output is not snapshot-stable (plotly embeds random div ids,
and float formatting may drift across versions), so these tests reduce each
artifact to a deterministic structural digest: section headings for the HTML
report, and trace/layout fingerprints for figures. A failing snapshot means
the *shape* of the artifact changed — new/removed sections, traces, or
titles — which is exactly what should require a deliberate review.

Regenerate after an intentional change with:

    uv run pytest tests/test_jquantstats/test_snapshots.py --snapshot-update
"""

import re
from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from jquantstats import Portfolio


@pytest.fixture
def snapshot_portfolio() -> Portfolio:
    """Deterministic 200-day, two-asset portfolio for snapshot tests."""
    n = 200
    start = date(2022, 1, 3)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    rng = np.random.default_rng(7)
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.010, n))),
            "B": pl.Series(50.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.008, n))),
        }
    )
    positions = pl.DataFrame({"date": dates, "A": pl.Series([600.0] * n), "B": pl.Series([400.0] * n)})
    return Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1_000.0)


def _figure_digest(fig: go.Figure) -> dict:
    """Reduce a Plotly figure to a stable structural fingerprint."""
    layout = fig.layout
    return {
        "traces": [{"type": trace.type, "name": trace.name} for trace in fig.data],
        "title": layout.title.text if layout.title is not None else None,
        "xaxis_title": layout.xaxis.title.text if layout.xaxis is not None and layout.xaxis.title else None,
        "yaxis_title": layout.yaxis.title.text if layout.yaxis is not None and layout.yaxis.title else None,
    }


def test_report_html_structure(snapshot_portfolio, snapshot):
    """The HTML report's section structure (headings, figure count, assets) is pinned."""
    html = snapshot_portfolio.report.to_html()
    digest = {
        "headings": re.findall(r"<h\d[^>]*>\s*([^<]+?)\s*</h\d>", html),
        "n_figures": html.count("Plotly.newPlot"),
        "assets": snapshot_portfolio.assets,
    }
    assert digest == snapshot


def test_snapshot_plot_structure(snapshot_portfolio, snapshot):
    """The NAV + drawdown snapshot figure's trace/layout structure is pinned."""
    assert _figure_digest(snapshot_portfolio.plots.snapshot()) == snapshot


def test_rolling_sharpe_plot_structure(snapshot_portfolio, snapshot):
    """The rolling Sharpe figure's trace/layout structure is pinned."""
    assert _figure_digest(snapshot_portfolio.plots.rolling_sharpe_plot()) == snapshot


def test_monthly_returns_heatmap_structure(snapshot_portfolio, snapshot):
    """The monthly returns heatmap's trace/layout structure is pinned."""
    assert _figure_digest(snapshot_portfolio.plots.monthly_returns_heatmap()) == snapshot


def test_trading_cost_impact_plot_structure(snapshot_portfolio, snapshot):
    """The trading-cost impact figure's trace/layout structure is pinned."""
    assert _figure_digest(snapshot_portfolio.plots.trading_cost_impact_plot()) == snapshot
