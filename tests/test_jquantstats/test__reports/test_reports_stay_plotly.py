"""The HTML reports stay Plotly whatever the global backend is set to.

Reports embed Plotly JSON and link plotly.js, so they cannot use a matplotlib
figure. Nothing about that fails loudly if it regresses: the report builders
wrap each chart in a try/except that turns a failure into an empty string or a
"Chart unavailable" notice, so a report would simply come out with no charts.
These tests are the alarm for that.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

import jquantstats
from jquantstats import Portfolio


@pytest.fixture
def simple_portfolio() -> Portfolio:
    """20-day single-asset Portfolio, enough to render every report chart."""
    n = 20
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    return Portfolio.from_cash_position(
        prices=pl.DataFrame({"date": dates, "A": pl.Series([100.0 * (1.005**i) for i in range(n)])}),
        cash_position=pl.DataFrame({"date": dates, "A": pl.Series([1000.0] * n, dtype=pl.Float64)}),
        aum=1e5,
    )


def test_data_report_renders_every_chart_under_a_matplotlib_default(data) -> None:
    """`Data.reports.full()` renders all eight charts even so.

    The count is the whole test. A matplotlib figure reaching this path does
    not raise — `_try_plotly_div` swallows the failure and returns "" — so the
    chart just disappears while the other seven still prove "Plotly is in the
    output". Asserting the total is what makes the failure visible: without the
    backend pinned, this is 7.
    """
    jquantstats.set_plot_backend("matplotlib")
    html = data.reports.full()

    assert "cdn.plot.ly" in html
    assert html.count("Plotly.newPlot") == 8


def test_portfolio_report_embeds_plotly_under_a_matplotlib_default(simple_portfolio) -> None:
    """`Portfolio.report.to_html()` renders Plotly charts even so."""
    jquantstats.set_plot_backend("matplotlib")
    html = simple_portfolio.report.to_html()

    assert "cdn.plot.ly" in html
    # The literal notice, not the class name — the stylesheet defines
    # `.chart-unavailable` whether or not any chart actually failed.
    assert '<p class="chart-unavailable">' not in html
    # All eight portfolio charts, so a partial failure is caught too.
    assert html.count("Plotly.newPlot") == 8


def test_result_writes_plotly_html_under_a_matplotlib_default(simple_portfolio, tmp_path) -> None:
    """`Result.create_reports()` still writes Plotly documents.

    `write_html` exists only on a Plotly figure, so a matplotlib figure reaching
    this path raises AttributeError rather than degrading quietly.
    """
    jquantstats.set_plot_backend("matplotlib")
    jquantstats.Result(portfolio=simple_portfolio).create_reports(tmp_path)

    snapshot = (tmp_path / "plots" / "snapshot.html").read_text(encoding="utf-8")
    assert "cdn.plot.ly" in snapshot


def test_reports_restore_the_caller_s_backend(data) -> None:
    """Pinning Plotly inside a report must not leak out of it."""
    jquantstats.set_plot_backend("matplotlib")
    data.reports.full()
    assert jquantstats.get_plot_backend() == "matplotlib"
