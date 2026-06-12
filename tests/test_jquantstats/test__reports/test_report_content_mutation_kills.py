"""Mutation-kill tests for the HTML report modules.

These tests target surviving mutants found by mutmut in
``src/jquantstats/_reports/_portfolio.py`` and
``src/jquantstats/_reports/_formatting.py`` (see ``.mutation-sweep/E.json``).

Strategy: a few broad *content* tests instead of one test per mutant.

- ``_stats_table_html`` is rendered once from a synthetic summary frame with
  hand-picked values for every metric, and the test asserts the exact
  label-cell/value-cell adjacency for each row.  Any mutation of a metric key,
  display label, format spec, suffix, best-value highlighting, or table-shell
  string changes (or drops) at least one asserted substring.
- mutmut wraps string literals in ``XX...XX``; the un-mutated stats table can
  never contain ``XX`` (labels, digits, ``%``, class names only), so a single
  ``"XX" not in table`` assertion kills every string-literal mutant on that
  path.  The full report HTML is *not* checked this way because Plotly may
  embed base64 payloads that legitimately contain ``XX``; full-report mutants
  are killed with exact, anchored substrings instead.
- Metric-key mutants in ``_METRIC_LABELS`` whose title-cased fallback equals
  the explicit label (e.g. ``win_rate`` -> "Win Rate") are output-equivalent,
  so the expected key set is pinned directly against a hard-coded list.

None of these tests require kaleido (``to_html`` never rasterises figures).
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import plotly.graph_objects as go
import polars as pl
import pytest

from jquantstats import Portfolio
from jquantstats._plots import PortfolioPlots
from jquantstats._reports import Report
from jquantstats._reports._formatting import _plotly_div, _table_html
from jquantstats._reports._portfolio import (
    _CATEGORIES,
    _METRIC_FORMATS,
    _METRIC_LABELS,
    _TEMPLATES_DIR,
    _stats_table_html,
)

# ── Expected stats-table content ──────────────────────────────────────────────
#
# (metric key, display label, value for asset A, value for asset B,
#  rendered A cell text, rendered B cell text)
#
# Values are hand-picked so that A beats B on every higher-is-better metric
# (deterministic best-value highlighting) and every rendered string is unique.
# Labels and rendered values are hard-coded on purpose: importing them from
# the module under test would make the assertions tautological under mutation.
_EXPECTED_STATS_ROWS: list[tuple[str, str, float, float, str, str]] = [
    ("avg_return", "Avg Return", 0.0111, 0.0012, "1.11%", "0.12%"),
    ("avg_win", "Avg Win", 0.0222, 0.0034, "2.22%", "0.34%"),
    ("avg_loss", "Avg Loss", -0.0133, -0.0056, "-1.33%", "-0.56%"),
    ("best", "Best Period", 0.0544, 0.0178, "5.44%", "1.78%"),
    ("worst", "Worst Period", -0.0455, -0.0189, "-4.55%", "-1.89%"),
    ("sharpe", "Sharpe Ratio", 1.61, 0.91, "1.61", "0.91"),
    ("calmar", "Calmar Ratio", 2.72, 0.82, "2.72", "0.82"),
    ("recovery_factor", "Recovery Factor", 3.83, 0.73, "3.83", "0.73"),
    ("max_drawdown", "Max Drawdown", -0.2466, -0.3211, "-24.66%", "-32.11%"),
    ("avg_drawdown", "Avg Drawdown", -0.0577, -0.0922, "-5.77%", "-9.22%"),
    ("max_drawdown_duration", "Max DD Duration", 17.0, 23.0, "17 days", "23 days"),
    ("win_rate", "Win Rate", 0.561, 0.482, "56.1%", "48.2%"),
    ("monthly_win_rate", "Monthly Win Rate", 0.672, 0.393, "67.2%", "39.3%"),
    ("profit_factor", "Profit Factor", 1.94, 0.64, "1.94", "0.64"),
    ("payoff_ratio", "Payoff Ratio", 1.05, 0.55, "1.05", "0.55"),
    ("volatility", "Volatility (ann.)", 0.1788, 0.2877, "17.88%", "28.77%"),
    ("skew", "Skewness", -0.36, 0.47, "-0.36", "0.47"),
    ("kurtosis", "Kurtosis", 4.27, 5.58, "4.27", "5.58"),
    ("value_at_risk", "VaR (95 %)", -0.0299, -0.0144, "-2.99%", "-1.44%"),
    ("conditional_value_at_risk", "CVaR (95 %)", -0.0388, -0.0166, "-3.88%", "-1.66%"),
]

# Metrics whose highest value is highlighted with the ``best-value`` class.
_BEST_HIGHLIGHTED: frozenset[str] = frozenset(
    {"sharpe", "calmar", "recovery_factor", "win_rate", "monthly_win_rate", "profit_factor", "payoff_ratio"}
)

_EXPECTED_CATEGORY_LABELS: tuple[str, ...] = (
    "Returns",
    "Risk-Adjusted Performance",
    "Drawdown",
    "Win / Loss",
    "Distribution & Risk",
)

_AUM: float = 1e6


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def stats_table() -> str:
    """Stats table rendered once from a synthetic two-asset summary frame."""
    summary = pl.DataFrame(
        {
            "metric": [row[0] for row in _EXPECTED_STATS_ROWS],
            "A": [row[2] for row in _EXPECTED_STATS_ROWS],
            "B": [row[3] for row in _EXPECTED_STATS_ROWS],
        }
    )
    return _stats_table_html(summary)


@pytest.fixture(scope="module")
def dated_portfolio() -> Portfolio:
    """Two-year, two-asset dated portfolio with non-zero turnover."""
    n = 504
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([100.0 * (1.0005**i) for i in range(n)], dtype=pl.Float64),
            "B": pl.Series([200.0 * (1.0002**i) for i in range(n)], dtype=pl.Float64),
        }
    )
    positions = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([1000.0 + float(i % 10) for i in range(n)], dtype=pl.Float64),
            "B": pl.Series([500.0 + float(i % 5) for i in range(n)], dtype=pl.Float64),
        }
    )
    return Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=_AUM)


@pytest.fixture(scope="module")
def dated_html(dated_portfolio: Portfolio) -> str:
    """Full report HTML for the dated portfolio, rendered once per module."""
    return dated_portfolio.report.to_html()


@pytest.fixture(scope="module")
def dateless_portfolio() -> Portfolio:
    """100-row, two-asset portfolio without a date column."""
    n = 100
    prices = pl.DataFrame(
        {
            "A": pl.Series([100.0 + 0.1 * i for i in range(n)], dtype=pl.Float64),
            "B": pl.Series([200.0 - 0.05 * i for i in range(n)], dtype=pl.Float64),
        }
    )
    positions = pl.DataFrame(
        {
            "A": pl.Series([1000.0] * n, dtype=pl.Float64),
            "B": pl.Series([500.0] * n, dtype=pl.Float64),
        }
    )
    return Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=_AUM)


@pytest.fixture(scope="module")
def dateless_html(dateless_portfolio: Portfolio) -> str:
    """Dateless report rendered once with a title that requires HTML escaping."""
    return dateless_portfolio.report.to_html(title="Q&A <Report>")


# ── Stats table: structure ────────────────────────────────────────────────────


def test_stats_table_exact_header_shell(stats_table: str):
    """Table opens with the exact shell, header cells and closing tags.

    Kills: ``XX``-wrapped shell strings in ``_table_html`` and
    ``_stats_table_html`` header generation, ``header_cells = None``, the
    ``"XXXX".join`` header mutant (breaks ``</th><th`` adjacency), and the
    ``c != "XXmetricXX"`` mutant (the ``metric`` column would become an
    asset header).
    """
    assert stats_table.startswith(
        '<table class="stats-table"><thead><tr>'
        '<th class="metric-header">Metric</th>'
        '<th class="asset-header">A</th><th class="asset-header">B</th>'
        "</tr></thead><tbody>"
    )
    assert stats_table.endswith("</tbody></table>")


def test_stats_table_contains_no_mutation_sentinel(stats_table: str):
    """The rendered stats table never contains the mutmut sentinel ``XX``.

    The un-mutated table consists only of fixed tag/class literals, metric
    labels, category names, the asset names ``A``/``B`` and formatted numbers,
    none of which can produce ``XX`` — so this single assertion kills every
    surviving ``XX``-string mutant on the ``_stats_table_html`` path
    (row/cell wrappers, suffix ``"" -> "XXXX"``, separators, section headers).
    """
    assert "XX" not in stats_table


def test_stats_table_first_category_block_exact(stats_table: str):
    r"""The tbody starts with the exact Returns section header and first row.

    Kills: section-header ``XX`` wrappers, ``colspan`` arithmetic mutants
    (``+ 2`` / ``- 1``), the ``rows_html = "XXXX".join`` mutant (breaks the
    ``</tr>\\n<tr`` adjacency) and the per-row ``cells = "XXXX".join`` mutant
    (breaks the ``</td><td`` adjacency between asset cells).
    """
    assert (
        '<tbody><tr class="table-section-header">'
        '<td colspan="3"><strong>Returns</strong></td></tr>\n'
        '<tr><td class="metric-name">Avg Return</td>'
        '<td class="metric-value">1.11%</td><td class="metric-value">0.12%</td></tr>\n'
    ) in stats_table


def test_stats_table_all_category_headers_present(stats_table: str):
    """Every category section header is rendered with the exact colspan.

    Kills the ``XX``-wrapped category-label mutants in ``_CATEGORIES``.
    """
    for label in _EXPECTED_CATEGORY_LABELS:
        assert f'<td colspan="3"><strong>{label}</strong></td>' in stats_table, label


# ── Stats table: per-metric rows ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("metric", "label", "rendered_a"),
    [(m, label, a_str) for m, label, _a, _b, a_str, _b_str in _EXPECTED_STATS_ROWS],
    ids=[row[0] for row in _EXPECTED_STATS_ROWS],
)
def test_stats_table_row_label_and_formatted_value(stats_table: str, metric: str, label: str, rendered_a: str):
    """Each metric row renders its exact label cell adjacent to its A-value cell.

    The label/value adjacency kills, per metric: key mutants in
    ``_METRIC_FORMATS`` (lookup falls back to ``.4f``), label-value mutants in
    ``_METRIC_LABELS`` (``XX`` wrapping), suffix mutants (``"" -> "XXXX"``),
    metric-name mutants inside ``_CATEGORIES`` (row disappears) and
    ``_HIGHER_IS_BETTER`` membership mutants (best-value class disappears).
    """
    best = "  best-value" if metric in _BEST_HIGHLIGHTED else ""
    assert f'<td class="metric-name">{label}</td><td class="metric-value{best}">{rendered_a}</td>' in stats_table


def test_stats_table_best_value_marks_only_the_best_asset(stats_table: str):
    """For a higher-is-better metric only the winning asset is highlighted.

    Kills: ``a != best_asset`` inversion, ``best_asset = max(...)[1]``
    (the float value can never equal an asset name), ``best_asset = None``,
    ``finite_pairs = None``, and ``metric not in _HIGHER_IS_BETTER`` inversion.
    """
    assert (
        '<td class="metric-name">Sharpe Ratio</td>'
        '<td class="metric-value  best-value">1.61</td>'
        '<td class="metric-value">0.91</td></tr>'
    ) in stats_table


def test_metric_dicts_and_categories_cover_exactly_the_expected_keys():
    """The format/label dicts and category lists use exactly the expected keys.

    Pins the key sets against a hard-coded list. This is the only way to kill
    the 11 output-equivalent ``_METRIC_LABELS`` key mutants whose title-cased
    fallback equals the explicit label (e.g. ``XXwin_rateXX`` still renders
    "Win Rate" via ``metric.replace("_", " ").title()``).
    """
    expected = frozenset(row[0] for row in _EXPECTED_STATS_ROWS)
    assert set(_METRIC_FORMATS) == expected
    assert set(_METRIC_LABELS) == expected
    assert {metric for _label, metrics in _CATEGORIES for metric in metrics} == expected
    assert [label for label, _metrics in _CATEGORIES] == list(_EXPECTED_CATEGORY_LABELS)


# ── Full report: header metadata, footer, CDN ─────────────────────────────────


def test_report_period_span_exact(dated_portfolio: Portfolio, dated_html: str):
    """The Period line renders the exact date range and period count.

    Kills: ``has_date`` mutants, dated-branch ``start_date``/``end_date`` and
    ``period_info`` ``None``/``XX`` mutants (the ``XX`` wrap breaks the
    ``</strong> `` adjacency asserted here).
    """
    dates = dated_portfolio.prices["date"]
    n = dated_portfolio.prices.height
    expected = f"<strong>Period:</strong> {dates.min()} → {dates.max()} &nbsp;|&nbsp; {n:,} periods</span>"
    assert expected in dated_html


def test_report_assets_and_aum_exact(dated_html: str):
    """Assets and AUM lines render exactly, with no sentinel padding.

    Kills: ``assets_list = "XX, XX".join``, ``assets_list = None`` and the
    ``aum=f"XX{...}XX"`` mutants (existing substring tests cannot, because
    ``XX1,000,000XX`` still contains ``1,000,000``).
    """
    assert "<strong>Assets:</strong> A, B</span>" in dated_html
    assert "<strong>AUM:</strong> 1,000,000</span>" in dated_html


def test_report_footer_ends_with_end_date(dated_portfolio: Portfolio, dated_html: str):
    """The footer renders 'Generated by ... | <end date>' exactly.

    Kills the ``footer_date = None`` mutant (would render ``None``).
    """
    end = dated_portfolio.prices["date"].max()
    assert f"Generated by <strong>jquantstats</strong>&nbsp;|&nbsp;{end}\n</footer>" in dated_html


def test_report_embeds_plotly_cdn_exactly_once(dated_html: str):
    """Plotly.js is pulled from the CDN exactly once.

    Kills: ``_first = False``/``None`` initialisation (zero CDN tags),
    the dropped ``_first = False`` reset (one CDN tag per figure) and
    ``include = None`` (zero CDN tags).
    """
    assert dated_html.count("cdn.plot.ly") == 1


def test_report_container_max_width(dated_html: str):
    """The container CSS uses the exact 1400px max-width.

    Kills the ``container_max_width="XX1400pxXX"`` mutant.
    """
    assert "max-width: 1400px;" in dated_html


# ── Full report: turnover table ───────────────────────────────────────────────


def test_report_turnover_table_exact(dated_html: str):
    """The turnover table renders shell, labels and 4-decimal values verbatim.

    A single anchored regex over the whole table kills every turnover-section
    mutant: ``XX``-wrapped shell strings (they break tag adjacency), the
    ``"XXXX".join`` row separator, ``turnover_rows``/``turnover_html = None``,
    ``replace("_", ...)`` mutants, ``row["XXmetricXX"]``/``row["XXvalueXX"]``
    (raise, turning the table into the unavailable notice) and
    ``iter_rows(named=False)`` (same).
    """
    value = r"-?\d+\.\d{4}"
    pattern = (
        re.escape(
            '<table class="stats-table"><thead><tr>'
            '<th class="metric-header">Metric</th><th class="asset-header">Value</th>'
            "</tr></thead><tbody>"
            '<tr><td class="metric-name">Mean Daily Turnover</td><td class="metric-value">'
        )
        + value
        + re.escape('</td></tr><tr><td class="metric-name">Mean Weekly Turnover</td><td class="metric-value">')
        + value
        + re.escape('</td></tr><tr><td class="metric-name">Turnover Std</td><td class="metric-value">')
        + value
        + re.escape("</td></tr></tbody></table>")
    )
    assert re.search(pattern, dated_html)


# ── Full report: dateless branch and autoescaping ─────────────────────────────


def test_report_dateless_period_and_footer_exact(dateless_html: str):
    """Without dates, the period line and footer render their exact fallbacks.

    Kills: ``f"XX{...} periodsXX"`` (existing tests only check the inner
    substring) and the else-branch ``footer_date ... else "XXXX"`` mutant.
    """
    assert "<strong>Period:</strong> 100 periods</span>" in dateless_html
    assert "Generated by <strong>jquantstats</strong>&nbsp;|&nbsp;\n</footer>" in dateless_html


def test_report_title_is_html_escaped(dateless_html: str):
    """A title containing markup is autoescaped in the rendered document.

    Kills the ``select_autoescape(["XXhtmlXX"])`` mutant: with the extension
    list broken, autoescaping is disabled for ``.html`` templates and the raw
    ``<Report>`` would be emitted.
    """
    assert "Q&amp;A &lt;Report&gt;</title>" in dateless_html
    assert "<Report>" not in dateless_html


# ── Full report: error notices, file output, slots ────────────────────────────


def test_report_failure_notices_exact(dateless_portfolio: Portfolio):
    """Chart and turnover failure notices render verbatim, unpadded.

    Kills the ``XX``-wrapped unavailable-notice mutants (existing substring
    assertions survive the wrap) and the except-branch ``turnover_html``
    mutants.
    """
    with (
        patch.object(PortfolioPlots, "snapshot", side_effect=RuntimeError("boom")),
        patch.object(Portfolio, "turnover_summary", side_effect=RuntimeError("nope")),
    ):
        html = dateless_portfolio.report.to_html()
    assert '<p class="chart-unavailable">Chart unavailable: boom</p>' in html
    assert '<p class="chart-unavailable">Turnover data unavailable: nope</p>' in html


def test_to_html_creates_nested_parent_directories(tmp_path: Path, dateless_portfolio: Portfolio):
    """to_html(path=...) creates missing intermediate directories.

    Kills the ``mkdir(parents=False, ...)`` mutant, which raises
    ``FileNotFoundError`` for a nested target path.
    """
    target = tmp_path / "deep" / "nested" / "dirs" / "report.html"
    result = dateless_portfolio.report.to_html(path=target)
    assert result == target
    assert result.exists()


def test_report_rejects_unknown_attributes():
    """Report uses __slots__ and rejects arbitrary attribute assignment.

    Kills the ``__slots__ = None`` mutant (instances would grow a __dict__).
    """
    report = Report(object())
    with pytest.raises(AttributeError):
        report.unexpected_attribute = 1  # type: ignore[attr-defined]


def test_templates_dir_resolves_to_real_templates():
    """_TEMPLATES_DIR points at the packaged templates directory.

    Kills the ``Path(...) * "templates"`` and ``_TEMPLATES_DIR = None``
    mutants explicitly (both also break module import, but this makes the
    kill independent of import-time behaviour).
    """
    assert isinstance(_TEMPLATES_DIR, Path)
    assert _TEMPLATES_DIR.name == "templates"
    assert (_TEMPLATES_DIR / "portfolio_report.html").is_file()
    assert (_TEMPLATES_DIR / "_base.html").is_file()


# ── _formatting helpers ───────────────────────────────────────────────────────


def test_plotly_div_default_omits_plotlyjs_bundle():
    """_plotly_div's default excludes the multi-megabyte Plotly.js bundle.

    Kills the ``include_plotlyjs: ... = True`` default mutant: inlining the
    bundle makes the output several MB, far above this ceiling.
    """
    div = _plotly_div(go.Figure())
    assert len(div) < 200_000
    assert "<html" not in div


def test_plotly_div_returns_div_fragment_not_document():
    """_plotly_div returns an HTML fragment, never a full document.

    Kills the ``full_html=True`` mutant (output would start with ``<html>``).
    """
    div = _plotly_div(go.Figure(), include_plotlyjs=False)
    assert div.strip().startswith("<div")
    assert "<html" not in div
    assert "<!DOCTYPE" not in div


def test_table_html_exact_shell():
    """_table_html wraps header cells and body rows in the exact table shell.

    Exact equality kills all five surviving ``XX`` shell mutants in
    ``_formatting._table_html``.
    """
    out = _table_html('<th class="asset-header">Z</th>', "<tr><td>r</td></tr>")
    assert out == (
        '<table class="stats-table"><thead><tr>'
        '<th class="metric-header">Metric</th><th class="asset-header">Z</th>'
        "</tr></thead><tbody><tr><td>r</td></tr></tbody></table>"
    )
