"""HTML rendering for the self-contained performance report.

These helpers turn the metrics table, drawdown summary, and Plotly
charts assembled by ``Reports.full`` into a single dark-themed HTML
document.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from ._formatting import _fmt, _plotly_div, _table_html

# ── Metrics-table section layout ───────────────────────────────────────────────

_SECTION_SPANS: list[tuple[str, list[str]]] = [
    (
        "Overview",
        [
            "Start Period",
            "End Period",
            "Time in Market",
            "Cumulative Return",
            "CAGR",
        ],
    ),
    (
        "Risk-Adjusted Ratios",
        [
            "Sharpe",
            "Prob. Sharpe Ratio",
            "Sortino",
            "Sortino / √2",
            "Omega",
        ],
    ),
    (
        "Drawdown",
        [
            "Max Drawdown",
            "Max DD Duration",
            "Avg Drawdown",
            "Recovery Factor",
            "Ulcer Index",
            "Serenity Index",
        ],
    ),
    (
        "Trading",
        [
            "Gain/Pain Ratio",
            "Gain/Pain (1M)",
            "Payoff Ratio",
            "Profit Factor",
            "Common Sense Ratio",
            "CPC Index",
            "Tail Ratio",
            "Outlier Win Ratio",
            "Outlier Loss Ratio",
        ],
    ),
    (
        "Recent Returns",
        [
            "MTD",
            "3M",
            "6M",
            "YTD",
            "1Y",
            "3Y (ann.)",
            "5Y (ann.)",
            "All-time (ann.)",
        ],
    ),
    (
        "Smart Ratios",
        ["Smart Sharpe", "Smart Sortino", "Smart Sortino / √2"],
    ),
    (
        "Risk",
        [
            "Volatility (ann.)",
            "Calmar",
            "Risk-Adjusted Return",
            "Risk-Return Ratio",
            "Ulcer Performance Index",
            "Skew",
            "Kurtosis",
        ],
    ),
    (
        "Averages",
        [
            "Avg. Return",
            "Avg. Win",
            "Avg. Loss",
            "Win/Loss Ratio",
            "Profit Ratio",
            "Win Rate",
            "Monthly Win Rate",
        ],
    ),
    (
        "Expected Returns",
        ["Expected Daily", "Expected Monthly", "Expected Yearly"],
    ),
    (
        "Tail Risk",
        [
            "Kelly Criterion",
            "Risk of Ruin",
            "Daily VaR",
            "Expected Shortfall (cVaR)",
        ],
    ),
    (
        "Streaks",
        ["Max Consecutive Wins", "Max Consecutive Losses"],
    ),
    (
        "Best / Worst",
        ["Best Day", "Worst Day"],
    ),
    (
        "Benchmark",
        ["Beta", "Alpha", "Correlation", "R²", "Treynor Ratio"],
    ),
]

_PCT_METRICS: frozenset[str] = frozenset(
    {
        "Time in Market",
        "Cumulative Return",
        "CAGR",
        "Prob. Sharpe Ratio",
        "Max Drawdown",
        "Avg Drawdown",
        "MTD",
        "3M",
        "6M",
        "YTD",
        "1Y",
        "3Y (ann.)",
        "5Y (ann.)",
        "All-time (ann.)",
        "Volatility (ann.)",
        "Risk-Adjusted Return",
        "Avg. Return",
        "Avg. Win",
        "Avg. Loss",
        "Win Rate",
        "Monthly Win Rate",
        "Expected Daily",
        "Expected Monthly",
        "Expected Yearly",
        "Kelly Criterion",
        "Risk of Ruin",
        "Daily VaR",
        "Expected Shortfall (cVaR)",
        "Best Day",
        "Worst Day",
        "Alpha",
        "Correlation",
    }
)


def _metrics_table_html(df: pl.DataFrame) -> str:
    """Render a metrics DataFrame as a styled HTML table with section headers.

    Args:
        df: DataFrame with a ``"Metric"`` column and one column per asset.

    Returns:
        An HTML ``<table>`` string.

    """
    assets = [c for c in df.columns if c != "Metric"]
    rows_by_label: dict[str, dict[str, Any]] = {
        str(row["Metric"]): {a: row.get(a) for a in assets} for row in df.iter_rows(named=True)
    }

    n_cols = len(assets) + 1
    header_cells = "".join(f'<th class="asset-header">{a}</th>' for a in assets)
    parts: list[str] = []

    rendered: set[str] = set()
    for section_label, section_metrics in _SECTION_SPANS:
        section_rows: list[str] = []
        for label in section_metrics:
            if label not in rows_by_label:
                continue
            vals = rows_by_label[label]
            rendered.add(label)
            suffix = "%" if label in _PCT_METRICS else ""
            cells = "".join(f'<td class="metric-value">{_fmt(vals.get(a), ".2f", suffix)}</td>' for a in assets)
            section_rows.append(f'<tr><td class="metric-name">{label}</td>{cells}</tr>\n')

        if section_rows:
            parts.append(
                f'<tr class="table-section-header"><td colspan="{n_cols}"><strong>{section_label}</strong></td></tr>\n'
            )
            parts.extend(section_rows)

    # Anything not matched by a section (e.g. string-valued rows like dates)
    for label, vals in rows_by_label.items():
        if label in rendered:
            continue
        raw = next(iter(vals.values()), None)
        if isinstance(raw, str):
            cells = "".join(f'<td class="metric-value">{vals.get(a, "")}</td>' for a in assets)
        else:
            cells = "".join(f'<td class="metric-value">{_fmt(vals.get(a), ".4f")}</td>' for a in assets)
        parts.append(f'<tr><td class="metric-name">{label}</td>{cells}</tr>\n')

    return _table_html(header_cells, "".join(parts))


def _drawdowns_section_html(data: Any, assets: list[str]) -> str:
    """Render worst-5 drawdown periods per asset as HTML tables.

    Args:
        data: The DataLike object (accessed via ``getattr`` for stats).
        assets: List of asset column names to render.

    Returns:
        HTML string containing one table per asset.

    """
    stats = getattr(data, "stats", None)
    if stats is None:
        return "<p>No drawdown data available.</p>"

    parts: list[str] = []
    try:
        dd_dict: dict[str, pl.DataFrame] = stats.drawdown_details()
    except Exception:
        return "<p>Drawdown details unavailable.</p>"

    for asset in assets:
        df = dd_dict.get(asset)
        if df is None or len(df) == 0:
            parts.append(f"<h3>{asset}</h3><p>No drawdown periods found.</p>")
            continue

        worst5 = df.sort("max_drawdown").head(5)
        rows = "".join(
            f"<tr>"
            f"<td>{row.get('start', '')}</td>"
            f"<td>{row.get('valley', '')}</td>"
            f"<td>{row.get('end', '') or '—'}</td>"
            f"<td>{_fmt(row.get('max_drawdown'), '.2%')}</td>"
            f"<td>{row.get('duration', '') or '—'}</td>"
            f"</tr>"
            for row in worst5.iter_rows(named=True)
        )
        parts.append(
            f"<h3>{asset}</h3>"
            '<table class="stats-table">'
            "<thead><tr>"
            "<th>Start</th><th>Valley</th><th>End</th><th>Max DD</th><th>Duration</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    return "\n".join(parts)


def _try_plotly_div(fig: Any, include_cdn: bool = False) -> str:
    """Convert a Plotly figure to an HTML div string.

    Args:
        fig: A Plotly Figure object (or anything with ``to_html``).
        include_cdn: Include the Plotly JS CDN ``<script>`` tag. Defaults to False.

    Returns:
        An HTML string, or an empty string if conversion fails.

    """
    try:
        return _plotly_div(fig, include_plotlyjs="cdn" if include_cdn else False)
    except Exception:
        return ""


_REPORT_CSS = """
body{margin:0;font-family:system-ui,sans-serif;background:#0f1117;color:#e2e8f0}
h1{color:#90cdf4;margin:0 0 4px}
h2{color:#63b3ed;border-bottom:1px solid #2d3748;padding-bottom:6px}
h3{color:#a0aec0;margin:16px 0 6px}
header{padding:24px 32px;background:linear-gradient(135deg,#1a202c,#2d3748);border-bottom:1px solid #4a5568}
.period-info{color:#a0aec0;font-size:.85rem;margin-top:4px}
main{padding:24px 32px}
section{margin-bottom:40px}
.stats-table{border-collapse:collapse;width:100%;font-size:.85rem}
.stats-table th,.stats-table td{padding:6px 12px;text-align:right;border-bottom:1px solid #2d3748}
.stats-table th:first-child,.stats-table td:first-child{text-align:left}
.metric-header,.asset-header{background:#1a202c;color:#90cdf4;font-weight:600}
.metric-name{color:#cbd5e0}
.metric-value{font-family:monospace;color:#e2e8f0}
.table-section-header td{background:#1a202c;color:#68d391;font-size:.75rem;text-transform:uppercase;
letter-spacing:.08em;padding:8px 12px}
footer{padding:16px 32px;color:#718096;font-size:.75rem;border-top:1px solid #2d3748}
"""


def _build_full_html(
    title: str,
    period_info: str,
    assets_str: str,
    metrics_html: str,
    drawdowns_html: str,
    charts_html: str,
) -> str:
    """Assemble the full HTML report from its component parts.

    Args:
        title: Page and ``<h1>`` title.
        period_info: Period metadata string for the header.
        assets_str: Comma-separated asset names for the header.
        metrics_html: Pre-rendered metrics ``<table>`` HTML.
        drawdowns_html: Pre-rendered worst-drawdowns HTML.
        charts_html: Pre-rendered Plotly chart divs.

    Returns:
        A complete, self-contained HTML document string.

    """
    from datetime import date

    footer_date = str(date.today())
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{_REPORT_CSS}</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="period-info">{period_info}</div>
  <div class="period-info">Assets: {assets_str}</div>
</header>
<main>
  <section id="metrics">
    <h2>Performance Metrics</h2>
    {metrics_html}
  </section>
  <section id="drawdowns">
    <h2>Worst 5 Drawdown Periods</h2>
    {drawdowns_html}
  </section>
  <section id="charts">
    <h2>Charts</h2>
    {charts_html}
  </section>
</main>
<footer>Generated by jquantstats · {footer_date}</footer>
</body>
</html>"""


# ── Reports dataclass ─────────────────────────────────────────────────────────
