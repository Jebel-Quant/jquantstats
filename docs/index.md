---
icon: material/chart-line
hide:
  - toc
---

# jquantstats

**Portfolio analytics for quants** — built on Polars, powered by Plotly.

[![PyPI version](https://badge.fury.io/py/jquantstats.svg)](https://badge.fury.io/py/jquantstats)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/jquantstats/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/jebel-quant/jquantstats/blob/main/LICENSE)
[![Coverage](https://jebel-quant.github.io/jquantstats/coverage-badge.svg)](https://jebel-quant.github.io/jquantstats/reports/coverage/index.html)
[![Downloads](https://static.pepy.tech/personalized-badge/jquantstats?period=month&units=international_system&left_color=black&right_color=orange&left_text=downloads%2Fmonth)](https://pepy.tech/project/jquantstats)

---

**Quick Links:** [📚 Repository](https://github.com/jebel-quant/jquantstats) • [📦 PyPI](https://pypi.org/project/jquantstats/) • [🐛 Issues](https://github.com/jebel-quant/jquantstats/issues) • [💬 Discussions](https://github.com/jebel-quant/jquantstats/discussions)

---

## 📋 Overview

jQuantStats is a Python library for portfolio analytics that helps quants and portfolio managers
understand their strategy performance in depth. It provides two complementary entry points:
a **Portfolio route** that works directly from price and position data, and a **Data route**
for arbitrary return streams. All analytics, visualisations, and HTML reports are available
from either entry point.

## 🚀 Installation

```bash
pip install jquantstats
```

Python **3.11+** is required. Add optional extras as needed:

```bash
pip install jquantstats[plot]   # static chart export (kaleido)
pip install jquantstats[web]    # FastAPI web server
```

## 💻 Quick Start

=== "Portfolio route"

    Use `Portfolio` when you have **prices** *and* **positions**.
    This is the recommended entry point — it exposes the full analytics suite.

    ```python
    import polars as pl
    from jquantstats import Portfolio

    pf = Portfolio.from_cash_position(
        prices=prices_df,      # pl.DataFrame: date + asset columns
        cash_position=pos_df,  # pl.DataFrame: date + asset columns (£ amounts)
        aum=1_000_000,
    )

    pf.stats.sharpe()          # {"AAPL": 1.34, "MSFT": 0.91, "portfolio": 1.21}
    pf.plots.snapshot()        # interactive NAV + drawdown dashboard
    pf.report.to_html()        # full HTML tearsheet
    ```

=== "Data route"

    Use `Data` when you only have a **return series** (or prices without positions).

    ```python
    import polars as pl
    from jquantstats import Data

    data = Data.from_returns(
        returns=returns_df,     # pl.DataFrame: date + return columns
        benchmark="Benchmark",  # optional column name
        rf=0.0,
    )

    data.stats.sharpe()
    data.stats.sortino()
    data.stats.annual_breakdown()
    data.plots.monthly_heatmap()
    data.reports.to_html()
    ```

## 🔬 Portfolio Route — Why It Matters

Working from positions rather than returns unlocks analysis that is impossible from a return series alone:

- **`pf.lag(n)`** — execution-delay analysis: simulate T+1, T+2 fills and see the Sharpe impact
- **`pf.tilt` / `pf.timing`** — decompose performance into allocation skill vs. timing skill
- **`pf.turnover` / `pf.turnover_summary()`** — daily and weekly turnover analytics
- **`pf.trading_cost_impact(max_bps=20)`** — sweep transaction-cost sensitivity in one call

```python
pf_t0 = pf           # ideal T+0 execution
pf_t1 = pf.lag(1)    # signal fires today, fills tomorrow

print(pf_t0.stats.sharpe())  # {"portfolio": 1.34}
print(pf_t1.stats.sharpe())  # {"portfolio": 1.28}

fig = pf.plots.lead_lag_ir_plot(start=-5, end=10)
```

See [Getting Started](getting_started.md) for a complete walkthrough.

## 📊 jQuantStats vs QuantStats

| Feature | QuantStats | jquantstats |
|---|---|---|
| DataFrame engine | pandas | Polars |
| Charts | matplotlib (static) | Plotly (interactive) |
| Multi-asset in one call | no | yes |
| Portfolio route | no | yes |
| Execution-delay analysis | no | `pf.lag(n)` |
| Tilt / timing decomp | no | yes |
| Turnover analytics | no | yes |
| Cost modelling | no | yes |
| Type annotations | partial | full |

See the [Migration Guide](MIGRATION.md) for a complete API mapping.

## 📄 Next Steps

<div class="grid cards" markdown>

-   :rocket: [**Getting Started**](getting_started.md)

    Step-by-step guide for both entry points.

-   :money_with_wings: [**Cost Models**](cost_models.md)

    Model transaction costs and sweep sensitivity.

-   :arrows_counterclockwise: [**Migration from QuantStats**](MIGRATION.md)

    Complete API mapping and behavioural differences.

-   :books: [**API Reference**](api.md)

    Full method signatures and docstrings.

-   :material-history: [**Changelog**](changelog.md)

    Release history and notable API changes.

</div>
