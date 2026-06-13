---
icon: material/rocket-launch
---

# Getting Started

## First five minutes

A complete, runnable example — returns in, Sharpe out:

```python
import polars as pl
from jquantstats import Data

returns = pl.DataFrame(
    {
        "Date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "Fund": [0.011, -0.004, 0.007, 0.002],
    }
).with_columns(pl.col("Date").str.to_date())

data = Data.from_returns(returns)
print(data.stats.sharpe())        # {'Fund': ...} — annualised
data.plots.snapshot().show()      # interactive Plotly chart
```

Swap the inline frame for your own returns (pandas works too — any
narwhals-compatible frame is accepted) and you have a full tear sheet via
`data.reports.full()`. The rest of this page covers the two entry points in
depth; runnable notebooks live under [Examples](examples.md).

---

jQuantStats provides two entry points depending on what data you have:

| You have… | Use… | Why |
|-----------|------|-----|
| Prices **and** positions | [`Portfolio`](#route-a-portfolio) | Unlocks execution-delay analysis, turnover analytics, and cost modelling |
| Returns (or prices only) | [`Data`](#route-b-data) | Lighter-weight path; easiest migration from QuantStats |

```mermaid
flowchart TD
    Q1{What data do you have?}
    Q1 -- "prices + positions" --> PF["Portfolio.from_cash_position(...)"]
    Q1 -- "returns only" --> DR["Data.from_returns(...)"]
    Q1 -- "prices only" --> DP["Data.from_prices(...)"]
    PF --> PA["stats · plots · report<br/>+ turnover · costs · lag() · attribution"]
    DR --> DA["stats · plots · report"]
    DP --> DA
```

Both routes expose the same stats, plots, and report API.

---

## Installation

```bash
pip install jquantstats
```

!!! info "Python version"
    Python **3.11+** is required.

Optional extras:

```bash
pip install jquantstats[plot]   # static chart export via kaleido
pip install jquantstats[web]    # FastAPI web server
```

---

## Route A: Portfolio

Use `Portfolio` when you have **prices** and **positions**. This unlocks
position-level analytics — turnover, cost modelling, execution-delay analysis —
that are impossible from returns alone.

Internally, `Portfolio` derives everything from the three raw inputs:

```mermaid
flowchart LR
    P[prices] --> PR["profits<br/>(per-asset daily P&amp;L)"]
    CP[positions] --> PR
    PR --> PT["profit<br/>(aggregate daily P&amp;L)"]
    PT --> NAV["nav_accumulated /<br/>nav_compounded"]
    AUM[aum] --> NAV
    PT --> RET["returns<br/>(profit / AUM)"]
    AUM --> RET
    RET --> OUT["stats · plots · report"]
    CP --> TO[turnover] --> COST["cost_adjusted_returns ·<br/>trading_cost_impact"]
```

### Build a Portfolio

=== "Cash positions"

    ```python
    from datetime import date
    import polars as pl
    from jquantstats import Portfolio

    prices = pl.DataFrame({
        "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
        "AAPL": [75.09, 74.36, 75.80],
        "MSFT": [160.62, 158.96, 159.03],
    })

    # Dollar amount held per asset each day
    positions = pl.DataFrame({
        "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
        "AAPL": [500_000.0, 500_000.0, 600_000.0],
        "MSFT": [300_000.0, 300_000.0, 300_000.0],
    })

    pf = Portfolio.from_cash_position(
        prices=prices,
        cash_position=positions,
        aum=1_000_000,
    )
    ```

=== "Share counts"

    ```python
    # If you track shares rather than dollar amounts
    pf = Portfolio.from_position(
        prices=prices,
        position=shares_df,   # units held per asset
        aum=1_000_000,
    )
    ```

=== "Risk-scaled positions"

    ```python
    # Positions expressed as a fraction of volatility budget
    pf = Portfolio.from_risk_position(
        prices=prices,
        risk_position=risk_df,
        aum=1_000_000,
        vola=vola_df,
    )
    ```

### Core series

```python
pf.returns          # daily portfolio returns  →  pl.Series
pf.nav_compounded   # compounded NAV curve     →  pl.Series
pf.drawdown         # drawdown from HWM        →  pl.Series
```

### Stats

```python
pf.stats.sharpe()        # (1)
pf.stats.max_drawdown()
pf.stats.volatility()
pf.stats.summary()       # full metrics table  →  pl.DataFrame
```

1. Returns a `dict` keyed by column name, e.g.
   `{'AAPL': 1.34, 'MSFT': 0.91, 'portfolio': 1.21}`

### Plots

```python
fig = pf.plots.snapshot()              # NAV + drawdown dashboard
fig = pf.plots.rolling_sharpe(window=60)
fig.show()                             # opens in browser / notebook
```

### Report

```python
html = pf.report.to_html()

with open("report.html", "w") as f:
    f.write(html)
```

---

## Route B: Data

Use `Data` when you already have a **return series** (or just prices without
positions). This is the lighter-weight path and accepts pandas DataFrames too.

=== "From returns"

    ```python
    from datetime import date
    import polars as pl
    from jquantstats import Data

    returns = pl.DataFrame({
        "Date":      [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
        "Strategy":  [0.012, -0.009,  0.005],
        "Benchmark": [0.004, -0.003,  0.002],
    })

    data = Data.from_returns(
        returns=returns,
        benchmark="Benchmark",  # column name to use as benchmark
        rf=0.0,                 # risk-free rate (float or time-varying DataFrame)
    )
    ```

=== "From prices"

    ```python
    data = Data.from_prices(
        prices=prices_df,   # pl.DataFrame: date + asset columns
    )
    ```

=== "From pandas"

    ```python
    import pandas as pd
    import polars as pl

    # Convert pd.Series with DatetimeIndex to pl.DataFrame
    returns_pl = pl.from_pandas(
        returns_pd.rename("Strategy").reset_index()
    )
    data = Data.from_returns(returns=returns_pl)
    ```

### Stats

```python
data.stats.sharpe()
data.stats.sortino()
data.stats.cagr()
data.stats.annual_breakdown()   # pl.DataFrame: year | return | sharpe | ...
```

### Plots

```python
fig = data.plots.snapshot()
fig = data.plots.monthly_heatmap()
fig = data.plots.returns_distribution()
fig.show()
```

### Report

```python
html = data.reports.to_html()
```

---

## Execution-Delay Analysis

!!! tip "Portfolio route only"
    Execution-delay analysis requires the `Portfolio` entry point.
    A return series does not carry enough information to reconstruct
    what happens under different execution assumptions.

Simulate what happens if signals are executed one or more days late:

```python
pf_t0 = pf           # ideal T+0
pf_t1 = pf.lag(1)    # T+1 fill — signal today, fills tomorrow
pf_t2 = pf.lag(2)    # T+2 fill

print(pf_t0.stats.sharpe())   # {"portfolio": 1.34}
print(pf_t1.stats.sharpe())   # {"portfolio": 1.28}
print(pf_t2.stats.sharpe())   # {"portfolio": 1.19}

# Visualise the full lead/lag sweep as a single chart
fig = pf.plots.lead_lag_ir_plot(start=-5, end=10)
fig.show()
```

`pf.lag(n)` returns a new `Portfolio` with positions shifted by `n` periods.
All downstream accessors — `.stats`, `.plots`, `.report` — recompute on the
shifted positions, so a single call gives you the full analytics picture
under a different execution assumption.

---

## Transaction Costs

!!! tip "Portfolio route only"
    Cost modelling requires the `Portfolio` entry point.

See [Cost Models](cost_models.md) for the full reference. Quick example:

```python
from jquantstats import CostModel, Portfolio

pf = Portfolio.from_cash_position(
    prices=prices,
    cash_position=positions,
    aum=1_000_000,
    cost_model=CostModel.turnover_bps(5.0),  # 5 bps one-way cost on AUM turnover
)

# Sweep Sharpe ratio across 0 → 20 bps — how robust is the strategy?
impact = pf.trading_cost_impact(max_bps=20)
print(impact)
```

---

## NaN / null handling

!!! warning "Polars vs pandas null semantics"
    Unlike pandas, Polars propagates `null` by default — if any value in a column
    is `null`, most statistics return `null` instead of a numeric result.

Use the `null_strategy` parameter to control this behaviour:

```python
# Mirrors pandas / QuantStats — silently drop rows with nulls
data = Data.from_returns(returns=returns_pl, null_strategy="drop")

# Forward-fill nulls before computing statistics
data = Data.from_returns(returns=returns_pl, null_strategy="forward_fill")

# Raise an error if any null is found (useful during development)
data = Data.from_returns(returns=returns_pl, null_strategy="raise")
```

The default (`null_strategy=None`) passes nulls through unchanged.

---

## Next steps

- [Cost Models](cost_models.md) — model transaction costs
- [Migration from QuantStats](MIGRATION.md) — complete API mapping
- [API Reference](api.md) — full method signatures
