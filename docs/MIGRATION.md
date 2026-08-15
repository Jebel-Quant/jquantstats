---
icon: material/swap-horizontal
---

# Migrating from QuantStats to jquantstats

**A heartfelt thank you to [Ran Aroussi](https://github.com/ranaroussi) for creating
[QuantStats](https://github.com/ranaroussi/quantstats) — a brilliant and widely loved
library that has helped countless quants and portfolio managers understand their
strategies better. This project would simply not exist without that foundation.
We have enormous respect for the original work and encourage everyone to check it out.**

`jquantstats` is a modern variation on the theme set by QuantStats — not a copy, not a
drop-in replacement, but a different take on the same problem space. Where QuantStats
offers a collection of standalone functions each operating on a return series,
jquantstats is built around a **portfolio-centric entry point**: you start from a
`Portfolio` object constructed from prices and positions, and analytics flow naturally
from there. This makes multi-asset analysis, cost modelling, and execution-lag studies
first-class citizens rather than afterthoughts.

This guide explains the **key conceptual and API differences** between the two libraries
so you know what to expect, and provides **concrete code translations** to help you move
your existing workflows over.


---

## Why migrate

| | QuantStats | jquantstats |
|---|---|---|
| DataFrame engine | pandas | Polars (zero pandas at runtime) |
| Charts | static matplotlib/seaborn | interactive Plotly |
| Type annotations | partial | full (PEP 484, `py.typed`) |
| Multi-asset | single `pd.Series` per call | multi-column DataFrame, one call |
| API style | functions (`qs.stats.sharpe(r)`) | object methods (`data.stats.sharpe()`) |
| Portfolio analytics | not supported | prices + positions route via `Portfolio` |
| Execution-delay analysis | not supported | `pf.lag(n)` + lead-lag IR plot |
| Tilt / timing decomposition | not supported | `pf.tilt`, `pf.timing`, `pf.tilt_timing_decomp` |
| Turnover analytics | not supported | `pf.turnover`, `pf.turnover_weekly()` |
| Cost modelling | not supported | `CostModel.per_unit()` / `CostModel.turnover_bps()` |
| Python version | 3.7+ | 3.11+ |

---

## Installation

```bash
pip install jquantstats
pip install jquantstats[plot]   # adds kaleido for static image export
```

Uninstall QuantStats once you're done:

```bash
pip uninstall quantstats
```

---

## Data format

QuantStats expects a **`pd.Series`** with a `DatetimeIndex`.
jquantstats expects a **`pl.DataFrame`** with a dedicated date column.

```python
import pandas as pd
import polars as pl

# QuantStats — pd.Series with DatetimeIndex
returns_pd: pd.Series = ...

# jquantstats — convert from pandas
returns_pl = pl.from_pandas(returns_pd.rename("MyStrategy").reset_index())
# → DataFrame columns: ["Date", "MyStrategy"]

# jquantstats — build from scratch
from datetime import date
returns_pl = pl.DataFrame({
    "Date":       [date(2020, 1, 2), date(2020, 1, 3), ...],
    "MyStrategy": [0.01, -0.005, ...],
})
```

The date column is found by type, not by name: whatever you call it — `Date`,
`timestamp`, `dt` — the first temporal column is used, and the resulting object
exposes it as `date` (`data.index["date"]`, `data.date_col == ["date"]`). Pass
`date_col=` only to nominate a specific column, which is required for a
non-temporal (e.g. integer) index since there is no temporal column to find.

### Benchmarks

```python
# QuantStats — benchmark passed to every function call
import quantstats as qs
qs.stats.information_ratio(returns_pd, benchmark=benchmark_pd)

# jquantstats — benchmark provided once at construction time
import jquantstats as jqs
data = jqs.Data.from_returns(
    returns=returns_pl,
    benchmark=benchmark_pl,
)
data.stats.information_ratio()   # benchmark used automatically
```

---

## Entry points

jquantstats has two entry points depending on what data you have.

### `Data` — returns series (drop-in QuantStats replacement)

```python
import jquantstats as jqs

data = jqs.Data.from_returns(
    returns=returns_pl,     # pl.DataFrame with date + return columns
    benchmark=benchmark_pl, # optional
)

data.stats.sharpe()
data.plots.snapshot()
data.reports.metrics()
```

You can also construct from prices:

```python
data = jqs.Data.from_prices(prices=prices_pl)
```

### `Portfolio` — prices + positions (no QuantStats equivalent)

If you have asset prices and position sizes, use `Portfolio` to get
execution-delay analysis, tilt/timing decomposition, turnover analytics,
and cost modelling — none of which are available in QuantStats.

```python
pf = jqs.Portfolio.from_cash_position(
    prices=prices_df,        # pl.DataFrame: date + asset columns
    cash_position=pos_df,    # pl.DataFrame: date + asset columns (£ amounts)
    aum=1_000_000,
)

pf.stats.sharpe()            # the full stats API, same as Data
pf.plots.snapshot()          # a portfolio-specific plot set — see note below
pf.report.to_html()          # note: .report (singular), unlike Data.reports

# Drop down to the returns-series API at any time
pf.data.stats.calmar()
pf.data.plots.drawdown()
```

`Portfolio.stats` is the same object you get from `Data`, but `Portfolio.plots`
is **not** a superset of `Data.plots` — it is a separate, smaller set aimed at
position-level diagnostics:

```text
snapshot                  correlation_heatmap        monthly_returns_heatmap
rolling_sharpe_plot       rolling_volatility_plot    annual_sharpe_plot
lead_lag_ir_plot          lagged_performance_plot
smoothed_holdings_performance_plot                   trading_cost_impact_plot
```

For any other chart — `drawdown()`, `histogram()`, `monthly_heatmap()`, … — go
through `pf.data.plots`, which is always available.

Other factory methods:

```python
# From share counts
pf = jqs.Portfolio.from_position(prices, position, aum)

# From risk-scaled positions, where vola is an EWMA lookback in periods
# (an int, or a dict[str, int] per asset) — not a volatility frame
pf = jqs.Portfolio.from_risk_position(prices, risk_position, aum, vola=32)
```

---

## API mapping

All jquantstats analytics live on the `.stats`, `.plots`, and reports accessors
of a `Data` (or `Portfolio`) instance — note that the reports accessor is
`Data.reports` but `Portfolio.report`.  Methods return a **`dict` keyed by
column name** rather than a scalar.

```python
# one-time setup
data = jqs.Data.from_returns(returns=returns_pl, benchmark=benchmark_pl)
```

### Statistics

| QuantStats | jquantstats | Notes |
|---|---|---|
| `qs.stats.sharpe(r, periods=252)` | `data.stats.sharpe(periods=252)` | |
| `qs.stats.smart_sharpe(r, periods=252)` | `data.stats.smart_sharpe(periods=252)` | |
| `qs.stats.sortino(r, periods=252)` | `data.stats.sortino(periods=252)` | |
| `qs.stats.smart_sortino(r)` | `data.stats.smart_sortino()` | |
| `qs.stats.adjusted_sortino(r)` | `data.stats.adjusted_sortino()` | |
| `qs.stats.calmar(r)` | `data.stats.calmar()` | |
| `qs.stats.omega(r)` | `data.stats.omega()` | |
| `qs.stats.treynor_ratio(r, benchmark=b)` | `data.stats.treynor_ratio()` | |
| `qs.stats.volatility(r, periods=252)` | `data.stats.volatility(periods=252)` | |
| `qs.stats.implied_volatility(r)` | `data.stats.implied_volatility()` | |
| `qs.stats.skew(r)` | `data.stats.skew()` | |
| `qs.stats.kurtosis(r)` | `data.stats.kurtosis()` | |
| `qs.stats.max_drawdown(r)` | `data.stats.max_drawdown()` | |
| `qs.stats.value_at_risk(r)` | `data.stats.value_at_risk(alpha=0.05)` | |
| `qs.stats.conditional_value_at_risk(r, confidence=0.95)` | `data.stats.conditional_value_at_risk(confidence=0.95)` | `alpha=0.05` is also accepted; see note below |
| `qs.stats.win_rate(r)` | `data.stats.win_rate()` | |
| `qs.stats.avg_return(r)` | `data.stats.avg_return()` | |
| `qs.stats.avg_win(r)` | `data.stats.avg_win()` | |
| `qs.stats.avg_loss(r)` | `data.stats.avg_loss()` | |
| `qs.stats.best(r)` | `data.stats.best()` | |
| `qs.stats.worst(r)` | `data.stats.worst()` | |
| `qs.stats.consecutive_wins(r)` | `data.stats.consecutive_wins()` | |
| `qs.stats.consecutive_losses(r)` | `data.stats.consecutive_losses()` | |
| `qs.stats.profit_factor(r)` | `data.stats.profit_factor()` | |
| `qs.stats.profit_ratio(r)` | `data.stats.profit_ratio()` | |
| `qs.stats.payoff_ratio(r)` | `data.stats.payoff_ratio()` | |
| `qs.stats.win_loss_ratio(r)` | `data.stats.payoff_ratio()` | `win_loss_ratio()` is deprecated in jquantstats; use `payoff_ratio()` |
| `qs.stats.gain_to_pain_ratio(r)` | `data.stats.gain_to_pain_ratio()` | |
| `qs.stats.risk_return_ratio(r)` | `data.stats.risk_return_ratio()` | |
| `qs.stats.cpc_index(r)` | `data.stats.cpc_index()` | |
| `qs.stats.common_sense_ratio(r)` | `data.stats.common_sense_ratio()` | |
| `qs.stats.tail_ratio(r)` | `data.stats.tail_ratio()` | |
| `qs.stats.outlier_win_ratio(r)` | `data.stats.outlier_win_ratio()` | |
| `qs.stats.outlier_loss_ratio(r)` | `data.stats.outlier_loss_ratio()` | |
| `qs.stats.kelly_criterion(r)` | `data.stats.kelly_criterion()` | |
| `qs.stats.exposure(r)` | `data.stats.exposure()` | |
| `qs.stats.cagr(r)` | `data.stats.cagr()` | |
| `qs.stats.rar(r)` | `data.stats.rar()` | |
| `qs.stats.recovery_factor(r)` | `data.stats.recovery_factor()` | |
| `qs.stats.risk_of_ruin(r)` | `data.stats.risk_of_ruin()` | |
| `qs.stats.ulcer_index(r)` | `data.stats.ulcer_index()` | |
| `qs.stats.ulcer_performance_index(r)` | `data.stats.ulcer_performance_index()` | |
| `qs.stats.serenity_index(r)` | `data.stats.serenity_index()` | |
| `qs.stats.information_ratio(r, benchmark=b)` | `data.stats.information_ratio()` | raw by default, matching QuantStats; see note below |
| `qs.stats.r_squared(r, benchmark=b)` | `data.stats.r_squared()` | |
| `qs.stats.r2(r, benchmark=b)` | `data.stats.r_squared()` | `r2` is a QuantStats alias; use `r_squared()` |
| `qs.stats.greeks(r, benchmark=b)` | `data.stats.greeks()` | |
| `qs.stats.probabilistic_sharpe_ratio(r)` | `data.stats.probabilistic_sharpe_ratio()` | |
| `qs.stats.probabilistic_sortino_ratio(r)` | `data.stats.probabilistic_sortino_ratio()` | |
| `qs.stats.probabilistic_adjusted_sortino_ratio(r)` | `data.stats.probabilistic_adjusted_sortino_ratio()` | |
| `qs.stats.probabilistic_ratio(r)` | `data.stats.probabilistic_ratio()` | |
| `qs.stats.geometric_mean(r)` | `data.stats.geometric_mean()` | |
| `qs.stats.ghpr(r)` | `data.stats.geometric_mean()` | `ghpr` is a QuantStats alias for geometric mean |
| `qs.stats.expected_return(r)` | `data.stats.expected_return()` | |
| `qs.stats.comp(r)` | `data.stats.comp()` | Total compounded return |
| `qs.stats.compsum(r)` | `data.stats.compsum()` | Rolling compounded returns series |
| `qs.stats.distribution(r)` | `data.stats.distribution()` | Return distribution by period (daily/weekly/monthly/quarterly/yearly) |
| `qs.stats.outliers(r)` | `data.stats.outliers()` | |
| `qs.stats.remove_outliers(r)` | `data.stats.remove_outliers()` | |
| `qs.stats.drawdown_details(r)` | `data.stats.drawdown_details()` | |
| `qs.stats.monthly_returns(r)` | `data.stats.monthly_returns()` | |
| `qs.stats.compare(r, benchmark=b)` | `data.stats.compare()` | |
| `qs.stats.montecarlo(r)` | `data.stats.montecarlo()` | Block-bootstrap Monte Carlo simulation |
| `qs.stats.montecarlo_sharpe(r)` | `data.stats.montecarlo_sharpe()` | Monte Carlo Sharpe distribution |
| `qs.stats.montecarlo_drawdown(r)` | `data.stats.montecarlo_drawdown()` | Monte Carlo max drawdown distribution |
| `qs.stats.montecarlo_cagr(r)` | `data.stats.montecarlo_cagr()` | Monte Carlo CAGR distribution |
| `qs.stats.rolling_sharpe(r)` | `data.stats.rolling_sharpe()` | |
| `qs.stats.rolling_sortino(r)` | `data.stats.rolling_sortino()` | |
| `qs.stats.rolling_volatility(r)` | `data.stats.rolling_volatility()` | |
| `qs.stats.rolling_greeks(r, benchmark=b)` | `data.stats.rolling_greeks()` | |
| `qs.stats.autocorr_penalty(r)` | `data.stats.autocorr_penalty()` | |
| `qs.stats.pct_rank(r)` | `data.stats.pct_rank()` | |

#### QuantStats functions with no jquantstats equivalent

The following QuantStats functions are intentionally not exposed as public
methods in jquantstats.  Each is either a shorthand alias for a canonical
method or a utility that is subsumed by a richer jquantstats API.  See the
[QuantStats Parity Benchmark](benchmark.md) for full rationale.

| QuantStats | Rationale |
|---|---|
| `qs.stats.cvar(r)` | Alias for `conditional_value_at_risk()` — jquantstats uses full names only |
| `qs.stats.var(r)` | Alias for `value_at_risk()` — jquantstats uses full names only |
| `qs.stats.ror(r)` | Alias for `risk_of_ruin()` — jquantstats uses full names only |
| `qs.stats.upi(r)` | Alias for `ulcer_performance_index()` — jquantstats uses full names only |
| `qs.stats.expected_shortfall(r)` | Alias for `conditional_value_at_risk()` — jquantstats uses the canonical CVaR term |
| `qs.stats.to_drawdown_series(r)` | Covered by `data.stats.drawdown()` (series) and `data.stats.drawdown_details()` (episode table) |

#### jquantstats-only stats methods

These methods have **no QuantStats equivalent** and are unique to jquantstats.

| jquantstats | Description |
|---|---|
| `data.stats.acf(nlags=20)` | Full autocorrelation function series |
| `data.stats.annual_breakdown()` | Year-by-year performance table |
| `data.stats.autocorr(lag=1)` | Autocorrelation at a given lag (qs only has `autocorr_penalty`) |
| `data.stats.avg_drawdown()` | Average drawdown across all drawdown episodes |
| `data.stats.down_capture()` | Downside capture ratio vs benchmark |
| `data.stats.hhi_positive()` | Herfindahl-Hirschman concentration index for positive returns |
| `data.stats.hhi_negative()` | Herfindahl-Hirschman concentration index for negative returns |
| `data.stats.max_drawdown_duration()` | Duration of the maximum drawdown in periods |
| `data.stats.monthly_win_rate()` | Win rate computed on monthly aggregated returns |
| `data.stats.periods_per_year` | Inferred annualisation factor (property) |
| `data.stats.sharpe_variance()` | Variance-penalised Sharpe variant |
| `data.stats.summary()` | Composite stats summary as a `pl.DataFrame` |
| `data.stats.up_capture()` | Upside capture ratio vs benchmark |
| `data.stats.worst_n_periods(n=5)` | Worst N individual return periods |

### Plots

| QuantStats | jquantstats |
|---|---|
| `qs.plots.snapshot(r)` | `data.plots.snapshot()` |
| `qs.plots.returns(r)` | `data.plots.returns()` |
| `qs.plots.log_returns(r)` | `data.plots.log_returns()` |
| `qs.plots.daily_returns(r)` | `data.plots.daily_returns()` |
| `qs.plots.yearly_returns(r)` | `data.plots.yearly_returns()` |
| `qs.plots.monthly_returns(r)` | `data.plots.monthly_returns()` |
| `qs.plots.monthly_heatmap(r)` | `data.plots.monthly_heatmap()` |
| `qs.plots.histogram(r)` | `data.plots.histogram()` |
| `qs.plots.distribution(r)` | `data.plots.distribution()` |
| `qs.plots.drawdown(r)` | `data.plots.drawdown()` |
| `qs.plots.drawdowns_periods(r)` | `data.plots.drawdowns_periods()` |
| `qs.plots.earnings(r)` | `data.plots.earnings()` |
| `qs.plots.rolling_sharpe(r)` | `data.plots.rolling_sharpe()` |
| `qs.plots.rolling_sortino(r)` | `data.plots.rolling_sortino()` |
| `qs.plots.rolling_volatility(r)` | `data.plots.rolling_volatility()` |
| `qs.plots.rolling_beta(r, benchmark=b)` | `data.plots.rolling_beta()` |
| `qs.plots.montecarlo(r)` | `data.plots.montecarlo()` |
| `qs.plots.montecarlo_distribution(r)` | `data.plots.montecarlo_distribution()` |
| — | `data.plots.compare()` |
| — | `data.plots.assets()` |

All `data.plots.*` methods return an interactive **Plotly figure** instead
of a static matplotlib figure.

### Reports

```python
# QuantStats
qs.reports.full(returns_pd, benchmark=benchmark_pd)
qs.reports.metrics(returns_pd)

# jquantstats
data.reports.metrics()                  # pl.DataFrame; mode="full" for the extended set
data.reports.full(title="My Strategy")  # str — self-contained HTML document

# The Portfolio route spells it differently: .report (singular), and the
# HTML entry point is to_html(), which returns a str or writes to `path`.
pf.report.to_html(title="My Portfolio", path="report.html")
```

There is no `data.reports.summary()`.  The composite stats table lives on the
stats accessor instead, as `data.stats.summary()` → `pl.DataFrame`.

---

## Accessing results

QuantStats functions return scalars. jquantstats methods return a `dict`
keyed by column name.

```python
# QuantStats — scalar
sharpe = qs.stats.sharpe(returns_pd)   # 1.23

# jquantstats — dict
result = data.stats.sharpe()            # {"MyStrategy": 1.23}
sharpe = result["MyStrategy"]           # 1.23
```

Multi-asset results come back in the same call:

```python
returns_multi = pl.DataFrame({
    "Date": dates,
    "AAPL": aapl_rets,
    "MSFT": msft_rets,
})
data = jqs.Data.from_returns(returns=returns_multi)
data.stats.sharpe()
# → {"AAPL": 1.34, "MSFT": 0.91}
```

---

## Behavioural differences

### NaN / null handling

pandas (and QuantStats) silently drop `NaN` values in most calculations.
Polars propagates `null` by default — if any value in a column is `null`,
most statistics will return `null` instead of a numeric result.

> **Note:** in Polars, `null` (a missing entry, equivalent to pandas `NaN`)
> and `NaN` (IEEE-754 "Not a Number") are distinct. jquantstats treats `null`
> as missing; `NaN` is a numeric value that propagates through calculations.

You can clean data manually before constructing `Data`:

```python
returns_pl = returns_pl.drop_nulls()
# or fill forward
returns_pl = returns_pl.with_columns(pl.all().forward_fill())
```

Or use the `null_strategy` parameter on `Data.from_returns` / `Data.from_prices`:

```python
# Mirrors pandas / QuantStats behaviour: silently drop rows with nulls
data = jqs.Data.from_returns(returns=returns_pl, null_strategy="drop")

# Forward-fill nulls before computing statistics
data = jqs.Data.from_returns(returns=returns_pl, null_strategy="forward_fill")

# Raise an informative error if any null is found (useful during development)
data = jqs.Data.from_returns(returns=returns_pl, null_strategy="raise")
```

The default (`null_strategy=None`) passes nulls through unchanged.

### `conditional_value_at_risk` — alpha vs confidence

The tail can be named either way round.  `confidence` is the QuantStats
spelling; `alpha = 1 − confidence` is the tail probability, matching
`value_at_risk` on the same object.  Passing **both** in one call raises
`ValueError`, and omitting both gives `alpha=0.05` (95% confidence).

```python
# QuantStats — confidence level
qs.stats.conditional_value_at_risk(r, confidence=0.95)

# jquantstats — either spelling, same result
data.stats.conditional_value_at_risk(confidence=0.95)
data.stats.conditional_value_at_risk(alpha=0.05)
```

Note the asymmetry with `value_at_risk`, which takes `alpha` only and has no
`confidence` parameter.

### `information_ratio` — annualisation

Both libraries return a **raw, non-annualised** information ratio by default,
so the ported number matches QuantStats as-is.  jquantstats can additionally
annualise on request, scaling by `√periods_per_year`:

```python
# QuantStats — raw
qs.stats.information_ratio(r, benchmark=b)

# jquantstats — raw (default), matching the above
data.stats.information_ratio()

# jquantstats — annualised
data.stats.information_ratio(annualise=True)
```

### No top-level functions

QuantStats exposes bare module-level functions. jquantstats has none.

```python
# ❌ does not exist
jqs.stats.sharpe(returns)

# ✅ correct
data = jqs.Data.from_returns(returns=returns_pl)
data.stats.sharpe()
```

---

## Portfolio-only features

These capabilities have no QuantStats equivalent and are only available
through the `Portfolio` entry point.

```python
pf = jqs.Portfolio.from_cash_position(prices, positions, aum=1_000_000)

# Execution-delay analysis
pf_lagged = pf.lag(1)                    # shift positions forward by 1 day
pf.plots.lead_lag_ir_plot(start=-5, end=5)   # information ratio across a lag range

# Tilt / timing decomposition
pf.tilt                                  # constant-weight (allocation skill)
pf.timing                                # weight deviations (timing skill)
pf.tilt_timing_decomp                    # side-by-side NAV comparison

# Turnover analytics
pf.turnover                              # daily one-way turnover (fraction of AUM)
pf.turnover_weekly()                     # weekly aggregate
pf.turnover_summary()                    # pl.DataFrame: metric/value rows for
                                         # mean_daily_turnover, mean_weekly_turnover,
                                         # turnover_std

# Cost modelling
from jquantstats import CostModel

pf_a = jqs.Portfolio.from_cash_position(
    prices, positions, aum,
    cost_model=CostModel.per_unit(0.01),    # £0.01 per share traded
)
pf_b = jqs.Portfolio.from_cash_position(
    prices, positions, aum,
    cost_model=CostModel.turnover_bps(5),   # 5 bps per unit of AUM turnover
)

pf.trading_cost_impact(max_bps=20)          # sweep cost sensitivity 0 → 20 bps
```

---

## Minimal migration checklist

1. **Install** `jquantstats` (`pip install jquantstats`).
2. **Convert** your `pd.Series` to a `pl.DataFrame` with a date column.
3. **Construct** a `Data` object once with `Data.from_returns(...)`.
4. **Replace** every `qs.stats.foo(r)` call with `data.stats.foo()["col"]`.
5. **Drop** any `NaN`-filled rows before passing data in, or use `null_strategy`.

---

## Further reading

- [API Reference](api.md) — complete method signatures
- [API Stability](STABILITY.md) — versioning and deprecation policy
