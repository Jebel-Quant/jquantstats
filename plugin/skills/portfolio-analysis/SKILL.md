---
name: portfolio-analysis
description: Analyse a jquantstats Portfolio or Data object — pick the right metric or chart for the question, and get the call shape right. Use when asked about performance, risk-adjusted returns, drawdowns, benchmark comparison, turnover, trading costs, execution delay, tilt/timing attribution, or for a tearsheet or report from prices, positions or a return series.
---

# Analysing a portfolio

## First: get the object, don't invent one

Read `${CLAUDE_PLUGIN_ROOT}/reference/portfolio-context.md` and follow it. In
short: if `.jquantstats/context.json` exists, that *is* the portfolio — rebuild
it with `load()` rather than asking the user to re-specify anything.

```python
import os, sys
sys.path.insert(0, os.path.join(os.environ["CLAUDE_PLUGIN_ROOT"], "scripts"))
from jqs_context import load

pf = load()
```

If no context exists, build one first — the digest it prints tells you the asset
count, span, inferred periodicity and null situation, all of which change which
answer is correct.

## The call shape

Three rules cover most mistakes:

**1. Metrics return a dict keyed by column, never a scalar.**

```python
pf.stats.sharpe()                  # {'returns': 2.13}   — Portfolio aggregates to one series
data.stats.sharpe()                # {'GOOG': 1.15, 'AAPL': 1.75}
```

A `Portfolio`'s series is always named `returns`. Multi-asset lives on `Data`.

**2. Half of `Portfolio` is properties, not methods.** Adding `()` raises.

| Access | Members |
|---|---|
| No parens (properties) | `turnover`, `turnover_weekly`, `tilt`, `timing`, `tilt_timing_decomp`, `drawdown`, `highwater`, `monthly`, `profit`, `profits`, `nav_accumulated`, `nav_compounded`, `net_cost_nav`, `position_delta_costs`, `all`, `assets`, `cost_model`, `returns` |
| No parens (dataclass fields) | `prices`, `cashposition`, `aum`, `cost_per_unit`, `cost_bps`, `annual_fee` |
| Call them | `lag(n)`, `truncate(start, end)`, `smoothed_holding(n)`, `turnover_summary()`, `trading_cost_impact(max_bps)`, `cost_adjusted_returns(cost_bps)`, `deduct_management_fee(annual_fee)`, `correlation(frame)`, `describe()` |

On `Data`: `assets`, `date_col`, `all` are properties; `returns`, `index`,
`benchmark` are fields; `head`, `tail`, `items`, `copy`, `resample`,
`truncate`, `describe` are methods.

**3. Periodicity kwargs are not uniformly named.** `sharpe`, `sortino`,
`volatility`, `omega`, `treynor_ratio` take `periods`; `information_ratio`,
`greeks`, `rolling_greeks`, `montecarlo_sharpe`, `montecarlo_cagr` take
`periods_per_year`. Both default to `None` → inferred. Check the digest's
`periods_per_year` before trusting any annualised number.

## Which tool for which question

| The question | The call |
|---|---|
| Headline numbers | `pf.stats.summary()` → `pl.DataFrame`; `data.reports.metrics(mode="basic"\|"full")` |
| Risk-adjusted return | `sharpe`, `sortino`, `adjusted_sortino`, `calmar`, `omega`, `rar`, `risk_return_ratio` |
| Drawdowns | `max_drawdown`, `drawdown_details` (episode table), `avg_drawdown`, `max_drawdown_duration`, `ulcer_index`, `ulcer_performance_index`, `recovery_factor`, `serenity_index` |
| Tails | `value_at_risk(alpha=0.05)`, `conditional_value_at_risk(confidence=0.95)`, `tail_ratio`, `worst_n_periods(n)`, `outliers`, `outlier_win_ratio` |
| Distribution shape | `skew`, `kurtosis`, `distribution`, `geometric_mean`, `expected_return`, `hhi_positive`, `hhi_negative` |
| Win/loss behaviour | `win_rate`, `monthly_win_rate`, `payoff_ratio`, `profit_factor`, `profit_ratio`, `gain_to_pain_ratio`, `consecutive_wins`, `consecutive_losses`, `kelly_criterion`, `risk_of_ruin` |
| Versus a benchmark | `greeks` (alpha/beta), `rolling_greeks(rolling_period)`, `information_ratio`, `r_squared`, `treynor_ratio`, `up_capture`, `down_capture` |
| Calendar breakdown | `monthly_returns`, `annual_breakdown`, `compare(aggregate="ME")` |
| Trading behaviour | `pf.turnover`, `pf.turnover_weekly`, `pf.turnover_summary()`, `stats.exposure()` |
| Cost sensitivity | `pf.trading_cost_impact(max_bps=20)`, `pf.cost_adjusted_returns(cost_bps)`, `pf.net_cost_nav`, `pf.deduct_management_fee(annual_fee)` |
| Execution delay | `pf.lag(n)`, `pf.plots.lagged_performance_plot(lags=[...])`, `pf.plots.lead_lag_ir_plot(start, end)` |
| Allocation vs timing skill | `pf.tilt`, `pf.timing`, `pf.tilt_timing_decomp` |
| Position smoothing | `pf.smoothed_holding(n)`, `pf.plots.smoothed_holdings_performance_plot(windows=[...])` |

A benchmark must be passed **at construction** (`Data.from_returns(..., benchmark=...)`),
not per call — so benchmark-relative metrics need a `Data` context that has one.
`greeks`, `information_ratio`, `rolling_greeks` and `treynor_ratio` take a
`benchmark=` *column name*; `up_capture` and `down_capture` are the odd ones out,
taking a `pl.Series`.

## The accessor asymmetry

`Portfolio` and `Data` are not symmetric, and this is the most common error:

| | `Data` | `Portfolio` |
|---|---|---|
| Stats | `data.stats` | `pf.stats` |
| Plots | `data.plots` — 20 charts | `pf.plots` — 10 charts, mostly different ones |
| Reports | `data.reports` → `.metrics()`, `.full()` | `pf.report` → `.to_html()` |

Note the singular/plural: **`Data.reports`, `Portfolio.report`**.

`PortfolioPlots` is not a superset. It offers exactly `snapshot`,
`correlation_heatmap`, `monthly_returns_heatmap`, `rolling_sharpe_plot`,
`rolling_volatility_plot`, `annual_sharpe_plot`, `lead_lag_ir_plot`,
`lagged_performance_plot`, `smoothed_holdings_performance_plot`,
`trading_cost_impact_plot`.

For anything else — `drawdown()`, `histogram()`, `monthly_heatmap()`,
`drawdowns_periods()`, `rolling_beta()`, `distribution()` — drop to the returns
view, which is always available and shares the same underlying series:

```python
pf.data.plots.drawdown()
pf.data.stats.calmar()
```

## Charts

Plots return a Plotly `go.Figure`, which a terminal cannot render. Write it out
and tell the user the path:

```python
fig = pf.plots.snapshot()
fig.write_html("_analysis/snapshot.html")        # interactive
fig.write_image("_analysis/snapshot.png")        # static; needs kaleido
```

For a full document: `pf.report.to_html(title=..., path=...)` returns a `Path`
when `path` is given, otherwise the HTML string. On the `Data` side,
`data.reports.full(title=...)` returns a self-contained HTML string.

## Reporting numbers

- Say which context produced them when more than one exists.
- State the periodicity actually used when quoting anything annualised.
- `information_ratio` is raw (non-annualised) by default, matching QuantStats;
  pass `annualise=True` for the √periods-scaled version.
- `volatility` annualises by default (`annualize=True` — note the *z*, unlike
  `information_ratio`'s `annualise`).
- If the digest carried warnings, resolve or repeat them alongside the numbers
  rather than quoting past them. Nulls and a misinferred `periods_per_year` both
  produce plausible-looking wrong answers.

## Before writing an unfamiliar call

Confirm the symbol exists — invented method names are the expensive failure here:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_api.py --show sortino
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_api.py --grep capture
```
