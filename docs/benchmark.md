---
icon: material/chart-bar
---

# QuantStats Parity Benchmark

This document tracks the feature parity between **jquantstats** and
[QuantStats](https://github.com/ranaroussi/quantstats), with a score out of 10
for each major category.  Gaps are either **closed** (method implemented) or
**documented** (intentional divergence noted below).

---

## Coverage scores

| Category | Score | Notes |
|---|---|---|
| Stats metrics | **10 / 10** | All public QuantStats stats functions covered or documented as intentional divergences |
| Monte Carlo | **10 / 10** | `montecarlo`, `montecarlo_sharpe`, `montecarlo_drawdown`, `montecarlo_cagr` (closed in #751) |
| Rolling metrics | **10 / 10** | `rolling_sharpe`, `rolling_sortino`, `rolling_volatility`, `rolling_greeks` |
| Plot types | **10 / 10** | All QuantStats plot types covered with interactive Plotly equivalents |
| Reports | **10 / 10** | HTML tearsheet, metrics table, and summary report |

---

## Stats metric coverage — intentional divergences

The following six public functions exist in `quantstats.stats` but are not
present as public methods on `data.stats`.  Each is an intentional design
decision, documented here so no gap goes unrecorded.

### Shorthand aliases not carried over

QuantStats exposes several short alias functions alongside their canonical
counterparts.  jquantstats deliberately uses **full names only**, consistent
with its deprecation of earlier aliases (`ghpr()`, `r2()`, `win_loss_ratio()`
were deprecated in favour of `geometric_mean()`, `r_squared()`, and
`payoff_ratio()` respectively).

| QuantStats shorthand | Canonical equivalent in jquantstats | Rationale |
|---|---|---|
| `qs.stats.cvar(r)` | `data.stats.conditional_value_at_risk()` | Prefer unambiguous full name |
| `qs.stats.var(r)` | `data.stats.value_at_risk()` | Prefer unambiguous full name; `var` is also a common Python built-in pattern |
| `qs.stats.ror(r)` | `data.stats.risk_of_ruin()` | Prefer unambiguous full name |
| `qs.stats.upi(r)` | `data.stats.ulcer_performance_index()` | Prefer unambiguous full name |
| `qs.stats.expected_shortfall(r)` | `data.stats.conditional_value_at_risk()` | Alias for CVaR; jquantstats uses the canonical term |

### `to_drawdown_series` — covered by richer API

`qs.stats.to_drawdown_series(returns)` converts a return series to a
drawdown series.  jquantstats provides equivalent — and richer —
functionality through two methods:

- **`data.stats.drawdown()`** — returns a `pl.DataFrame` with the per-period
  drawdown from the running high-water mark, one column per asset.
- **`data.stats.drawdown_details()`** — returns a full episode table
  (start date, end date, max drawdown, duration) for every drawdown period.

A bare drawdown series is therefore available as
`data.stats.drawdown()["asset_col"]`; the standalone conversion utility is
not needed.

---

## Behavioural notes

### `downside_deviation` convention

jquantstats and QuantStats both use the Red Rock Capital Sortino paper
convention: the downside semi-deviation is the root-mean-square of
**strictly negative** returns divided by the **total** observation count
(not only the negative count).  The implementations are aligned.

### `information_ratio` annualisation

Both libraries return a **raw (non-annualised)** information ratio by default,
so the values agree without adjustment.  jquantstats additionally accepts
`annualise=True`, which scales the ratio by `sqrt(periods_per_year)`; QuantStats
has no equivalent option.  See the
[Migration guide](MIGRATION.md#information_ratio-annualisation).
