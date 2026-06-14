# jquantstats vs. QuantStats

`jquantstats` started life inspired by [QuantStats](https://github.com/ranaroussi/quantstats),
and the two libraries share a goal: help quants and portfolio managers understand
strategy performance through metrics, plots, and reports. They have since
diverged considerably. This document explains where they differ, what
`jquantstats` adds, and — honestly — where QuantStats still has the edge.

> **TL;DR** — QuantStats is the mature, batteries-included tool for analysing a
> **returns series**, with a huge installed base and tight `yfinance`
> integration. `jquantstats` is a modern, Polars-native rebuild whose
> distinguishing feature is **position-level analysis**: it starts from prices +
> positions, not just returns, which unlocks turnover, cost modelling,
> execution-delay studies, and tilt/timing attribution that QuantStats
> structurally cannot offer.

---

## At a glance

| Dimension | jquantstats | QuantStats |
|---|---|---|
| DataFrame engine | Polars-native (narwhals abstraction; no pandas at runtime) | pandas |
| Plotting | Plotly-native (interactive HTML) | Matplotlib (static); optional `plotly` extra *converts* figures |
| Primary input | **Prices + positions** *or* returns | Returns series only |
| Position-level analytics | Yes (turnover, costs, attribution, lag) | No |
| Market-data fetching | Not built in (bring your own data) | Built in via `yfinance` |
| Python support | 3.11+ | 3.10+ |
| Type hints / `py.typed` | Full, ships marker | Partial |
| Test rigor | 100% line+branch coverage, mutation testing, property tests | Lighter |
| Maturity / community | Newer, smaller | Established, large user base |
| API style | Explicit `Portfolio` / `Data` objects | `extend_pandas()` monkey-patches Series |

---

## 1. Two entry points vs. one

**QuantStats** operates on a single returns series (a pandas `Series` or
`DataFrame`). You typically write:

```python
import quantstats as qs
qs.extend_pandas()                 # monkey-patch helper methods onto pandas
returns.sharpe()                   # now available on the Series
qs.reports.html(returns, "SPY")    # full tear sheet vs. a benchmark
```

This is convenient and discoverable, but it means QuantStats only ever sees
**what already happened to your equity curve**. If you only have returns, that
is all you can analyse.

**jquantstats** offers *two* complementary entry points feeding one analytics
pipeline:

- **`Data.from_returns(...)`** — the QuantStats-equivalent route. Start from a
  returns series (+ optional benchmark, risk-free rate) and get stats, plots,
  and reports.
- **`Portfolio.from_cash_position(...)`** (and `from_position`,
  `from_risk_position`) — the differentiating route. Start from **raw prices and
  positions**, and `jquantstats` compiles the NAV/returns chain itself.

A `Portfolio` exposes `portfolio.data`, so you can always drop down into the
returns-only API. The bridge is one-directional: a `Data` object built from
returns can never recover the positions it never had.

---

## 2. The big differentiator: position-level analytics

Because `jquantstats` knows your *positions*, not just your *returns*, it can
answer questions QuantStats cannot even pose. This is the heart of the
difference.

### Turnover
`portfolio.turnover`, `turnover_weekly`, and `turnover_summary()` quantify how
much trading the strategy actually does — essential for understanding capacity
and realism. From a returns series alone, turnover is unknowable.

### Trading costs
Two independent, deliberately-non-combinable cost models:

- **Per-unit** (`CostModel.per_unit(...)`) — e.g. £0.01/share, applied at
  construction and reflected in `net_cost_nav`.
- **Turnover-bps** (`CostModel.turnover_bps(...)`) — e.g. 5 bps of AUM turnover,
  used by `trading_cost_impact(max_bps)` to **sweep** cost assumptions and show
  how Sharpe degrades as costs rise.

A strategy that looks great gross can be unviable net of costs; `jquantstats`
makes that visible. QuantStats has no notion of trading costs because it has no
notion of trades.

### Execution-delay (lead/lag) analysis
`portfolio.lag(n)` shifts positions by *n* periods to simulate execution delay,
returning a new portfolio with recomputed NAV. Combined with
`plots.lead_lag_ir_plot(start, end)` and `lagged_performance_plot(...)`, you can
see how much of the edge survives a T+1 (or T+5) fill. This is a standard
robustness check for systematic strategies — and impossible from returns alone.

### Tilt / timing attribution
`portfolio.tilt`, `portfolio.timing`, and `tilt_timing_decomp` decompose
performance into **allocation skill** (constant average weights) vs. **timing
skill** (deviations from the average). This tells you *why* a strategy worked,
not just *that* it did.

### Position smoothing & correlations
`smoothed_holding(n)` (rolling-average holdings) and `correlation()` across
assets round out the position-aware toolkit.

---

## 3. Shared ground — the metrics both provide

For the returns-series use case, the two libraries overlap heavily.
`jquantstats` reimplements (and validates against QuantStats in its test suite)
the familiar metric set:

- **Ratios**: Sharpe (incl. probabilistic & "smart" variants), Sortino, Omega,
  Calmar, Treynor, Information ratio, recovery factor, CAGR.
- **Risk**: VaR, Conditional VaR (CVaR), Ulcer index, max drawdown and drawdown
  details, risk of ruin, tail ratio, gain-to-pain.
- **Distribution**: skew, kurtosis, payoff ratio, profit factor, win rate,
  consecutive wins/losses, avg win/loss.
- **Benchmark/factor**: alpha, beta, R², tracking error, up/down capture.
- **Rolling**: Sharpe, Sortino, volatility, beta over configurable windows.
- **Monte Carlo**: block-bootstrap distributions of total return, Sharpe, max
  drawdown, CAGR.
- **Temporal**: monthly returns heatmap, annual breakdown, worst-n periods,
  monthly win rate.

If your only goal is "give me a Sharpe and a tear sheet from a returns series,"
both libraries do the job. The QuantStats test dependency exists precisely to
keep `jquantstats`' numbers honest against the reference implementation.

---

## 4. Engineering & ergonomics differences

### Polars vs. pandas
`jquantstats` is **Polars-native** with a `narwhals` abstraction layer, so it
also accepts pandas/other frames at the boundary without a pandas *runtime*
dependency. This brings speed and predictable null semantics (Polars `null` is
distinct from IEEE-754 `NaN`, and null-handling is explicit:
`null_strategy={"raise","drop","forward_fill"}`). QuantStats is pandas through
and through.

### Interactive vs. static plots
`jquantstats` renders **Plotly** figures natively — zoomable, hoverable, and
embeddable as self-contained interactive HTML (`portfolio.report.to_html()`).
QuantStats' core plotting is **Matplotlib** (with `seaborn` as a dependency),
producing static images that are simpler and lighter but not interactive.
Recent QuantStats versions add an optional `plotly` extra, but it is a
`to_plotly(fig)` *converter* that wraps existing Matplotlib figures rather than
a native interactive charting layer — so the experience is still
Matplotlib-first.

### API design
QuantStats favours `extend_pandas()` monkey-patching, which is discoverable but
mutates pandas globally. `jquantstats` uses explicit `Portfolio`/`Data` objects
with lazy, memoised accessors (`.stats`, `.plots`, `.report`, `.utils`) and a
frozen, slotted dataclass core — no global side effects.

### Code quality posture
`jquantstats` enforces 100% line **and** branch coverage, 100% docstring
coverage, strict typing (ships a `py.typed` marker), plus mutation testing
(`mutmut`), property-based tests (`hypothesis`), and snapshot tests (`syrupy`).
This is a heavier quality bar than QuantStats maintains.

---

## 5. Where QuantStats still wins

This comparison would be dishonest without naming QuantStats' genuine
advantages:

- **Maturity & community.** QuantStats has years of use, a large user base,
  Stack Overflow answers, blog tutorials, and battle-tested edge-case handling.
  `jquantstats` is newer and smaller.
- **Built-in data fetching.** QuantStats integrates with `yfinance`, so
  `qs.utils.download_returns("AAPL")` and benchmark comparison "just work."
  `jquantstats` deliberately stays out of the data-acquisition business — you
  bring your own frame.
- **Lower barrier for the returns-only case.** If you *only* have a returns
  series and want a one-liner tear sheet, QuantStats' `qs.reports.html(...)` is
  about as frictionless as it gets, and `extend_pandas()` feels natural to
  pandas users.
- **Slightly broader Python support.** Current QuantStats (0.0.81) requires
  Python 3.10+, versus 3.11+ for `jquantstats` — a one-version gap rather than
  the wide range older QuantStats releases once spanned.
- **Familiarity.** Many desks already have QuantStats in their stack and
  notebooks built around it.

---

## 6. Which should you use?

**Reach for QuantStats when:**
- You have a returns series and want a fast, familiar tear sheet.
- You rely on `yfinance` for prices/benchmarks.
- You value the larger ecosystem, documentation base, and Matplotlib-first plots.

**Reach for jquantstats when:**
- You have **positions and prices**, and want to analyse *how* the strategy
  traded — turnover, costs, execution delay, tilt/timing attribution.
- You want **net-of-cost** realism and cost-sensitivity sweeps.
- You prefer **interactive Plotly** reports and a **Polars-native**, strictly
  typed, heavily tested codebase.
- You're building on modern Python (3.11+) and want explicit objects over
  pandas monkey-patching.

In short: QuantStats analyses the *outcome* (returns). `jquantstats` can analyse
the *outcome and the process that produced it* (positions → NAV → returns) —
and falls back to the returns-only view whenever that's all you have.

---

*Sources: jquantstats `README.md`, `docs/ARCHITECTURE.md`, and `pyproject.toml`
in this repository; [QuantStats on GitHub](https://github.com/ranaroussi/quantstats).*
