---
name: portfolio-robustness
description: Stress-test a jquantstats result rather than just computing it — is this Sharpe real, does it survive execution delay, trading costs, autocorrelation, outliers, or a different sub-period? Use when asked whether a backtest holds up, how sensitive it is to costs or lag, whether performance is luck, or for Monte Carlo, probabilistic Sharpe, or lead/lag analysis.
---

# Is the result real?

Computing a Sharpe and asking whether it means anything are different jobs. This
skill is the second one.

Start from the shared context — read
`${CLAUDE_PLUGIN_ROOT}/reference/portfolio-context.md`, then:

```python
import os, sys

sys.path.insert(0, os.path.join(os.environ["CLAUDE_PLUGIN_ROOT"], "scripts"))
from jqs_context import load

pf = load()
```

Record each stress case as its own **named context** so the comparison is
reproducible rather than a one-off object (`jqs_load.py ... --name lag1`).

## Six ways a good backtest dies

Work down this list; the early entries kill more strategies than the late ones.

### 1. Execution delay

If the edge disappears when positions are shifted by a period, it was reading
the future. This is a `Portfolio`-only test — a return series cannot express it.

```python
{n: pf.lag(n).stats.sharpe()["returns"] for n in range(4)}

pf.plots.lead_lag_ir_plot(start=-10, end=19)  # IR across the lag spectrum
pf.plots.lagged_performance_plot(lags=[0, 1, 2, 5])
```

A peak at lag 0 that falls off a cliff at lag 1 is the signature to report. The
lead side (negative lags) matters too: information ratio that is strong *ahead*
of the trade is a look-ahead artefact, not skill.

### 2. Trading costs

```python
pf.trading_cost_impact(max_bps=20)  # pl.DataFrame — a cost sweep
pf.plots.trading_cost_impact_plot(max_bps=20)
pf.cost_adjusted_returns(cost_bps=5)  # a returns *frame*, not a metric
pf.turnover_summary()  # what the sweep is actually charging for

# Metrics on a cost- or fee-adjusted series go through pf.as_data(), which keeps
# only the 'returns' column. Data.from_returns(frame) would also score the
# 'profit' and 'NAV_accumulated' columns as if they were assets.
net = pf.deduct_management_fee(annual_fee=0.0085, base=pf.cost_adjusted_returns(cost_bps=5))
pf.as_data(net).stats.sharpe()
```

Report the bps level at which the edge reaches zero — that number is far more
useful than a Sharpe at one assumed cost. Pair it with turnover: high turnover
plus cost sensitivity is the common way a paper result fails in production.

### 3. Autocorrelation inflating the Sharpe

Serially correlated returns understate risk, so the Sharpe flatters.

```python
pf.stats.autocorr(lag=1)
pf.stats.acf(nlags=20)
pf.stats.autocorr_penalty()
pf.stats.smart_sharpe()  # sharpe / autocorr_penalty
pf.stats.smart_sortino()
```

A large gap between `sharpe` and `smart_sharpe` is the finding. Illiquid or
smoothed holdings are the usual cause.

### 4. Statistical significance

A Sharpe from a short sample is barely an estimate.

```python
pf.stats.sharpe_variance()  # sampling variance of the estimate
pf.stats.probabilistic_sharpe_ratio()  # P(true Sharpe > benchmark Sharpe)
pf.stats.probabilistic_sortino_ratio()
pf.stats.probabilistic_adjusted_sortino_ratio()
pf.stats.probabilistic_ratio(base="sharpe")  # or pass a callable
```

`probabilistic_sharpe_ratio` returns NaN when the standard deviation or the
higher moments are unusable, or when estimated Sharpe variance is non-positive —
treat NaN as "the sample cannot support this claim", not as an error to route
around.

### 5. Path dependence — Monte Carlo

Block bootstrap, so serial structure is partly preserved:

```python
pf.stats.montecarlo(n=1000, period=252)  # pl.DataFrame (n, n_assets) terminal returns
pf.stats.montecarlo_sharpe(n=1000, period=252)
pf.stats.montecarlo_cagr(n=1000, period=252)
pf.stats.montecarlo_drawdown(n=1000, period=252)  # the distribution that sizing depends on

pf.data.plots.montecarlo(n=100, period=252)  # fan chart
pf.data.plots.montecarlo_distribution(n=1000, period=252)
```

Report where the *observed* result sits in the simulated distribution. The
drawdown distribution is usually the decision-relevant one: a strategy whose
95th-percentile simulated drawdown exceeds the risk budget is not viable at that
size regardless of its Sharpe.

### 6. Concentration — a few days carrying everything

```python
pf.stats.worst_n_periods(n=5)
pf.stats.outliers(quantile=0.95)
pf.stats.remove_outliers(quantile=0.95)  # recompute metrics on the trimmed series
pf.stats.outlier_win_ratio(), pf.stats.outlier_loss_ratio()
pf.stats.hhi_positive(), pf.stats.hhi_negative()  # concentration of gains / losses
```

If the Sharpe collapses once the top 5% of days are removed, the result rests on
a handful of observations. Say so explicitly.

## Stability through time

Full-sample metrics hide regime dependence.

```python
pf.stats.rolling_sharpe(rolling_period=126)
pf.stats.rolling_volatility()
pf.stats.annual_breakdown()
pf.stats.pct_rank(window=60)
pf.plots.annual_sharpe_plot()

pf.truncate(start="2018-01-01", end="2020-12-31").stats.sharpe()
```

Record sub-periods as named contexts (`--start` / `--end`) rather than inline
truncation when the comparison is the deliverable.

## Position smoothing

Whether the signal needs its exact timing, or works held longer:

```python
pf.smoothed_holding(5).stats.sharpe()
pf.plots.smoothed_holdings_performance_plot(windows=[1, 5, 10, 20])
```

Performance that improves under smoothing usually means the raw signal is noisy
and turnover is being wasted.

## Ruin and sizing

```python
pf.stats.risk_of_ruin()
pf.stats.kelly_criterion()
pf.stats.ulcer_index(), pf.stats.serenity_index()
pf.stats.max_drawdown_duration()  # calendar days underwater
pf.stats.recovery_factor()
```

## How to report

- Lead with what **failed**, not what passed. A robustness pass whose findings
  are all reassuring is usually a pass that did not test hard enough.
- Give the breaking point, not a verdict: the bps at which the edge vanishes,
  the lag at which IR crosses zero, the percentile the observed result occupies.
- Name the sample. A 320-row backtest cannot support strong claims regardless of
  the metric, and `sharpe_variance` / `probabilistic_sharpe_ratio` should be
  quoted alongside any headline Sharpe from a short history.
- Carry the digest's warnings through. Nulls, a positional date axis, or a
  misinferred `periods_per_year` invalidate a robustness conclusion exactly as
  they invalidate the underlying metric — if any are outstanding, use
  `portfolio-diagnostics` first.
