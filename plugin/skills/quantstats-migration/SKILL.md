---
name: quantstats-migration
description: Translate QuantStats (pandas) code into jquantstats (Polars) — the function-to-object shift, the API mapping, and the semantic differences that silently change numbers. Use whenever writing or reviewing jquantstats code, porting a `qs.stats.*` / `qs.plots.*` / `qs.reports.*` workflow, or converting a pandas tearsheet to Polars.
---

# QuantStats → jquantstats

jquantstats is **not** a drop-in replacement for QuantStats. It is a different
library with a similar vocabulary. Most metric *names* carry over; the call
shape, the data type, and the return type do not.

If you are about to write `jqs.stats.sharpe(returns)`, stop — that function does
not exist. Read the model below first.

## The model

| | QuantStats | jquantstats |
|---|---|---|
| Call shape | module-level function | method on an instance |
| Input | `pd.Series` with `DatetimeIndex` | `pl.DataFrame` with a date column |
| Return | scalar | `dict` keyed by column name |
| Benchmark | passed per call | passed once at construction |
| Plots | matplotlib | Plotly `go.Figure` |

```python
# QuantStats
sharpe = qs.stats.sharpe(returns_pd)                  # 1.23

# jquantstats
data = jqs.Data.from_returns(returns=returns_pl)      # once
sharpe = data.stats.sharpe()["MyStrategy"]            # {"MyStrategy": 1.23}
```

There are **no module-level analytics functions** in jquantstats. Everything
hangs off a constructed object.

## Step 1 — get the data into Polars

```python
import polars as pl

# from an existing pd.Series
returns_pl = pl.from_pandas(returns_pd.rename("MyStrategy").reset_index())
# → columns ["Date", "MyStrategy"]
```

The date column is auto-detected as the first temporal column, whatever it is
called, and is exposed as `date` on the resulting object (`data.date_col ==
["date"]`). Pass `date_col=` only to nominate a column explicitly — required
for a date-free, integer-indexed frame. Every other column is an asset —
multi-asset is native, so port a loop over series into a single wide frame and
one call.

**Nulls matter.** pandas drops `NaN` silently; Polars propagates `null`, so one
missing value turns a whole metric into `null`. To reproduce QuantStats numbers,
pass `null_strategy="drop"`:

```python
data = jqs.Data.from_returns(returns=returns_pl, null_strategy="drop")
```

Options: `None` (default, pass through), `"drop"`, `"forward_fill"`, `"raise"`.
Note that in Polars `null` (missing) and `NaN` (IEEE-754) are distinct;
`null_strategy` acts on `null` only.

## Step 2 — pick an entry point

**`Data`** — the QuantStats-equivalent route. Use it for any return or price
series.

```python
import jquantstats as jqs

data = jqs.Data.from_returns(returns=returns_pl, rf=0.0, benchmark=benchmark_pl)
data = jqs.Data.from_prices(prices=prices_pl)          # same kwargs
```

**`Portfolio`** — no QuantStats equivalent. Use it when the user has prices
*and* positions, or asks for turnover, trading costs, execution delay, or
tilt/timing attribution.

```python
pf = jqs.Portfolio.from_cash_position(prices, cash_position, aum=1_000_000)
pf = jqs.Portfolio.from_position(prices, position, aum)          # share counts
pf = jqs.Portfolio.from_risk_position(prices, risk_position, aum, vola=32)
```

`vola` is an EWMA lookback in periods (`int`, or `dict[str, int]` per asset) —
not a volatility frame.

## Step 3 — accessors

The two entry points are **not symmetric**. This is the most common mistake:

| | `Data` | `Portfolio` |
|---|---|---|
| Stats | `data.stats` | `pf.stats` |
| Plots | `data.plots` (`DataPlots`) | `pf.plots` (`PortfolioPlots` — a different, smaller set) |
| Reports | `data.reports` → `.metrics()`, `.full()` | `pf.report` → `.to_html()` |

Note the singular/plural split: **`Data.reports`, `Portfolio.report`**.

`PortfolioPlots` is not a superset of `DataPlots`. It offers only
`snapshot`, `correlation_heatmap`, `monthly_returns_heatmap`,
`rolling_sharpe_plot`, `rolling_volatility_plot`, `annual_sharpe_plot`,
`lead_lag_ir_plot`, `lagged_performance_plot`,
`smoothed_holdings_performance_plot`, `trading_cost_impact_plot`.

For anything else — `drawdown()`, `histogram()`, `monthly_heatmap()`, … — drop
down to the returns view, which is always available:

```python
pf.data.plots.drawdown()
pf.data.stats.calmar()
```

## Step 4 — translate the calls

The default rule covers the large majority of metrics:

```
qs.stats.foo(r, **kw)   →   data.stats.foo(**kw)["col"]
qs.plots.foo(r)         →   data.plots.foo()            # returns a Plotly figure
```

### Names that do not carry over

jquantstats exposes canonical names only — QuantStats' short aliases are absent
and raise `AttributeError`:

| QuantStats | jquantstats |
|---|---|
| `var` | `value_at_risk()` |
| `cvar`, `expected_shortfall` | `conditional_value_at_risk()` |
| `ror` | `risk_of_ruin()` |
| `upi` | `ulcer_performance_index()` |
| `r2` | `r_squared()` |
| `ghpr` | `geometric_mean()` |
| `win_loss_ratio` | `payoff_ratio()` |
| `to_drawdown_series` | `drawdown()` (series) / `drawdown_details()` (episode table) |

### Numbers that differ for the same name

- **`conditional_value_at_risk`** accepts either spelling of the tail —
  `confidence=0.95` (the QuantStats convention) or `alpha=0.05` — but not both
  in the same call, which raises `ValueError`. Omitting both gives
  `alpha=0.05`.
- **`value_at_risk`** takes `alpha` (default `0.05`) only; it has no
  `confidence` parameter, unlike its CVaR sibling on the same object.
- **`information_ratio`** matches QuantStats by default (raw, non-annualised).
  Pass `annualise=True` for the √periods-scaled version.
- **`volatility`** annualises by default (`annualize=True`).
- Periodicity kwargs are not uniform: `sharpe`/`sortino`/`volatility` take
  `periods`, while `information_ratio`/`greeks` take `periods_per_year`. Both
  default to `None` → inferred from the data (`data.stats.periods_per_year`).

### Reports

```python
data.reports.metrics(mode="basic")     # pl.DataFrame; mode="full" for the extended set
data.reports.full(title="...")         # str — self-contained HTML document
pf.report.to_html(title="...", path=None)   # str, or Path when path is given
```

There is no `data.reports.summary()` and no `data.reports.to_html()`. The
composite stats table is `data.stats.summary()` → `pl.DataFrame`.

## Step 5 — verify, don't recall

This mapping is a summary of a moving API. Before finalising code, confirm the
symbols exist rather than trusting memory:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_api.py --show sharpe
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_api.py --grep drawdown
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_api.py stats
```

`--show` reports signature and docstring from the installed version, and says so
plainly when a name is absent. A missing name was either renamed to its
canonical form (see the table above) or never ported — say so rather than
inventing a wrapper.

## Step 6 — record the context once

Ported code that rebuilds its own portfolio in every script drifts. Record the
construction once so later analysis shares one object:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_load.py data \
    --returns data/returns.csv --benchmark data/spy.csv --null-strategy drop
```

See `${CLAUDE_PLUGIN_ROOT}/reference/portfolio-context.md`. Subsequent snippets
then use `load()` instead of repeating the constructor, and the digest it prints
surfaces the null and periodicity problems that make ported numbers disagree
with QuantStats.

## Portfolio-only capabilities

Reach for these when the request goes beyond what a return series can express;
none have a QuantStats counterpart.

Mind which of these are properties: only `lag`, `turnover_summary`,
`trading_cost_impact`, `smoothed_holding` and `truncate` take parentheses.

```python
pf.lag(1)                       # shift positions — execution-delay study
pf.plots.lead_lag_ir_plot()     # IR across lags
pf.tilt, pf.timing, pf.tilt_timing_decomp   # allocation vs timing skill
pf.turnover, pf.turnover_weekly             # properties — no parentheses
pf.turnover_summary()                       # a method
pf.trading_cost_impact(...)     # cost sensitivity sweep

from jquantstats import CostModel
CostModel.per_unit(0.01)        # currency per unit traded
CostModel.turnover_bps(5)       # bps of AUM turnover
# or the shorthand kwargs on any constructor: cost_per_unit=, cost_bps=, annual_fee=
```

## Checklist

1. Convert to `pl.DataFrame` with a leading date column.
2. Construct once — `Data.from_returns(...)`, with `null_strategy="drop"` to
   mirror pandas.
3. Pass the benchmark at construction, not per call.
4. Replace `qs.stats.foo(r)` with `data.stats.foo()["col"]`.
5. Check the alias table and the differing-semantics list before trusting a
   ported number.
6. Use `Portfolio` only when positions exist; otherwise `Data`.
