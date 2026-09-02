---
name: portfolio-diagnostics
description: Diagnose wrong, null or surprising jquantstats results — null propagation, a misinferred periods_per_year, a lost date axis, benchmark misalignment, or a raised JQuantStatsError. Use when a metric returns null or NaN, when numbers disagree with QuantStats or another tool, when a Sharpe or volatility looks implausible, or when construction fails.
---

# Diagnosing a portfolio

Metrics here fail *quietly* more often than loudly. A wrong number that looks
plausible is the failure mode to hunt.

## Start with the digest, not the metric

Read `${CLAUDE_PLUGIN_ROOT}/reference/portfolio-context.md`, then rebuild and
re-read the digest — it already carries the evidence:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_load.py show
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_load.py check   # inputs still match?
```

`check` reporting `STALE` explains a whole class of "the number changed" reports:
the inputs moved after the digest was written. Work through the digest's
`warnings`, `nulls`, `nans` and `shape` blocks before touching the metric that
was complained about.

## The five silent failures

### 1. Nulls — inconsistent, not uniform

Polars keeps `null` (missing) and `NaN` (IEEE-754) distinct, and
`null_strategy` acts on **null only**. Handling then varies by metric:
aggregations skip nulls, which shrinks the effective sample without saying so,
while cumulative and rolling paths can propagate them.

```python
frame.null_count()  # nulls per column
frame.select(pl.col(c).is_nan().sum() for c in numeric)  # NaNs are separate
```

Fix by making the sample explicit at construction:
`Data.from_returns(..., null_strategy="drop")`. Options are `None` (default,
pass through), `"drop"`, `"forward_fill"`, `"raise"`. Use `"raise"` while
debugging — turning a silent failure loud is usually the fastest route. When
reproducing QuantStats numbers, `"drop"` is what matches pandas.

`NullsInReturnsError` is the loud version of this, raised where a null cannot be
tolerated at all.

### 2. A lost date axis — the `date` naming trap

The `Portfolio` → `Data` bridge keeps the date column **only when it is named
exactly `date`** (lowercase). Under any other name the dates are dropped and
replaced by a positional integer index. Nothing is raised.

The consequences are not cosmetic: `periods_per_year` falls back to a default
instead of being inferred, so every annualised metric shifts, and date-axis
charts lose their x-axis. On identical data, renaming `date` → `Date` moved
Sharpe from 100.33 to 83.36 in testing.

```python
pf.data.index.columns  # ['date'] good — ['index'] means the axis was dropped
```

The plugin's loader normalises Portfolio inputs to `date` and reports having
done so. If a portfolio was built by hand, check this first.

### 3. `periods_per_year` inferred wrongly

Annualisation of `sharpe`, `sortino` and `volatility` runs off this number.

```python
pf.stats.periods_per_year  # what the library will use
```

The digest also reports `periods_per_year_empirical` — observations per calendar
year measured from the span. A mismatch means the annualisation is wrong by
roughly its square root: business-daily should land near 252, weekly near 52,
monthly near 12. If the inference is off, pass `periods=` (or
`periods_per_year=`, depending on the metric — the two kwarg names are not used
consistently) explicitly rather than accepting the default.

### 4. Benchmark misalignment

The benchmark is passed once at construction and aligned to the returns.
`BenchmarkAlignmentWarning` is emitted when that alignment drops rows — do not
suppress it. `beta`, `alpha`, `r_squared`, `information_ratio` and the capture
ratios then describe only the overlap, which may be much shorter than the
portfolio.

```python
data.benchmark.height, data.returns.height  # compare
```

`NoBenchmarkError` means a benchmark-dependent statistic was requested on a
context built without one — rebuild with `--benchmark`, don't work around it.

### 5. The right name, different semantics

Same metric name, different convention from QuantStats:

- `conditional_value_at_risk` accepts `confidence=0.95` *or* `alpha=0.05`, never
  both (that raises `ValueError`); omitting both gives `alpha=0.05`.
- `value_at_risk` takes `alpha` only — no `confidence` parameter, on the same
  object as its CVaR sibling that has one.
- `information_ratio` is raw and non-annualised by default; pass
  `annualise=True` to scale it.
- `volatility` annualises by default — and spells it `annualize`, with a *z*.

## When construction raises

The library uses typed domain errors, so the class names the problem:

| Exception | Cause |
|---|---|
| `MissingDateColumnError` | no date column where one is required — check `date_col` |
| `InvalidPricesTypeError`, `InvalidCashPositionTypeError` | a pandas object or Series was passed; both must be `pl.DataFrame` |
| `RowCountMismatchError` | prices and positions have different row counts — align before constructing, don't truncate blindly |
| `NonPositiveAumError` | `aum` must be strictly positive |
| `NoAssetColumnsError` | no numeric asset columns — usually the date column was not excluded, or everything parsed as string |
| `PositionExprColumnError` | a position expression names columns absent from prices |
| `NegativeCostBpsError`, `NegativeAnnualFeeError`, `InvalidMaxBpsError` | invalid cost inputs |
| `UncleanSeriesError` | a derived series hit null or non-finite values — trace back to the input nulls |
| `NonPositiveWindowError`, `NonPositivePeriodsPerYearError` | invalid rolling window or annualisation factor |
| `IntegerIndexBoundError` | `truncate` received a non-integer where a row index was expected |

All subclass `JQuantStatsError`, so `except JQuantStatsError` catches the domain
set without swallowing real bugs.

## Reconciling against QuantStats

When numbers disagree with a QuantStats reference, work in this order — the
common cause is upstream of the metric:

1. **Sample.** `null_strategy="drop"` to match pandas' silent `NaN` dropping.
2. **Periodicity.** Confirm `periods_per_year` matches what the other tool assumed.
3. **Date axis.** Confirm the index is temporal, not positional.
4. **Convention.** Check the tail spelling, the annualisation flag, and whether
   the alias you are comparing against maps to a different canonical metric.
5. **Only then** compare the metric itself.

The repo's own `tests/test_jquantstats/test_migration/` validates metrics against
the `quantstats` reference implementation and is the authority on which
differences are intentional.

## Verify names before concluding a metric is broken

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_api.py --show value_at_risk
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_api.py --grep sortino
```

A missing name is usually a QuantStats alias that was never ported (`cvar`,
`ror`, `upi`, `r2`, `ghpr`, `win_loss_ratio`). Report that rather than wrapping it.
