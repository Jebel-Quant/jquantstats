---
icon: material/help-circle
---

# FAQ & Troubleshooting

Answers to the errors and questions users hit most often. If your problem
isn't covered here, please [open an issue](https://github.com/jebel-quant/jquantstats/issues).

---

## Errors

### `NullsInReturnsError: DataFrame 'returns' contains null values …`

Polars propagates `null` through calculations (pandas silently drops `NaN`),
so jquantstats refuses null-containing input by default. Pick a strategy:

```python
data = Data.from_returns(returns=df, null_strategy="drop")          # drop rows with any null
data = Data.from_returns(returns=df, null_strategy="forward_fill")  # interior forward-fill
```

Or clean the frame yourself before construction. See
[NaN vs null semantics](#what-is-the-difference-between-nan-and-null) below —
they are *not* the same thing.

### `AttributeError: No benchmark data available`

You called a benchmark-relative metric (e.g. `greeks`, `r_squared`,
`information_ratio`) on a `Data` object built without a benchmark:

```python
data = Data.from_returns(returns=df, benchmark=benchmark_df)  # <- required
```

### `ValueError: Index must be monotonically increasing`

Your date column is unsorted. Sort before construction:

```python
df = df.sort("Date")
```

### `ValueError: Index must contain at least two timestamps`

Statistics need at least two observations — single-row (or empty) input is
rejected at construction time rather than producing NaN everywhere.

### `ValueError: No overlapping dates between returns and benchmark`

Returns and benchmark are joined on dates (inner join). If the two frames
share no dates, there is nothing to compare. Check that both use the same
date column type (`pl.Date` vs `pl.Datetime` mismatches are a common cause).

### `UncleanSeriesError: series … contains null/non-finite values`

A derived portfolio series (e.g. `profit`) picked up NaN/inf. The usual
culprits are gaps, zeros, or negative values in the **prices** frame —
`pct_change` on a zero price produces infinity. Inspect
`portfolio.prices` for zero/negative entries.

### `ValueError: annual_breakdown requires a date column` (and friends)

Some features need a temporal `date` column (`pl.Date` or `pl.Datetime`):
calendar-based stats (`annual_breakdown`, `monthly_win_rate`), the monthly
heatmap, and the correlation heatmap. Integer-indexed data works everywhere
else.

---

## Questions

### What is the difference between NaN and null?

- **`null`** (Polars missing value) — "no observation". Rejected at
  construction unless you pass a `null_strategy`.
- **`NaN`** (IEEE-754 float) — "computed, but indeterminate". Propagates
  through statistics by design: a metric returns NaN when its value is
  mathematically undefined (zero variance, empty tail, …).

QuantStats (pandas) blurs this distinction; jquantstats keeps it. The
[migration guide](MIGRATION.md) has a full comparison.

### Can I pass pandas DataFrames?

Yes. `from_returns` / `from_prices` accept any
[narwhals-compatible](https://narwhals-dev.github.io/narwhals/) frame
(pandas, Polars, PyArrow, …) and convert it internally. Results are always
Polars.

### Why is my Sharpe ratio NaN?

Zero dispersion. Constant returns (including all-zero) have no meaningful
mean/std ratio, so jquantstats returns NaN instead of an absurdly large
number. The same guard applies to `trading_cost_impact`.

### Can `rf` be an integer? A Series?

`rf=0` (int) works — it is coerced to float. A time-varying risk-free rate
is passed as a two-column frame (date + rate); if the rate column isn't
named `"rf"` it is renamed internally with a warning.

### How do I migrate from QuantStats?

See the [migration guide](MIGRATION.md) — it maps every QuantStats function
to its jquantstats equivalent and lists the intentional behavioral
differences.

### Which imports are stable?

Only top-level imports (`from jquantstats import Portfolio, Data, CostModel`)
are covered by the semantic-versioning guarantee. Underscore-prefixed modules
are private. See [API Stability](STABILITY.md).
