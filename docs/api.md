---
icon: material/code-tags
---

# API Reference

The public API surface of jquantstats. All stable exports are importable
directly from the top-level package:

```python
from jquantstats import Portfolio, Data, CostModel
```

| Export | What it is | Start here when… |
|--------|------------|------------------|
| [`Portfolio`](#portfolio) | Position-level analytics: NAV, turnover, costs, attribution, `lag()` | you have **prices and positions** |
| [`Data`](#data) | Returns-level analytics: 50+ stats, plots, HTML report | you have **returns** (or prices only) |
| [`CostModel`](#costmodel) | Declarative trading-cost specification (per-unit or turnover-bps) | you want cost-adjusted analytics |
| [`Result`](#result) | Lightweight container returned by some report helpers | you consume report internals |
| `NativeFrame`, `NativeFrameOrScalar` | Type aliases for narwhals-compatible input frames | you type-annotate your own code |

Exceptions live in [`jquantstats.exceptions`](#exceptions) and all inherit
from `JQuantStatsError`, so `except JQuantStatsError` catches the whole family.

See [API Stability](STABILITY.md) for the versioning and deprecation policy,
and the [FAQ](faq.md) for common errors.

---

## Portfolio

::: jquantstats.Portfolio

## Data

::: jquantstats.Data

## CostModel

::: jquantstats.CostModel

## Result

`Result` bundles a `Portfolio` with an optional per-asset expected-returns
frame (`mu`) and exports a standard artifact set in one call:
`create_reports(output_dir)` writes CSVs (prices, profit, returns, positions,
tilt/timing decomposition, and the `mu` signal when present) plus interactive
HTML plots. Use it when you want a reproducible on-disk report bundle from a
backtest or experiment; call `portfolio.plots` / `portfolio.report` directly
when you just want figures in a notebook. `mu` (when given) must be a Polars
DataFrame with one column per portfolio asset — anything else raises at
construction time.

::: jquantstats.Result

## Exceptions

::: jquantstats.exceptions
