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

::: jquantstats.Result

## Exceptions

::: jquantstats.exceptions
