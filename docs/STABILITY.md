---
icon: material/shield-check
---

# API Stability

This document defines the public API surface of **jquantstats** and the
stability guarantees that apply from **v1.0.0** onwards.

!!! info "Current status: 0.x — the guarantees below are not yet in force"

    jquantstats has not reached `v1.0.0`.  The surface described here is what
    *will* be frozen at 1.0, and is already treated as settled in practice, but
    the formal Semantic Versioning contract begins at that tag.  See
    [Before v1.0.0](#before-v100) for the policy that applies today.

## Stable public exports

The following names are exported from the top-level `jquantstats` package and
are covered by the stability guarantee described below.

| Name | Kind | Imported from |
|------|------|---------------|
| `Portfolio` | class | `jquantstats.portfolio` |
| `Data` | class | `jquantstats.data` |
| `Stats` | class | `jquantstats._stats` |
| `Plots` | class | `jquantstats._plots` |
| `NativeFrame` | type alias | `jquantstats._types` |
| `NativeFrameOrScalar` | type alias | `jquantstats._types` |

All of the above are importable directly from `jquantstats`:

```python
from jquantstats import Portfolio, Data
```

`Data`, `Stats`, and `Plots` instances are returned by the public API
(e.g. `Data.from_returns()` returns a `Data`; `data.stats` is a `Stats`;
`data.plots` is a `Plots`).  Their public methods and attributes are
stable even though the classes live in private modules.

## What "stable" means

From **v1.0.0** onwards jquantstats follows [Semantic Versioning](https://semver.org/):

- **Patch** (`1.x.y → 1.x.z`): bug fixes only; no API changes.
- **Minor** (`1.x → 1.y`): backwards-compatible additions (new methods,
  new keyword arguments with defaults, new exports).  Existing code
  continues to work unchanged.
- **Major** (`1.x → 2.0`): breaking changes are permitted.  A breaking
  change is any removal, rename, or signature change of a stable export
  or its public methods/attributes.

## What is *not* stable

Anything that is **not** in the table above is considered internal and
may change or be removed in any release:

- Private modules and subpackages: `_stats/`, `_plots/`, `_reports/`,
  `_utils/`, the `_portfolio_*.py` mixins, `_types.py`, `_protocol.py`,
  `_cost_model.py`, `_cache.py`, `_data_reshape.py`.
- Private classes, functions, or attributes whose names begin with an
  underscore (e.g. `Data._raw_returns`, `Stats._df`).
- Sub-module paths such as `jquantstats.portfolio`
  — import from the top-level package instead.

```python
# ✅ stable
from jquantstats import Portfolio, Data

# ❌ not stable — internal path, may change
from jquantstats.portfolio import Portfolio
```

## Deprecation policy

When a stable export needs to be changed or removed:

1. The old name / signature is kept for **one minor version** with a
   `DeprecationWarning` that names the replacement.
2. The breaking change is then made in the **next minor release** (or in
   the next major release if the migration is larger).

Example timeline:

| Release | Action |
|---------|--------|
| `1.3.0` | `old_name` deprecated; `DeprecationWarning` raised on use; `new_name` available |
| `1.4.0` | `old_name` removed |

## Before v1.0.0

Releases tagged `0.x.y` carry **no formal stability guarantee** — Semantic
Versioning, as described above, begins at `v1.0.0`.  That is the contractual
position.  In practice the project is more conservative than that, and the
paragraphs below describe what a caller can actually rely on today.

**What is already settled.**  The exports in the table above — `Portfolio`,
`Data`, `Stats`, `Plots` and the two type aliases — are not expected to be
renamed or removed before 1.0.  The constructors (`Portfolio.from_position`,
`from_cash_position`, `from_risk_position`, `Data.from_returns`) and the
`.stats` / `.plots` / `.report` accessors are likewise treated as fixed
points.  Changes here would be disruptive enough that they are held for the
1.0 boundary.

**What may still move.**  Individual metric methods on `Stats` may gain
keyword arguments, change default values, or be renamed for consistency as
the QuantStats-parity work settles; chart signatures on `Plots` may change
as the Plotly builders are refactored.  Anything private — see
[What is *not* stable](#what-is-not-stable) — may change in any release,
including the internal module layout, which has been reorganised more than
once during 0.x.

**How changes are communicated.**  Every release notes its changes in the
[changelog](changelog.md).  A minor bump (`0.10 → 0.11`) is where a breaking
change to a public name will appear; patch releases (`0.10.0 → 0.10.1`) are
bug fixes only.  Where a rename is avoidable, the old name is kept for one
minor version with a `DeprecationWarning`, following the same courtesy as the
post-1.0 [deprecation policy](#deprecation-policy) — but before 1.0 this is a
practice, not a promise.

**Pinning advice.**  Pin to a minor version (`jquantstats>=0.10,<0.11`) if you
need the surface to hold still; a bare `jquantstats>=0.10` may pick up a
breaking change at the next minor.  Once `v1.0.0` is tagged, the guarantees in
the sections above replace everything here.
