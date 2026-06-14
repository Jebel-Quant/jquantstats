# jquantstats vs. QuantStats — Engineering Comparison

This document compares the two libraries by their **underlying engineering**:
data model, architecture, API design, type safety, testing rigor, and
dependency footprint. It is a companion to [`difference.md`](difference.md),
which compares them by *features*.

All figures were measured against the versions installed in this repo's
environment: **jquantstats 0.9.6** and **QuantStats 0.0.81** (both pure-Python,
neither ships a compiled extension).

> **TL;DR** — QuantStats is a **procedural, pandas-centric** library: a handful
> of large modules of free functions that operate on (and monkey-patch onto)
> pandas Series. jquantstats is an **object-oriented, Polars-native** library:
> many small modules composed via mixins and protocols into frozen, immutable,
> memoised objects, behind a strict-typed, 100%-covered, mutation-tested quality
> gate. QuantStats optimises for *familiarity and reach*; jquantstats optimises
> for *correctness, type-safety, and maintainability*.

---

## At a glance

| Dimension | jquantstats 0.9.6 | QuantStats 0.0.81 |
|---|---|---|
| Data engine | Polars (+ narwhals boundary layer) | pandas + numpy |
| Paradigm | OO: frozen dataclasses, mixins, protocols | Procedural: module-level functions |
| Public API | Explicit `Portfolio`/`Data` objects + accessors | Free functions + `extend_pandas()` monkey-patch |
| State model | Immutable (frozen, slotted) + memoised | Stateless functions over mutable frames |
| Source size | ~10.6k LOC across ~34 files | ~12.3k LOC across ~12 files |
| Largest module | 1.4k LOC (`_plots/_data.py`) | 3.3k LOC (`stats.py`) |
| Runtime deps | 6 (no pandas) | 8 (incl. pandas, matplotlib, yfinance) |
| Type coverage | Full annotations, strict (`ty`), `py.typed` | Partial (~50% in `stats.py`), `py.typed` |
| Docstring coverage | 100% enforced (`interrogate`) | Good, not gated |
| Tests shipped | 15k LOC test suite in repo | Not shipped in wheel |
| Quality gates | 100% line+branch cov, mutation, property, snapshot | Lighter (pytest + coverage, no enforced gate) |
| Plotting engine | Plotly-native | Matplotlib core + optional `to_plotly()` converter |
| Python | 3.11+ | 3.10+ |

---

## 1. Data engine & numeric core

**QuantStats** is built directly on **pandas + numpy**. Every metric is a
function that takes a `pd.Series`/`pd.DataFrame` and returns a scalar or Series:

```python
def sharpe(returns, rf=0.0, periods=252, annualize=True, smart=False) -> float | _pd.Series:
    ...
```

It carries explicit compatibility shims — `_compat.py` (430 LOC) and
`_numpy_compat.py` (288 LOC) — to absorb breaking changes across pandas/numpy
versions. This is the cost of binding tightly to a fast-moving numeric stack.

**jquantstats** is **Polars-native** and computes via Polars/narwhals
expressions rather than pandas operations. `narwhals` (`import narwhals as nw`)
acts as a boundary adapter, so the library *accepts* pandas, Polars, or other
frames at its edges while keeping a **zero-pandas runtime** internally. There is
no per-version compatibility shim layer because Polars' API is more stable and
the narwhals abstraction isolates engine differences.

A concrete consequence is **null semantics**: jquantstats distinguishes Polars
`null` from IEEE-754 `NaN` and forces an explicit choice at ingestion
(`null_strategy={"raise","drop","forward_fill"}`), whereas QuantStats inherits
pandas' `NaN`-conflates-everything conventions.

---

## 2. Architecture & code organization

This is the sharpest engineering divide.

### QuantStats — flat & procedural
A handful of large, monolithic modules:

```
stats.py            3307 LOC   (85 free functions, 0 classes)
reports.py          2515 LOC
_plotting/core.py   2137 LOC
_plotting/wrappers  2114 LOC
utils.py            1002 LOC
```

There are **no classes in `stats.py`** — it is a flat namespace of functions.
This is easy to read, easy to grep, and easy to contribute a one-off metric to.
The flip side is large files, implicit coupling through shared helpers, and
behaviour distributed across long modules.

### jquantstats — composed & layered
Many small, single-responsibility modules assembled by composition:

```
src/jquantstats/
├── portfolio.py            Portfolio = 4 mixins
│   ├── _portfolio_nav.py          (NAV/returns chain)
│   ├── _portfolio_attribution.py  (tilt/timing)
│   ├── _portfolio_turnover.py     (turnover)
│   └── _portfolio_cost.py         (cost models)
├── data.py                 Data (returns route)
├── _stats/   (5 mixins: _basic, _performance, _reporting, _rolling, _montecarlo)
├── _plots/   (Plotly figures, protocol-segregated)
├── _reports/ (Jinja2 HTML)
├── _utils/, _cache.py, exceptions.py, result.py
└── *_protocol.py  (interface-segregation protocols)
```

Three patterns define its engineering:

- **Mixin composition.** `Portfolio` and `Stats` are each assembled from focused
  mixins. Every mixin declares the attributes it expects in `TYPE_CHECKING`
  blocks, so it type-checks standalone.
- **Protocol layering / interface segregation.** Subpackages depend on minimal
  `*Like` protocols (`_protocol.py`, `_plots/_protocol.py`, …) rather than on
  the concrete classes, which avoids circular imports without runtime glue.
- **Domain exceptions.** A dedicated `exceptions.py` (347 LOC) with named errors
  (`MissingDateColumnError`, `NullsInReturnsError`, `BenchmarkAlignmentWarning`,
  …) instead of bare `ValueError`s.

The trade-off: jquantstats has more files and more indirection. There is more to
learn before you can locate a calculation, but each unit is small, isolated, and
independently testable.

---

## 3. API design & state model

**QuantStats** exposes free functions and, via `extend_pandas()`, *monkey-patches
~100+ methods onto pandas itself*:

```python
qs.extend_pandas()
returns.sharpe()          # method now lives on pd.Series globally
qs.stats.sharpe(returns)  # or call the function directly
```

This is maximally discoverable for pandas users but mutates a third-party class
process-wide — a global side effect that can surprise other code in the same
interpreter. Functions are **stateless**: they recompute from the input frame on
every call, and the input frame is mutable.

**jquantstats** uses **explicit objects with namespaced accessors** and no global
mutation:

```python
pf = Portfolio.from_cash_position(prices=..., cash_position=..., aum=...)
pf.stats.sharpe()    # accessor namespaces: .stats / .plots / .report / .utils
```

The core objects are **frozen, slotted dataclasses** — immutable after
construction. Because frozen+slotted means neither `functools.cached_property`
(needs `__dict__`) nor attribute assignment works, the library ships a bespoke
`cached_in_slot` decorator (`_cache.py`) that memoises derived values into
explicitly-declared slot fields via `object.__setattr__`:

```python
@property
@cached_in_slot("_profits_cache")
def profits(self) -> pl.DataFrame: ...
```

So expensive derived quantities (NAV, drawdown, returns) are computed once and
cached, while the object stays immutable. The docstring is candid that this
cache is not thread-safe but is *correct* under races (every thread computes the
same deterministic value).

**Net:** QuantStats trades immutability for the convenience of patched pandas;
jquantstats trades convenience for immutability, memoisation, and zero global
state.

---

## 4. Type safety

Both ship a `py.typed` marker, but the depth differs.

- **QuantStats** has *improved* markedly: functions now carry signatures and
  rich docstrings (e.g. `sharpe(...) -> float | _pd.Series`), and `pyright` is a
  dev dependency. But annotation coverage is partial — in `stats.py`, **37 of 85
  functions** have return annotations and **~133 of 256 parameters** are typed.
  Typing is aspirational rather than enforced.

- **jquantstats** annotates fully (modern `str | None`, `Self`, `Literal`,
  `ParamSpec`), runs the strict `ty` type-checker in CI, and uses
  `from __future__ import annotations` with `TYPE_CHECKING` import blocks to keep
  runtime imports lean and break import cycles. Typing is a gate, not a goal.

Docstrings: jquantstats enforces **100% docstring coverage** via `interrogate`;
QuantStats has good docstrings but no coverage gate.

---

## 5. Testing & quality gates

This is where the engineering philosophies diverge most.

**QuantStats** does not ship its test suite in the wheel. Its dev tooling is
conventional: `pytest`, `pytest-cov`, `ruff`, `pyright`. There is no evidence of
an enforced coverage threshold, mutation testing, or property-based testing.

**jquantstats** treats verification as a first-class deliverable:

- **~15k LOC of tests** (larger than the ~10.6k LOC of source).
- **100% line *and* branch coverage**, enforced (`fail_under=100`) with a
  dedicated `test_coverage_gate.py`.
- **Mutation testing** (`mutmut`, gated in CI via `.github/workflows/mutation.yml`
  and a `mutation_gate.py`) — verifies the tests actually *catch* injected bugs,
  not just execute lines.
- **Property-based testing** (`hypothesis`) for numerical invariants.
- **Snapshot/regression testing** (`syrupy`).
- **Performance benchmarks** (`pytest-benchmark`, with a PR-benchmark workflow).
- **Validation against QuantStats itself** — QuantStats is a *test* dependency
  so jquantstats' metrics are checked against the reference implementation.

The repo also carries a richer CI surface (CodeQL, release automation, weekly
runs, a Makefile) reflecting a heavier engineering process.

---

## 6. Dependency footprint

**jquantstats runtime (6):** `jinja2`, `narwhals`, `numpy`, `plotly`, `polars`,
`scipy`. No pandas. Data fetching and web/static-export deps are *optional*
extras (`web`, `plot`).

**QuantStats runtime (8):** `matplotlib`, `numpy`, `pandas`, `python-dateutil`,
`scipy`, `seaborn`, `tabulate`, `yfinance`. The base install therefore pulls in
the full plotting stack *and* a network/market-data client (`yfinance`) whether
or not you use them.

Engineering implication: jquantstats keeps the *base* surface small and pushes
optional capability behind extras; QuantStats bundles batteries (plotting +
data) into the default install, which is more convenient but a larger and more
opinionated dependency graph.

---

## 7. Plotting & reporting internals

- **QuantStats** renders with **Matplotlib** at its core (`_plotting/core.py`),
  with `seaborn` as a dependency. A `wrappers.py` adds an optional `to_plotly(fig)`
  *converter* (behind the `plotly` extra) that re-wraps Matplotlib figures — so
  interactivity is bolted on, not native. Reports are assembled in `reports.py`.

- **jquantstats** builds **Plotly figures natively** (`_plots/`) and renders
  self-contained interactive HTML through **Jinja2 templates** (`_reports/`,
  hence the `jinja2` runtime dependency). Plot and report layers are
  protocol-segregated from the data layer.

---

## 8. Maturity & versioning

A fair point in QuantStats' favour: its `0.0.x` version string understates a
**mature, widely-deployed** codebase with years of real-world edge-case
hardening, a large user base, and extensive third-party documentation.
jquantstats (`0.9.6`) is younger; its engineering rigor is high but its
battle-testing in the wild is necessarily shallower.

---

## Summary

| If you value… | Lean toward |
|---|---|
| Familiar pandas idioms, monkey-patched convenience | QuantStats |
| Batteries-included install (plots + data fetch) | QuantStats |
| Maximum community/edge-case maturity | QuantStats |
| Immutability, memoisation, zero global state | jquantstats |
| Strict typing + enforced docstrings | jquantstats |
| 100% coverage, mutation & property testing | jquantstats |
| Polars-native performance, lean base deps | jquantstats |
| Small, single-responsibility, testable modules | jquantstats |

Both are pure-Python portfolio-analytics libraries with overlapping metric sets.
The engineering difference is one of *philosophy*: QuantStats is a pragmatic,
procedural toolkit that meets pandas users where they are; jquantstats is a
rigorously-engineered, type-safe, immutable, Polars-native rebuild that trades
some approachability for correctness guarantees and long-term maintainability.

---

*Measurements taken from the installed packages and this repository
(`pyproject.toml`, `src/jquantstats/`, `tests/`, `.github/workflows/`) and from
the QuantStats source on
[GitHub](https://github.com/ranaroussi/quantstats).*
