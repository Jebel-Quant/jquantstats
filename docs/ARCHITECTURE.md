# Architecture

A short tour of how jQuantStats is put together, for contributors.

## Two entry points, one analytics pipeline

```text
prices + positions ──► Portfolio ──► .data ──► Data ◄── returns series
                          │                     │
                          │                     ├── .stats    (Stats facade)
                          ├── .stats ───────────┤
                          ├── .plots  (PortfolioPlots)        ├── .plots   (DataPlots)
                          ├── .report (Report)                ├── .reports (Reports)
                          └── .utils  (PortfolioUtils)        └── .utils   (DataUtils)
```

- **`Portfolio`** (`portfolio.py`) starts from raw inputs — prices, cash
  positions, AUM — and compiles the NAV/returns chain. Use it when you know
  *what you held*.
- **`Data`** (`data.py`) starts from a returns series (plus optional
  benchmark and risk-free rate). Use it when you only know *what you made*.
- The bridge is one-directional: `portfolio.data` produces a `Data` object
  from the portfolio's daily returns, so every returns-series analytic is
  always available from a Portfolio. `Portfolio.plots`/`report` are
  deliberately *not* delegated — they exploit raw prices/positions that a
  bare returns series doesn't have.

## Mixin composition

Both core classes stay small by composing focused mixins:

- `Portfolio` = `PortfolioNavMixin` (NAV & returns chain) +
  `PortfolioAttributionMixin` (tilt/timing) + `PortfolioTurnoverMixin` +
  `PortfolioCostMixin`, declared in `_portfolio_*.py` modules.
- `Stats` (`_stats/_stats.py`) composes `_core`, `_basic`, `_performance`,
  `_reporting`, `_rolling`, and `_montecarlo` mixins — roughly: primitives,
  distribution/risk metrics, Sharpe-family metrics, summary/report tables,
  rolling windows, and Monte Carlo simulation.

Each mixin declares the attributes it expects from the composed class inside
an `if TYPE_CHECKING:` block, so it type-checks standalone without importing
the concrete class.

## Facades and lazy composition

`Data` and `Portfolio` expose analytics through lazy accessor properties
(`.stats`, `.plots`, `.reports`/`.report`, `.utils`) that construct a facade
object on first use. On `Portfolio` (a frozen, slotted dataclass) the results
are memoised in declared slot fields via `cached_in_slot` (`_cache.py`) —
`functools.cached_property` can't be used because slots leave no `__dict__`,
and plain assignment is blocked by `frozen=True`.

## Protocol layering

The analytics subpackages (`_stats`, `_plots`, `_reports`, `_utils`) never
import the concrete `Data`/`Portfolio` classes at runtime — that would be
circular, since those classes compose the subpackages. Instead:

- `_protocol.py` (root) defines the single shared `DataLike` and `StatsLike`
  structural protocols.
- Each subpackage defines its *own* minimal `PortfolioLike`
  (`_plots/_protocol.py`, `_reports/_protocol.py`, `_utils/_protocol.py`)
  listing only the members it consumes — interface segregation, kept
  deliberately un-merged so subpackages don't re-couple to the full
  Portfolio surface.

## Where things live

| Concern | Location |
|---|---|
| Public API surface | `__init__.py` (`Portfolio`, `Data`, `CostModel`, `Result`, `interpolate`) |
| Input validation & domain errors | `exceptions.py`, validated in `__post_init__`/factories |
| Cost models (per-unit vs turnover-bps) | `_cost_model.py`, applied in `_portfolio_cost.py` |
| HTML reports | `_reports/` + `templates/portfolio_report.html` |
| Web API (optional `[web]` extra) | `api/app.py` |
| Quality gates | 100% line+branch coverage, 100% docstring coverage, ruff, ty, mutation gate (`bin/mutation_gate.py`) |
