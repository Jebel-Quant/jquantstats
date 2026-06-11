---
icon: material/cash-multiple
---

# Cost Models

`CostModel` is the public API for specifying transaction costs in jQuantStats.
Import it directly from the top-level package:

```python
from jquantstats import CostModel, Portfolio
```

!!! info "Portfolio route only"
    Cost modelling requires the `Portfolio` entry point — it needs position
    data to compute turnover and per-unit costs.

---

## Overview

Every portfolio strategy incurs transaction costs. jQuantStats models these
costs through `CostModel`, a frozen dataclass that encapsulates **exactly one**
cost model at a time and enforces that the two models are never accidentally
combined.

| Model | Constructor | Scales with |
|-------|-------------|-------------|
| **Model A — per-unit** | `CostModel.per_unit(cost)` | Units of position change |
| **Model B — turnover-bps** | `CostModel.turnover_bps(bps)` | AUM turnover |
| **No cost** | `CostModel.zero()` | — |

---

## Model A: `per_unit` — position-delta cost

Use this model when your cost scales with the **number of units traded** — for
example, a fixed commission per share or a tick-size cost for futures.

```python
from jquantstats import CostModel, Portfolio

# £0.01 cost per share traded (one-way)
cost = CostModel.per_unit(0.01)

pf = Portfolio.from_cash_position(
    prices=prices,
    cash_position=positions,
    aum=1_000_000,
    cost_model=cost,
)

# NAV after deducting position-delta costs
print(pf.net_cost_nav)

# Inspect the raw per-day cost series
print(pf.position_delta_costs)
```

??? tip "When to use `per_unit`"
    - Equity portfolios where commissions are quoted per share.
    - Futures portfolios where tick-size friction dominates.
    - Any strategy where absolute position changes (not AUM fraction) drive costs.

---

## Model B: `turnover_bps` — AUM-proportional cost

Use this model when your cost scales with **notional turnover as a fraction of
AUM** — for example, a bid/ask spread expressed in basis points.

```python
from jquantstats import CostModel, Portfolio

# 5 bps one-way cost on AUM turnover
cost = CostModel.turnover_bps(5.0)

pf = Portfolio.from_cash_position(
    prices=prices,
    cash_position=positions,
    aum=1_000_000,
    cost_model=cost,
)

# Sweep Sharpe ratio across 0 → 20 bps in a single call
impact = pf.trading_cost_impact(max_bps=20)
print(impact)
```

??? tip "When to use `turnover_bps`"
    - Macro or fund-of-funds portfolios where trades are expressed as a fraction of AUM.
    - Strategies where the bid/ask spread (in bps) is the dominant cost driver.
    - Any scenario where you want to sweep cost assumptions and see impact on risk-adjusted returns.

---

## No transaction costs

```python
from jquantstats import CostModel

cost = CostModel.zero()
# Equivalent to the default — no cost_model argument
```

---

## Combining models is an error

`CostModel` enforces mutual exclusivity. Passing both a non-zero `per_unit`
and a non-zero `cost_bps` raises a `ValueError`:

```python
from jquantstats import CostModel

# This raises ValueError — only one model may be active at a time
CostModel(cost_per_unit=0.01, cost_bps=5.0)
```

!!! warning
    Always use the named constructors — `per_unit`, `turnover_bps`, `zero` —
    rather than the dataclass constructor directly. They make your intent
    explicit and guarantee mutual exclusivity.

---

## API Reference

::: jquantstats.CostModel

---

## Precedence: `cost_model` vs raw parameters

The Portfolio factories accept costs two ways — a `CostModel` object or the
raw `cost_per_unit` / `cost_bps` floats. **When `cost_model` is passed, it
wins**: its values replace whatever was given for `cost_per_unit`/`cost_bps`
in the same call. Pass one or the other, not both:

```python
# ❌ cost_per_unit=0.05 is silently replaced by the cost model's 0.01
pf = Portfolio.from_cash_position(..., cost_per_unit=0.05, cost_model=CostModel.per_unit(0.01))

# ✅ pick one channel
pf = Portfolio.from_cash_position(..., cost_model=CostModel.per_unit(0.01))
```

Note that a `CostModel` itself is exclusive: it carries *either* a per-unit
cost *or* a turnover-bps cost, never both.
