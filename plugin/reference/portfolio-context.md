# The portfolio context

Shared protocol for every jquantstats skill. One portfolio, many ways of
reasoning about it — the skills differ, the object does not.

## Why a recipe rather than a live object

Each Bash call is a fresh Python process, so no `Portfolio` instance survives
between tool calls. What survives is a **recipe**: the paths and constructor
arguments needed to rebuild it. Rebuilding is not approximating — the same
inputs through the same recorded constructor give a bit-identical object, lag
and costs included.

Two files under `.jquantstats/`:

| File | Contents | Lifecycle |
|---|---|---|
| `context.json` | the recipe: entry point, constructor, input paths, args, post-transforms | authoritative, commit it |
| `digest.json` | derived facts: shape, nulls, anchor metrics, warnings, input fingerprints | regenerable, gitignore it |

## Getting the portfolio

Everything runs through the wrapper, which finds a Python that can import the
library (active venv → repo `.venv` → `python3` → `uv run --with jquantstats`):

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_load.py show
```

**If `.jquantstats/context.json` already exists**, that is the portfolio. Do not
ask the user to re-specify it, and do not build a different one. `show` rebuilds
it and reprints the digest.

**If it does not exist**, create it once:

```bash
# prices + positions: unlocks turnover, costs, execution delay, attribution
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_load.py portfolio \
    --prices data/prices.csv --cash-position data/pos.csv --aum 1e6 --cost-bps 5

# a return or price series: the QuantStats-equivalent route
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_load.py data \
    --returns data/returns.csv --benchmark data/spy.csv --null-strategy drop
```

Position input picks the constructor: `--cash-position` (currency),
`--position` (share counts), `--risk-position` (risk units, with `--vola`).
Never guess `--aum`: it is unrecoverable from the data, so ask.

## Using it in analysis

Every snippet starts the same way. Read the recipe, never re-derive it:

```python
import os, sys
sys.path.insert(0, os.path.join(os.environ["CLAUDE_PLUGIN_ROOT"], "scripts"))
from jqs_context import load

pf = load()              # the active context
pf = load("lag1")        # a named variant
```

## Variants

Comparisons are first-class here — lag sweeps, cost sweeps, sub-periods. Record
each as its own named context rather than juggling ad-hoc objects:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_load.py portfolio \
    --prices data/prices.csv --cash-position data/pos.csv --aum 1e6 \
    --lag 1 --name lag1
```

`jqs_load.py list` shows them all (`*` marks the active one); `activate <name>`
switches. Each keeps its own digest and its own fingerprints.

## Staleness

`digest.json` stores a SHA-256 of every input. Before quoting any number from
it, confirm it still describes the files on disk:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_load.py check
```

`STALE` means the inputs changed after the digest was written. Recompute; do not
quote the stored value. The digest's `anchors` block exists as a tripwire, not a
cache: if a fresh computation disagrees with it, something in the rebuild is
wrong and that is worth investigating before reporting anything.

## Recipe schema

Hand-editable, and the fields are not optional decoration:

```json
{
  "schema": 1,
  "active": "base",
  "contexts": {
    "base": {
      "name": "base",
      "entry_point": "Portfolio",
      "constructor": "from_cash_position",
      "inputs": {
        "prices":        { "path": "data/prices.csv", "date_col": "Date" },
        "cash_position": { "path": "data/pos.csv" }
      },
      "args": { "aum": 1000000.0, "cost_bps": 5.0 },
      "post": [{ "op": "lag", "n": 1 }]
    }
  }
}
```

- `entry_point` / `constructor` — `Portfolio` and `Data` expose different
  accessors, so which one is held changes every downstream call.
- `date_col` — omitted means "the first column".
- `args` — `aum` is required for Portfolio. Cost args change every return
  downstream, so a cost-free rebuild silently disagrees with earlier answers.
- `post` — the transform replay, in order (`lag`, `truncate`, `resample`). **The
  one people forget.** A `lag(1)` study rebuilt unlagged looks entirely
  plausible and is wrong.

An input can also name a repo-owned function instead of a file, for positions
that come from an expression, a query, or a model:

```json
"cash_position": { "script": "analysis/build.py", "callable": "positions" }
```

The function takes no arguments and returns a `pl.DataFrame`. Logic stays in a
reviewable `.py` file rather than as source embedded in JSON.

## Checking the API before you use it

The library's surface moves and its short aliases do not exist. Confirm a symbol
rather than recalling it:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_api.py --show sharpe
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_api.py --grep drawdown
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_api.py stats
```

`--show` on a missing name says so. That means it was renamed to a canonical
form or never ported — report that, do not write a wrapper to paper over it.
