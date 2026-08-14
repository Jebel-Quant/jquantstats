---
description: Build the shared jquantstats portfolio context from data files, or show the existing one
argument-hint: "[--prices data/prices.csv --cash-position data/pos.csv --aum 1e6] | show | list | check"
allowed-tools: Bash, Read, Glob, Grep
---

Manage the portfolio context that every jquantstats skill reads.

Arguments: `$ARGUMENTS`

Follow `${CLAUDE_PLUGIN_ROOT}/reference/portfolio-context.md`.

## With arguments

Pass them straight through, choosing the subcommand from what was given —
`portfolio` when a position file is named, `data` for a return or price series,
or the bare subcommand when the user typed `show`, `list`, `check` or
`activate`:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_load.py <subcommand> $ARGUMENTS
```

Then relay the digest, leading with any `WARNING` lines — those change which
answers are correct later, so they must not be buried.

## With no arguments

If `.jquantstats/context.json` exists, just show it:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_load.py show
```

Otherwise discover candidates rather than guessing. Look for price, position,
return and benchmark files (`*.csv`, `*.parquet` under `data/`, `input/`,
the repo root), read the header row of each to see its date column and asset
columns, then **propose** a concrete command and confirm before running it.

Two things must not be invented:

- **`--aum`** is unrecoverable from the data. Ask for it.
- **Which position flag applies** — `--cash-position` (currency),
  `--position` (share counts), `--risk-position` (risk units) — changes every
  number downstream. Ask if the file's units are not obvious from its
  magnitudes and column names.

If only returns or prices are available with no positions, build a `Data`
context instead, and mention that turnover, cost and lag analysis need
positions.
