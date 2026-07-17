# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## Project

**jquantstats** — portfolio analytics for quants, built on [Polars](https://pola.rs/)
(zero pandas at runtime) with interactive [Plotly](https://plotly.com/python/) charts.
Python `>=3.11`, MIT-licensed, published to PyPI. This is a
[Rhiza](https://github.com/jebel-quant/rhiza)-managed repository (template `v1.2.0`).

## Commands

Prefer bare `make <target>`. **Never call `.venv/bin/...` directly** — the Makefile
resolves the environment for you.

| Target | Purpose |
|---|---|
| `make install` | Create/refresh the uv environment and install deps |
| `make fmt` | Format and lint-fix with ruff |
| `make test` | Run pytest with coverage (fails under the coverage threshold) |
| `make typecheck` | Static type checking |
| `make docs-coverage` | Docstring coverage report (interrogate) |
| `make deptry` | Detect unused / missing / misplaced dependencies |
| `make security` | Security scan (bandit / semgrep) |
| `make book` | Build the MkDocs documentation site |
| `make serve` | Serve the docs locally |
| `make marimo` | Run/validate the marimo notebooks |
| `make benchmark` | Run the QuantStats-parity benchmark |
| `make mutation` | Run mutation testing |
| `make clean` | Remove build/test artifacts |

Run `make` (or `make help` if available) to list all targets. The Makefile is
repo-owned (see its header) and can be edited without breaking template sync; it
includes the template-managed API via `.rhiza/rhiza.mk`.

## Architecture

Source lives under `src/jquantstats/`. Two public entry points, both exposing the
same `.stats`, `.plots`, and `.report` accessors:

- **`Portfolio`** (`portfolio.py`) — the primary route. Built from prices + positions
  (`from_cash_position`, `from_position`, `from_risk_position`); compiles NAV and
  unlocks position-level analysis (execution-delay via `lag(n)`, tilt/timing
  attribution, turnover, cost models) that a return series can't support.
- **`Data`** (`data.py`) — the returns route (`from_returns`), for arbitrary return streams.

Internal layering (leading-underscore modules are private):

- `_portfolio_*.py` — Portfolio internals split by concern: `_base`, `_constructors`,
  `_nav`, `_cost`, `_turnover`, `_transform`, `_attribution`.
- `_cost_model.py` — the two independent cost models (per-unit, turnover-bps).
- `_stats/` — metric mixins (`_basic`, `_basic_core`, `_performance`, `_reporting`,
  `_montecarlo`).
- `_plots/` — Plotly chart builders (portfolio + data variants).
- `_reports/` — HTML report rendering (`_html.py`) via Jinja2 `templates/`.
- `_utils/` — data helpers. `_protocol.py`, `_types.py`, `exceptions.py`,
  `result.py`, `_cache.py` — shared plumbing.

An optional FastAPI service lives in `api/` (installed via the `[web]` extra).

## Rhiza template split

This repo mixes **locally-owned** source with files **synced from
`jebel-quant/rhiza`**. The synced set is enumerated in `.rhiza/template.lock`
(`files:` block) and includes `.github/workflows/rhiza_*.yml`, `.rhiza/`,
`docs/mkdocs-base.yml`, and other tooling config.

**Rule:** gaps or bugs in Rhiza-managed files are fixed **upstream** in the template
and pulled in via a sync — do **not** patch them locally, or the next sync will
revert your change. Everything below is locally owned and edited here:

- `src/`, `tests/`, `api/`
- `pyproject.toml`, `README.md`, `CLAUDE.md`, `Makefile`, `mkdocs.yml`
- `docs/*.md` project documentation (but **not** `docs/mkdocs-base.yml`)

## Conventions

- **Coverage threshold:** `COVERAGE_FAIL_UNDER = 100` (`.rhiza/make.d/custom-env.mk`) —
  the suite must keep 100% line coverage.
- **Docstrings:** Google style; interrogate is expected to report 100% (`make docs-coverage`).
- **Tests** live under `tests/`; `test_quantstats.py` validates metrics against the
  `quantstats` reference implementation (a dev-only dependency).
- Fully type-annotated public API (`py.typed`).
