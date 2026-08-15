---
description: Full review of the active jquantstats portfolio — performance, risk, robustness, and what would break it
argument-hint: "[context name] [focus, e.g. costs | drawdowns | benchmark]"
allowed-tools: Bash, Read, Glob, Grep, Write
---

Review the portfolio in the shared context. Arguments: `$ARGUMENTS`

Treat any argument as a context name if it matches one from `jqs_load.py list`,
otherwise as a focus area to weight the review toward.

## Sequence

**1. Load and validate.** Follow
`${CLAUDE_PLUGIN_ROOT}/reference/portfolio-context.md`:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_load.py show
bash "${CLAUDE_PLUGIN_ROOT}/scripts/jqs.sh" jqs_load.py check
```

If the digest carries warnings or `check` reports `STALE`, resolve that first —
use the `portfolio-diagnostics` skill. Do not report metrics computed over a
known-broken sample; a plausible wrong number is worse than a delay.

**2. Performance and risk.** Use the `portfolio-analysis` skill. One Python
snippet, not one per metric: `summary()`, drawdown depth and duration, tail
measures, and — if the context has a benchmark — alpha, beta and capture.

**3. Robustness.** Use the `portfolio-robustness` skill. For a `Portfolio`
context always cover execution delay and cost sensitivity; both are cheap and
both kill more strategies than anything in step 2 reveals. Add autocorrelation
and sample-significance checks when the history is short.

**4. Charts.** Write figures to `_analysis/` and give the user the paths — a
`go.Figure` cannot render in a terminal. `pf.report.to_html(path=...)` for a full
document.

## Reporting

- Open with the two or three findings that would change a decision, not with a
  metric dump. The digest and the report already hold the full table.
- Quote the periodicity behind any annualised figure, and the sample length
  behind any Sharpe.
- Give breaking points where you have them: the bps at which the edge vanishes,
  the lag at which it dies, the percentile the observed drawdown occupies.
- State what you did **not** test. An unqualified clean bill of health from a
  short backtest is the least useful thing this review can produce.
