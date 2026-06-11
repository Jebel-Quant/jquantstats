# Examples

Five interactive [marimo](https://marimo.io) notebooks ship with the
repository under `book/marimo/notebooks/`. Each is rendered into the
documentation (links below) and can be run locally:

```bash
make marimo                       # open the notebook server
# or run one directly:
uv run marimo edit book/marimo/notebooks/analytics_demo.py
```

## [Analytics Demo](notebooks/analytics_demo.html)

The end-to-end tour: build a `Data` object from returns, compute the core
metric suite (Sharpe, Sortino, drawdown, win rates), and compare assets
against a benchmark. Start here if you're new to the library.

## [Portfolio Construction](notebooks/portfolio_construction.html)

The Portfolio route: turn prices and positions into a NAV curve via
`Portfolio.from_cash_position` and friends (`from_position`,
`from_risk_position`), then study execution-delay (`lag`) and smoothed
holdings. Shows the tilt/timing attribution decomposition.

## [Risk Metrics](notebooks/risk_metrics.html)

Deep dive on the risk suite: volatility, value-at-risk, conditional VaR,
ulcer index, drawdown analysis, and the rolling variants of each.

## [Plots and Reports](notebooks/plots_and_reports.html)

Every chart the `plots` facade produces (snapshot, heatmaps, rolling
metrics, histograms) and how to generate the self-contained HTML report
with `reports.full()` / `portfolio.report`.

## [Monte Carlo](notebooks/monte_carlo.html)

Resampled return paths: simulate alternative histories from observed
returns, then read distributional answers (path percentiles, Sharpe
dispersion, CAGR ranges) off the simulation.
