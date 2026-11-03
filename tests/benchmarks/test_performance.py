"""End-to-end performance benchmarks for computationally intensive methods.

Dataset: 5 assets × 10 years of daily returns (2,520 observations).
The fixture is session-scoped so construction cost is paid once.

Benchmark results (5 assets × 10 years daily; run `make benchmark` to refresh)
===============================================================================

Method                  | min (ms) | median (ms) | Notes
------------------------|----------|-------------|-------------------------------
summary()               |    22.26 |       22.57 | all scalar stats, 5 assets
rolling_sharpe()        |     0.46 |        0.49 | 126-day window, native Polars
rolling_volatility()    |     0.29 |        0.35 | 126-day window, native Polars
montecarlo() n=1000     |    12.21 |       12.58 | vectorised numpy, no Py loop
reports.full() html     |   342.04 |      343.95 | metrics + drawdowns + charts

Implementation notes
====================
* ``monthly_returns`` no longer uses ``map_elements``; month numbers are mapped
  to abbreviations via native Polars ``replace_strict`` (see _reporting.py).
* ``montecarlo`` / ``montecarlo_sharpe`` / ``montecarlo_drawdown`` /
  ``montecarlo_cagr`` were rewritten to generate all *n* block-bootstrap paths
  in a single vectorised numpy call, eliminating the ``for i in range(n)``
  Python loop that previously iterated once per simulation (see _montecarlo.py).
* ``rolling_sharpe`` and ``rolling_volatility`` already used native Polars
  ``rolling_mean`` / ``rolling_std`` expressions — no changes required.
* No ``map_elements`` or Python-level UDFs remain in any performance-critical
  path covered by these benchmarks.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from jquantstats import Data

# ── Dataset constants ─────────────────────────────────────────────────────────

DETERMINISTIC_SEED = 42
TRADING_DAYS_PER_YEAR = 252
N_YEARS = 10
N_ASSETS = 5
DAILY_MEAN_RETURN = 0.0003
DAILY_VOLATILITY = 0.012
N_ROWS = TRADING_DAYS_PER_YEAR * N_YEARS  # 2 520


# ── Dataset builder ───────────────────────────────────────────────────────────


def _build_multi_asset_data() -> Data:
    """Build a deterministic 5-asset × 10-year daily returns ``Data`` object."""
    rng = np.random.default_rng(DETERMINISTIC_SEED)
    start = dt.date(2015, 1, 2)
    dates: list[dt.date] = []
    d = start
    while len(dates) < N_ROWS:
        if d.weekday() < 5:  # Mon-Fri
            dates.append(d)
        d += dt.timedelta(days=1)

    asset_names = [f"A{i}" for i in range(1, N_ASSETS + 1)]
    returns_data: dict[str, list[float]] = {
        name: rng.normal(loc=DAILY_MEAN_RETURN, scale=DAILY_VOLATILITY, size=N_ROWS).tolist() for name in asset_names
    }
    returns_data["Date"] = dates  # type: ignore[assignment]
    df = pl.DataFrame(returns_data).select(["Date", *asset_names])
    return Data.from_returns(returns=df)


# ── Session-scoped fixture ────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def multi_asset_data() -> Data:
    """Deterministic 5-asset × 10-year daily ``Data`` fixture (session-scoped)."""
    return _build_multi_asset_data()


# ── Benchmark tests ───────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="summary")
def test_benchmark_summary(benchmark: pytest.BenchmarkFixture, multi_asset_data: Data) -> None:
    """Benchmark ``stats.summary()`` — exercises all scalar statistics.

    Covers: avg_return, avg_win, avg_loss, win_rate, profit_factor,
    payoff_ratio, monthly_win_rate, best, worst, volatility, sharpe, skew,
    kurtosis, value_at_risk, conditional_value_at_risk, max_drawdown,
    avg_drawdown, max_drawdown_duration, calmar, recovery_factor.
    """
    result = benchmark(multi_asset_data.stats.summary)
    assert result.shape[1] == N_ASSETS + 1  # metric col + one per asset


@pytest.mark.benchmark(group="rolling")
def test_benchmark_rolling_sharpe(benchmark: pytest.BenchmarkFixture, multi_asset_data: Data) -> None:
    """Benchmark ``stats.rolling_sharpe()`` — native Polars rolling expressions."""
    result = benchmark(multi_asset_data.stats.rolling_sharpe, rolling_period=126, periods_per_year=252)
    assert result.shape == (N_ROWS, N_ASSETS + 1)  # date col + asset cols


@pytest.mark.benchmark(group="rolling")
def test_benchmark_rolling_volatility(benchmark: pytest.BenchmarkFixture, multi_asset_data: Data) -> None:
    """Benchmark ``stats.rolling_volatility()`` — native Polars rolling expressions."""
    result = benchmark(multi_asset_data.stats.rolling_volatility, rolling_period=126, periods_per_year=252)
    assert result.shape == (N_ROWS, N_ASSETS + 1)


@pytest.mark.benchmark(group="montecarlo")
def test_benchmark_montecarlo(benchmark: pytest.BenchmarkFixture, multi_asset_data: Data) -> None:
    """Benchmark ``stats.montecarlo()`` — vectorised block-bootstrap simulation.

    Uses n=1000 (the default) and period=252 to mirror a realistic workload.
    """
    np.random.seed(DETERMINISTIC_SEED)
    result = benchmark(multi_asset_data.stats.montecarlo, n=1000, period=252)
    assert result.shape == (1000, N_ASSETS)


@pytest.mark.benchmark(group="html")
def test_benchmark_reports_full(benchmark: pytest.BenchmarkFixture, multi_asset_data: Data) -> None:
    """Benchmark ``reports.full()`` — full HTML report generation."""
    html = benchmark(multi_asset_data.reports.full)
    assert html.startswith("<!DOCTYPE html>")
    assert "Performance Metrics" in html
