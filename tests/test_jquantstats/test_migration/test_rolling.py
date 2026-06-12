"""Parametrized migration tests comparing rolling stats against quantstats."""

import numpy as np
import pytest
import quantstats as qs

from jquantstats._stats._performance import _RiskStatsMixin

from ..tolerances import TOL_FLOAT64


@pytest.mark.parametrize(
    ("method", "kwargs", "atol"),
    [
        ("rolling_sharpe", {"rolling_period": 126, "periods_per_year": 252}, 1e-6),
        ("rolling_volatility", {"rolling_period": 126, "periods_per_year": 252}, 1e-6),
        ("rolling_sortino", {"rolling_period": 126, "periods_per_year": 252}, 1e-6),
        ("compsum", {}, 1e-12),
        ("implied_volatility", {"periods": 252, "annualize": True}, 1e-6),
    ],
)
def test_rolling(stats, method, kwargs, atol):
    """Verify each rolling method produces the same time-series as quantstats."""
    aapl = stats.all.to_pandas().set_index("Date")["AAPL"]
    jqs_df = getattr(stats, method)(**kwargs)
    qs_series = getattr(qs.stats, method)(aapl, **kwargs)

    jqs_pd = jqs_df.to_pandas().set_index("Date")["AAPL"].dropna()
    qs_clean = qs_series.dropna()
    common = jqs_pd.index.intersection(qs_clean.index)

    assert len(common) > 0
    np.testing.assert_allclose(jqs_pd[common].values, qs_clean[common].values, atol=atol)


def test_pct_rank(stats):
    """Verify pct_rank matches quantstats on AAPL prices."""
    aapl_returns = stats.returns["AAPL"]
    aapl_prices = _RiskStatsMixin.prices(aapl_returns)
    aapl_prices_pd = aapl_prices.to_pandas()
    aapl_prices_pd.index = stats.index["Date"].to_pandas()

    jqs_df = stats.pct_rank(window=60)
    qs_series = qs.stats.pct_rank(aapl_prices_pd, window=60)

    jqs_pd = jqs_df.to_pandas().set_index("Date")["AAPL"].dropna()
    qs_clean = qs_series.dropna()
    common = jqs_pd.index.intersection(qs_clean.index)

    assert len(common) > 0
    np.testing.assert_allclose(jqs_pd[common].values, qs_clean[common].values, atol=TOL_FLOAT64)
