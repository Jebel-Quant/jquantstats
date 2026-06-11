"""Tests for comparing jquantstats with quantstats library functionality."""

import pytest

from ..tolerances import TOL_PARITY


def test_autocorrelation(stats, aapl):
    """Lag-1 autocorrelation matches pandas Series.autocorr(lag=1)."""
    x = stats.autocorr(lag=1)
    y = aapl.autocorr(lag=1)
    assert x["AAPL"] == pytest.approx(y, abs=TOL_PARITY)


def test_autocorrelation_lag5(stats, aapl):
    """Lag-5 autocorrelation matches pandas Series.autocorr(lag=5)."""
    x = stats.autocorr(lag=5)
    y = aapl.autocorr(lag=5)
    assert x["AAPL"] == pytest.approx(y, abs=TOL_PARITY)


def test_acf_matches_pandas(stats, aapl):
    """ACF values at each lag match pandas Series.autocorr()."""
    nlags = 10
    result = stats.acf(nlags=nlags)
    aapl_col = result["AAPL"].to_list()
    assert aapl_col[0] == pytest.approx(1.0)
    for k in range(1, nlags + 1):
        assert aapl_col[k] == pytest.approx(aapl.autocorr(lag=k), abs=TOL_PARITY)
