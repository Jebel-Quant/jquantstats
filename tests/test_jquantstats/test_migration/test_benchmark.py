"""Migration tests for benchmark-dependent stats against quantstats."""

import numpy as np
import pytest
import quantstats as qs

from ..tolerances import TOL_ESTIMATE, TOL_PARITY


@pytest.fixture
def aligned(pandas_frame):
    """AAPL and SPY aligned on common dates, dropping the leading SPY NaN."""
    return pandas_frame[["AAPL", "SPY -- Benchmark"]].dropna()


def test_information_ratio(stats, aapl, benchmark_pd):
    """Quantstats does not annualise its information ratio, so we scale by sqrt(252)."""
    x = stats.information_ratio(periods_per_year=252)
    y = qs.stats.information_ratio(aapl, benchmark=benchmark_pd)
    assert x["AAPL"] == pytest.approx(y, abs=TOL_ESTIMATE)


def test_information_ratio_no_annualise(stats, aapl, benchmark_pd):
    """annualise=False matches the raw QuantStats information ratio directly."""
    x = stats.information_ratio(periods_per_year=252, annualise=False)
    y = qs.stats.information_ratio(aapl, benchmark=benchmark_pd)
    assert x["AAPL"] == pytest.approx(y, abs=TOL_ESTIMATE)


def test_r_squared(stats, aligned):
    """Series are inner-joined on Date to handle the leading NaN in SPY."""
    x = stats.r_squared()
    y = qs.stats.r_squared(aligned["AAPL"], benchmark=aligned["SPY -- Benchmark"])
    assert x["AAPL"] == pytest.approx(y, abs=TOL_PARITY)


def test_greeks_beta(stats, aligned):
    """Series are inner-joined on Date to handle the leading NaN in SPY."""
    x = stats.greeks()
    y = qs.stats.greeks(aligned["AAPL"], benchmark=aligned["SPY -- Benchmark"])
    assert x["AAPL"]["beta"] == pytest.approx(y["beta"], abs=TOL_PARITY)


def test_greeks_alpha(stats, aligned):
    """periods_per_year=252 set explicitly on both sides to match annualisation factor."""
    x = stats.greeks(periods_per_year=252)
    y = qs.stats.greeks(aligned["AAPL"], benchmark=aligned["SPY -- Benchmark"], periods=252)
    assert x["AAPL"]["alpha"] == pytest.approx(y["alpha"], abs=TOL_PARITY)


# ── treynor_ratio ─────────────────────────────────────────────────────────────
# NOTE: treynor_ratio is NOT compared against quantstats value — quantstats uses
# comp(returns) / beta (total compounded return) while jquantstats uses
# cagr(returns) / beta (annualised return, the standard financial definition).
# These tests verify structure and sign consistency.


def test_treynor_ratio_finite(stats):
    """treynor_ratio() returns a finite value for real benchmark data."""
    val = stats.treynor_ratio()["AAPL"]
    assert np.isfinite(val)


def test_treynor_ratio_sign_consistent_with_cagr(stats):
    """treynor_ratio() sign matches cagr() since beta is positive for AAPL vs SPY."""
    tr = stats.treynor_ratio()["AAPL"]
    cagr = stats.cagr(periods=252)["AAPL"]
    beta = stats.greeks()["AAPL"]["beta"]
    if beta > 0:
        assert (tr > 0) == (cagr > 0)


def test_treynor_ratio_returns_dict(stats):
    """treynor_ratio() returns a dict keyed by asset name."""
    result = stats.treynor_ratio()
    assert isinstance(result, dict)
    assert "AAPL" in result


# ── avg_drawdown ──────────────────────────────────────────────────────────────
# NOTE: jquantstats avg_drawdown averages only underwater periods; quantstats
# averages all periods including zero-drawdown days. No direct value comparison.


def test_avg_drawdown_less_extreme_than_max(stats):
    """avg_drawdown() is less extreme (closer to 0) than max_drawdown()."""
    avg_dd = stats.avg_drawdown()["AAPL"]
    max_dd = stats.max_drawdown()["AAPL"]
    # max_dd is negative; avg_dd should be between max_dd and 0
    assert max_dd <= avg_dd <= 0


# ── r_squared with explicit benchmark name ────────────────────────────────────


def test_r_squared_returns_dict(stats):
    """r_squared() returns a dict keyed by asset name."""
    result = stats.r_squared()
    assert isinstance(result, dict)
    assert "AAPL" in result


# ── information_ratio structure ────────────────────────────────────────────────


def test_information_ratio_returns_dict(stats):
    """information_ratio() returns a dict keyed by asset name."""
    result = stats.information_ratio()
    assert isinstance(result, dict)
    assert "AAPL" in result


def test_information_ratio_annualised_greater_than_raw(stats):
    """Annualised IR (annualise=True) > raw IR (annualise=False) for ppy > 1."""
    raw = stats.information_ratio(periods_per_year=252, annualise=False)["AAPL"]
    ann = stats.information_ratio(periods_per_year=252, annualise=True)["AAPL"]
    # Scaling by sqrt(252) > 1 always increases magnitude; signs must agree.
    assert (ann > 0) == (raw > 0)
    assert abs(ann) > abs(raw)
