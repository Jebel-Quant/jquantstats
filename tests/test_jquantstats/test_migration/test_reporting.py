"""Migration tests comparing reporting stats against quantstats."""

import numpy as np
import pytest
import quantstats as qs

from ..tolerances import TOL_FLOAT64, TOL_PINNED


def test_monthly_returns(stats, aapl):
    """monthly_returns values match quantstats for all years and months."""
    jqs_df = stats.monthly_returns(eoy=True, compounded=True)["AAPL"]
    qs_df = qs.stats.monthly_returns(aapl, eoy=True, compounded=True)

    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC", "EOY"]
    for col in months:
        jqs_vals = jqs_df.sort("year")[col].to_numpy()
        qs_vals = qs_df[col].to_numpy()
        np.testing.assert_allclose(jqs_vals, qs_vals, atol=TOL_FLOAT64, err_msg=f"Mismatch in {col}")


def test_monthly_returns_no_eoy(stats, aapl):
    """monthly_returns without eoy omits EOY and matches quantstats."""
    jqs_df = stats.monthly_returns(eoy=False, compounded=True)["AAPL"]
    qs_df = qs.stats.monthly_returns(aapl, eoy=False, compounded=True)

    assert "EOY" not in jqs_df.columns
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    for col in months:
        jqs_vals = jqs_df.sort("year")[col].to_numpy()
        qs_vals = qs_df[col].to_numpy()
        np.testing.assert_allclose(jqs_vals, qs_vals, atol=TOL_FLOAT64, err_msg=f"Mismatch in {col}")


@pytest.mark.parametrize("period", ["Daily", "Monthly", "Quarterly", "Yearly"])
def test_distribution(stats, aapl, period):
    """Distribution values and outliers match quantstats for non-weekly periods."""
    jqs_dist = stats.distribution(compounded=True)["AAPL"][period]
    qs_dist = qs.stats.distribution(aapl, compounded=True)[period]

    jqs_total = sorted(jqs_dist["values"] + jqs_dist["outliers"])
    qs_total = sorted(qs_dist["values"] + qs_dist["outliers"])

    assert len(jqs_total) == len(qs_total), f"{period}: count mismatch {len(jqs_total)} vs {len(qs_total)}"
    np.testing.assert_allclose(jqs_total, qs_total, atol=TOL_FLOAT64)


def test_compare_returns(stats, aapl, benchmark_pd):
    """Compare Returns column matches quantstats on non-null rows."""
    jqs_df = stats.compare()["AAPL"]
    qs_df = qs.stats.compare(aapl, benchmark_pd)

    jqs_vals = jqs_df["Returns"].drop_nulls().to_numpy()
    qs_vals = qs_df["Returns"].dropna().to_numpy()

    assert len(jqs_vals) == len(qs_vals)
    np.testing.assert_allclose(jqs_vals, qs_vals, atol=TOL_PINNED)


def test_compare_benchmark(stats, aapl, benchmark_pd):
    """Compare Benchmark column matches quantstats on non-null rows."""
    jqs_df = stats.compare()["AAPL"]
    qs_df = qs.stats.compare(aapl, benchmark_pd)

    jqs_vals = jqs_df["Benchmark"].drop_nulls().to_numpy()
    qs_vals = qs_df["Benchmark"].dropna().to_numpy()

    assert len(jqs_vals) == len(qs_vals)
    np.testing.assert_allclose(jqs_vals, qs_vals, atol=TOL_PINNED)


def test_implied_volatility_scalar(stats, aapl):
    """implied_volatility scalar (annualize=False) matches quantstats."""
    jqs_val = stats.implied_volatility(periods=252, annualize=False)["AAPL"]
    qs_val = float(qs.stats.implied_volatility(aapl, periods=252, annualize=False))
    assert jqs_val == pytest.approx(qs_val, abs=TOL_PINNED)
