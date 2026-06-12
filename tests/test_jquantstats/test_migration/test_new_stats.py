"""Migration tests for functions added after the initial quantstats port."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
import quantstats as qs

from jquantstats import Data

from ..tolerances import TOL_ESTIMATE, TOL_FLOAT64, TOL_PARITY, TOL_PINNED


@pytest.fixture
def aligned(pandas_frame):
    """AAPL and SPY aligned on common dates."""
    return pandas_frame[["AAPL", "SPY -- Benchmark"]].dropna()


# ── r_squared ─────────────────────────────────────────────────────────────────


def test_r_squared(stats, aligned):
    """r_squared() matches quantstats."""
    x = stats.r_squared()
    y = qs.stats.r2(aligned["AAPL"], aligned["SPY -- Benchmark"])
    assert x["AAPL"] == pytest.approx(y, abs=TOL_PARITY)


# ── outliers / remove_outliers ────────────────────────────────────────────────


def test_outliers_count(stats, aapl):
    """outliers() returns the same values as quantstats."""
    jqs = stats.outliers()["AAPL"]
    qs_vals = qs.stats.outliers(aapl, quantile=0.95)
    assert len(jqs) == len(qs_vals)


def test_outliers_values(stats, aapl):
    """outliers() values match quantstats."""
    jqs = stats.outliers()["AAPL"].sort().to_list()
    qs_vals = sorted(qs.stats.outliers(aapl, quantile=0.95).tolist())
    np.testing.assert_allclose(jqs, qs_vals, atol=TOL_FLOAT64)


def test_remove_outliers_count(stats, aapl):
    """remove_outliers() returns the same number of values as quantstats."""
    jqs = stats.remove_outliers()["AAPL"]
    qs_vals = qs.stats.remove_outliers(aapl, quantile=0.95)
    assert len(jqs) == len(qs_vals)


def test_remove_outliers_values(stats, aapl):
    """remove_outliers() values match quantstats."""
    jqs = stats.remove_outliers()["AAPL"].sort().to_list()
    qs_vals = sorted(qs.stats.remove_outliers(aapl, quantile=0.95).tolist())
    np.testing.assert_allclose(jqs, qs_vals, atol=TOL_FLOAT64)


# ── expected_return ───────────────────────────────────────────────────────────


def test_expected_return_no_aggregate(stats, aapl):
    """expected_return() without aggregate matches quantstats geometric mean."""
    jqs = stats.expected_return()["AAPL"]
    qs_val = qs.stats.expected_return(aapl)
    assert jqs == pytest.approx(qs_val, abs=TOL_FLOAT64)


# ── drawdown_details ──────────────────────────────────────────────────────────
#
# qs.stats.drawdown_details takes a pre-computed drawdown series and includes a
# 1-period "drawdown" on the very first day of data when the initial return is
# negative (because the drawdown series is non-zero there).  jquantstats uses
# `nav < hwm` (strict) which is always False on day 1, so that 1-day stub is
# absent.  Tests below strip that row from the qs result before comparing.


def _qs_details(aapl):
    """Return qs drawdown details, dropping any 1-day stub on the first data date."""
    df = qs.stats.drawdown_details(qs.stats.to_drawdown_series(aapl))
    first_date = str(aapl.index[0].date())
    return df[df["start"] != first_date].reset_index(drop=True)


def test_drawdown_details_period_count(stats, aapl):
    """drawdown_details() finds the same number of drawdown periods as quantstats."""
    jqs_df = stats.drawdown_details()["AAPL"]
    qs_df = _qs_details(aapl)
    assert len(jqs_df) == len(qs_df)


def test_drawdown_details_start_dates(stats, aapl):
    """drawdown_details() start dates match quantstats."""
    import polars as pl

    jqs_df = stats.drawdown_details()["AAPL"]
    qs_df = _qs_details(aapl)

    jqs_starts = jqs_df.sort("start")["start"].cast(pl.String).to_list()
    qs_starts = sorted(qs_df["start"].tolist())
    assert jqs_starts == qs_starts


def test_drawdown_details_max_drawdown(stats, aapl):
    """drawdown_details() max drawdown values match quantstats (within 1e-4)."""
    jqs_df = stats.drawdown_details()["AAPL"].sort("start")
    qs_df = _qs_details(aapl).sort_values("start")

    # jqs returns fraction; qs returns percentage — multiply jqs by 100
    jqs_vals = (jqs_df["max_drawdown"] * 100).to_list()
    qs_vals = qs_df["max drawdown"].tolist()
    np.testing.assert_allclose(jqs_vals, qs_vals, atol=TOL_ESTIMATE)


# ── rolling_greeks ────────────────────────────────────────────────────────────


def test_rolling_greeks_beta(stats, aapl, benchmark_pd):
    """rolling_greeks() beta matches quantstats on overlapping dates."""
    jqs_df = stats.rolling_greeks(rolling_period=126, periods_per_year=252)
    qs_df = qs.stats.rolling_greeks(aapl, benchmark_pd, periods=126)

    jqs_pd = jqs_df.to_pandas().set_index("Date")["AAPL_beta"].dropna()
    qs_beta = qs_df["beta"].dropna()
    common = jqs_pd.index.intersection(qs_beta.index)

    assert len(common) > 100
    np.testing.assert_allclose(jqs_pd[common].values, qs_beta[common].values, atol=TOL_ESTIMATE)


# NOTE: rolling_greeks alpha is intentionally NOT compared against quantstats.
# qs.stats.rolling_greeks computes alpha as:
#   global_mean(returns) - rolling_beta * global_mean(benchmark)
# using the full-series mean instead of the rolling-window mean.  This is
# a known defect in quantstats.  jquantstats uses the correct OLS intercept:
#   rolling_mean(returns) - rolling_beta * rolling_mean(benchmark)


# ── treynor_ratio ─────────────────────────────────────────────────────────────


# NOTE: treynor_ratio is intentionally NOT compared against quantstats.
# qs.stats.treynor_ratio computes: comp(returns) / beta   (total compounded return)
# jquantstats computes:           cagr(returns) / beta    (annualised return)
# The jquantstats definition is the standard financial one.


# ── treynor_ratio property tests ──────────────────────────────────────────────


def test_treynor_ratio_positive_for_aapl(stats):
    """treynor_ratio() is positive for AAPL (positive CAGR, positive beta)."""
    val = stats.treynor_ratio()["AAPL"]
    assert np.isfinite(val)
    assert val > 0


def test_treynor_ratio_returns_dict(stats):
    """treynor_ratio() returns a dict keyed by asset name."""
    result = stats.treynor_ratio()
    assert isinstance(result, dict)
    assert "AAPL" in result


# ── hhi_positive / hhi_negative ───────────────────────────────────────────────
# These are jquantstats-specific; quantstats has no equivalent.
# Tests verify mathematical properties of the Herfindahl-Hirschman Index.


def test_hhi_positive_in_range(stats):
    """hhi_positive() returns a value in [0, 1] for real data."""
    val = stats.hhi_positive()["AAPL"]
    assert 0 <= val <= 1


def test_hhi_negative_in_range(stats):
    """hhi_negative() returns a value in [0, 1] for real data."""
    val = stats.hhi_negative()["AAPL"]
    assert 0 <= val <= 1


def test_hhi_positive_near_zero_for_uniform_returns():
    """hhi_positive() is 0 when all positive returns are equal (perfect spread)."""
    n = 200
    returns = pl.DataFrame({"Date": [date(2020, 1, 1) + timedelta(days=i) for i in range(n)], "asset": [0.01] * n})
    val = Data.from_returns(returns=returns).stats.hhi_positive()["asset"]
    assert val == pytest.approx(0.0, abs=TOL_PINNED)


def test_hhi_negative_nan_fewer_than_three():
    """hhi_negative() returns NaN when fewer than 3 negative returns are present."""
    vals = [0.01, 0.02, -0.01, 0.01, 0.02, 0.01, 0.03, 0.01, 0.02, 0.01]
    returns = pl.DataFrame({"Date": [date(2020, 1, 1) + timedelta(days=i) for i in range(10)], "asset": vals})
    val = Data.from_returns(returns=returns).stats.hhi_negative()["asset"]
    assert np.isnan(val)


def test_hhi_returns_dict(stats):
    """hhi_positive() and hhi_negative() return dicts keyed by asset name."""
    assert isinstance(stats.hhi_positive(), dict)
    assert isinstance(stats.hhi_negative(), dict)
    assert "AAPL" in stats.hhi_positive()
    assert "AAPL" in stats.hhi_negative()


# ── sharpe_variance ───────────────────────────────────────────────────────────
# sharpe_variance is not in quantstats; tests verify mathematical properties.


def test_sharpe_variance_positive(stats):
    """sharpe_variance() returns a positive value for real data."""
    val = stats.sharpe_variance()["AAPL"]
    assert val > 0


def test_sharpe_variance_finite(stats):
    """sharpe_variance() returns a finite value."""
    val = stats.sharpe_variance()["AAPL"]
    assert np.isfinite(val)


def test_sharpe_variance_decreases_with_sample_size():
    """Longer return series → smaller Sharpe ratio variance (more data = tighter estimate)."""
    rng = np.random.default_rng(0)
    base_vals = (rng.standard_normal(50) * 0.01 + 0.001).tolist()
    extended_vals = base_vals + (rng.standard_normal(450) * 0.01 + 0.001).tolist()

    def _make(vals):
        n = len(vals)
        return pl.DataFrame({"Date": [date(2020, 1, 1) + timedelta(days=i) for i in range(n)], "asset": vals})

    var_short = Data.from_returns(returns=_make(base_vals)).stats.sharpe_variance()["asset"]
    var_long = Data.from_returns(returns=_make(extended_vals)).stats.sharpe_variance()["asset"]
    assert var_long < var_short


def test_sharpe_variance_returns_dict(stats):
    """sharpe_variance() returns a dict keyed by asset name."""
    result = stats.sharpe_variance()
    assert isinstance(result, dict)
    assert "AAPL" in result


# ── probabilistic_ratio generic ───────────────────────────────────────────────
# The generic probabilistic_ratio() evaluates variance at the observed ratio,
# while the specialized methods (probabilistic_sharpe_ratio etc.) evaluate
# variance at benchmark SR = 0.  The formulas differ; no cross-comparison.


def test_probabilistic_ratio_sharpe_in_range(stats):
    """probabilistic_ratio('sharpe') returns a value in [0, 1]."""
    val = stats.probabilistic_ratio("sharpe")["AAPL"]
    assert 0 <= val <= 1


def test_probabilistic_ratio_sortino_in_range(stats):
    """probabilistic_ratio('sortino') returns a value in [0, 1]."""
    val = stats.probabilistic_ratio("sortino")["AAPL"]
    assert 0 <= val <= 1


def test_probabilistic_ratio_adjusted_sortino_in_range(stats):
    """probabilistic_ratio('adjusted_sortino') returns a value in [0, 1]."""
    val = stats.probabilistic_ratio("adjusted_sortino")["AAPL"]
    assert 0 <= val <= 1


def test_probabilistic_ratio_invalid_base_raises(stats):
    """probabilistic_ratio() raises ValueError for unknown base strings."""
    with pytest.raises(ValueError, match="base must be one of"):
        stats.probabilistic_ratio("unknown_metric")


def test_probabilistic_ratio_custom_callable(stats):
    """probabilistic_ratio() accepts a custom per-series callable."""
    result = stats.probabilistic_ratio(lambda s: float(s.mean() / (s.std(ddof=1) or 1.0)))
    assert isinstance(result, dict)
    assert "AAPL" in result
    assert 0 <= result["AAPL"] <= 1


def test_probabilistic_ratio_returns_dict(stats):
    """probabilistic_ratio() returns a dict with all asset keys."""
    result = stats.probabilistic_ratio("sharpe")
    assert isinstance(result, dict)
    assert len(result) == len(stats.assets)


# ── annual_breakdown ──────────────────────────────────────────────────────────
# annual_breakdown is not in quantstats; tests verify structure and consistency.


def test_annual_breakdown_returns_dataframe(stats):
    """annual_breakdown() returns a Polars DataFrame."""
    assert isinstance(stats.annual_breakdown(), pl.DataFrame)


def test_annual_breakdown_has_required_columns(stats):
    """annual_breakdown() has 'year', 'metric', and asset columns."""
    result = stats.annual_breakdown()
    assert "year" in result.columns
    assert "metric" in result.columns
    assert "AAPL" in result.columns


def test_annual_breakdown_covers_multiple_years(stats):
    """annual_breakdown() spans at least 2 calendar years of real data."""
    years = stats.annual_breakdown()["year"].unique().to_list()
    assert len(years) >= 2


def test_annual_breakdown_sharpe_finite_per_year(stats):
    """Sharpe metric rows in annual_breakdown() contain only finite values."""
    result = stats.annual_breakdown()
    sharpe_rows = result.filter(pl.col("metric") == "sharpe")
    for val in sharpe_rows["AAPL"].drop_nulls().to_list():
        assert np.isfinite(val)


def test_annual_breakdown_metrics_match_summary_set(stats):
    """annual_breakdown() metric names are a subset of summary() metric names."""
    summary_metrics = set(stats.summary()["metric"].to_list())
    breakdown_metrics = set(stats.annual_breakdown()["metric"].to_list())
    assert breakdown_metrics <= summary_metrics


# ── expected_return with aggregate ───────────────────────────────────────────
# Tests properties of the aggregated expected return; quantstats has the same
# aggregate parameter but calendar grouping may differ, so no value comparison.


@pytest.mark.parametrize("aggregate", ["weekly", "monthly", "quarterly", "annual", "yearly"])
def test_expected_return_aggregate_finite(stats, aggregate):
    """expected_return(aggregate=...) returns a finite value for each frequency."""
    val = stats.expected_return(aggregate=aggregate)["AAPL"]
    assert np.isfinite(val)


def test_expected_return_yearly_equals_annual(stats):
    """'yearly' is an alias for 'annual' in expected_return()."""
    annual = stats.expected_return(aggregate="annual")["AAPL"]
    yearly = stats.expected_return(aggregate="yearly")["AAPL"]
    assert annual == pytest.approx(yearly, abs=TOL_FLOAT64)


def test_expected_return_invalid_aggregate_raises(stats):
    """expected_return() raises ValueError for unrecognised aggregate strings."""
    with pytest.raises(ValueError, match="aggregate must be one of"):
        stats.expected_return(aggregate="decennial")


# ── geometric_mean annualized ─────────────────────────────────────────────────


def test_geometric_mean_annualized_finite(stats):
    """geometric_mean(annualize=True) returns a finite value."""
    val = stats.geometric_mean(annualize=True)["AAPL"]
    assert np.isfinite(val)


def test_geometric_mean_annualized_positive_for_aapl(stats):
    """geometric_mean(annualize=True) is positive for AAPL (net positive cumulative return)."""
    val = stats.geometric_mean(annualize=True)["AAPL"]
    assert val > 0


def test_geometric_mean_annualized_greater_than_per_period(stats):
    """Annualised geometric mean exceeds per-period geometric mean for positive returns."""
    per_period = stats.geometric_mean()["AAPL"]
    annualized = stats.geometric_mean(annualize=True)["AAPL"]
    if per_period > 0:
        assert annualized > per_period


# ── periods_per_year property ─────────────────────────────────────────────────
# Not in quantstats; verified as a positive number and approximately 252 for daily data.


def test_periods_per_year_positive(stats):
    """periods_per_year is a positive number."""
    val = stats.periods_per_year
    assert val > 0


def test_periods_per_year_daily_approx_252(stats):
    """periods_per_year is in [200, 280] for daily AAPL data."""
    val = stats.periods_per_year
    assert 200 < val < 280


# ── worst_n_periods ───────────────────────────────────────────────────────────
# jquantstats extension; quantstats' worst(n=1) returns a scalar, not a list.


def test_worst_n_periods_length(stats):
    """worst_n_periods(n=5) returns exactly 5 values."""
    assert len(stats.worst_n_periods(n=5)["AAPL"]) == 5


def test_worst_n_periods_sorted_ascending(stats):
    """worst_n_periods() values are sorted ascending (worst first)."""
    worst = stats.worst_n_periods(n=5)["AAPL"]
    assert worst == sorted(worst)


def test_worst_n_periods_matches_bottom_n(stats):
    """worst_n_periods(n) matches the n smallest non-null returns."""
    n = 5
    worst_jqs = stats.worst_n_periods(n=n)["AAPL"]
    expected = stats.returns["AAPL"].drop_nulls().sort().head(n).to_list()
    np.testing.assert_allclose(worst_jqs, expected, atol=TOL_FLOAT64)


def test_worst_n_periods_returns_dict(stats):
    """worst_n_periods() returns a dict keyed by asset name."""
    result = stats.worst_n_periods(n=3)
    assert isinstance(result, dict)
    assert "AAPL" in result


# ── drawdown_details additional column coverage ───────────────────────────────
# start-date and max_drawdown comparisons are in the tests above;
# these tests cover valley, duration, and recovery_duration.


def test_drawdown_details_has_all_columns(stats):
    """drawdown_details() DataFrame has start, valley, end, duration, max_drawdown, recovery_duration."""
    details = stats.drawdown_details()["AAPL"]
    expected_cols = {"start", "valley", "end", "duration", "max_drawdown", "recovery_duration"}
    assert expected_cols <= set(details.columns)


def test_drawdown_details_duration_positive(stats):
    """Duration column is positive for all recorded drawdown episodes."""
    details = stats.drawdown_details()["AAPL"]
    assert (details["duration"] > 0).all()


def test_drawdown_details_valley_on_or_after_start(stats):
    """Valley date is on or after the episode start date."""
    details = stats.drawdown_details()["AAPL"]
    for row in details.iter_rows(named=True):
        assert row["valley"] >= row["start"]


def test_drawdown_details_end_on_or_after_valley(stats):
    """End date (when non-null) is on or after the valley date."""
    details = stats.drawdown_details()["AAPL"].drop_nulls(subset=["end"])
    for row in details.iter_rows(named=True):
        assert row["end"] >= row["valley"]


def test_drawdown_details_recovery_duration_non_negative(stats):
    """recovery_duration (when non-null) is non-negative."""
    details = stats.drawdown_details()["AAPL"].drop_nulls(subset=["recovery_duration"])
    assert (details["recovery_duration"] >= 0).all()


def test_drawdown_details_max_drawdown_negative(stats):
    """max_drawdown column values are all non-positive (drawdown expressed as negative fraction)."""
    details = stats.drawdown_details()["AAPL"]
    assert (details["max_drawdown"] <= 0).all()


# ── avg_drawdown ──────────────────────────────────────────────────────────────
# NOTE: jquantstats avg_drawdown averages only underwater periods, while
# quantstats would average all periods including zero-drawdown days.
# The jquantstats definition is a more meaningful severity measure; no direct
# quantstats comparison.


def test_avg_drawdown_non_positive(stats):
    """avg_drawdown() is non-positive (zero or a negative fraction)."""
    val = stats.avg_drawdown()["AAPL"]
    assert val <= 0


def test_avg_drawdown_at_most_max_drawdown(stats):
    """avg_drawdown() magnitude is less than or equal to max_drawdown() magnitude."""
    avg_dd = stats.avg_drawdown()["AAPL"]
    max_dd = stats.max_drawdown()["AAPL"]  # already negative
    assert max_dd <= avg_dd <= 0


def test_avg_drawdown_returns_dict(stats):
    """avg_drawdown() returns a dict keyed by asset name."""
    result = stats.avg_drawdown()
    assert isinstance(result, dict)
    assert "AAPL" in result


# ── max_drawdown_duration ─────────────────────────────────────────────────────
# Not in quantstats; tests verify type and that real data produces a positive value.


def test_max_drawdown_duration_non_negative_int(stats):
    """max_drawdown_duration() returns a non-negative integer for each asset."""
    result = stats.max_drawdown_duration()
    val = result["AAPL"]
    assert isinstance(val, int)
    assert val >= 0


def test_max_drawdown_duration_positive_for_real_data(stats):
    """max_drawdown_duration() is positive for real-world data with drawdowns."""
    assert stats.max_drawdown_duration()["AAPL"] > 0


def test_max_drawdown_duration_returns_dict(stats):
    """max_drawdown_duration() returns a dict keyed by asset name."""
    result = stats.max_drawdown_duration()
    assert isinstance(result, dict)
    assert "AAPL" in result


# ── monthly_win_rate ──────────────────────────────────────────────────────────
# Not in quantstats; tests verify the value is in [0, 1] and is consistent with
# monthly_returns.


def test_monthly_win_rate_in_range(stats):
    """monthly_win_rate() returns a value in [0, 1]."""
    val = stats.monthly_win_rate()["AAPL"]
    assert 0 <= val <= 1


def test_monthly_win_rate_non_trivial(stats):
    """monthly_win_rate() is strictly between 0 and 1 for real multi-year data."""
    val = stats.monthly_win_rate()["AAPL"]
    assert 0 < val < 1


def test_monthly_win_rate_returns_dict(stats):
    """monthly_win_rate() returns a dict keyed by asset name."""
    result = stats.monthly_win_rate()
    assert isinstance(result, dict)
    assert "AAPL" in result


# ── up_capture / down_capture ─────────────────────────────────────────────────
# NOTE: jquantstats uses per-period geometric mean capture (strat_geom / bench_geom).
# quantstats uses annualised CAGR-based capture.  The two formulas are not
# equivalent, so no direct quantstats comparison is made here.


def test_up_capture_finite(stats):
    """up_capture() returns a finite value for real data."""
    bench = stats.all["SPY -- Benchmark"]
    val = stats.up_capture(bench)["AAPL"]
    assert np.isfinite(val)


def test_up_capture_positive(stats):
    """up_capture() is positive (geometric mean of positive-benchmark periods > 0)."""
    bench = stats.all["SPY -- Benchmark"]
    val = stats.up_capture(bench)["AAPL"]
    assert val > 0


def test_down_capture_finite(stats):
    """down_capture() returns a finite value for real data."""
    bench = stats.all["SPY -- Benchmark"]
    val = stats.down_capture(bench)["AAPL"]
    assert np.isfinite(val)


def test_up_capture_returns_dict(stats):
    """up_capture() returns a dict keyed by asset name."""
    result = stats.up_capture(stats.all["SPY -- Benchmark"])
    assert isinstance(result, dict)
    assert "AAPL" in result


def test_down_capture_returns_dict(stats):
    """down_capture() returns a dict keyed by asset name."""
    result = stats.down_capture(stats.all["SPY -- Benchmark"])
    assert isinstance(result, dict)
    assert "AAPL" in result


def test_up_capture_nan_for_all_negative_benchmark():
    """up_capture() returns NaN when the benchmark has no positive periods."""
    n = 50
    returns = pl.DataFrame({"Date": [date(2020, 1, 1) + timedelta(days=i) for i in range(n)], "asset": [0.01] * n})
    bench_df = pl.DataFrame({"Date": [date(2020, 1, 1) + timedelta(days=i) for i in range(n)], "bench": [-0.01] * n})
    data = Data.from_returns(returns=returns, benchmark=bench_df)
    bench_series = data.all["bench"]
    result = data.stats.up_capture(bench_series)
    assert np.isnan(result["asset"])
