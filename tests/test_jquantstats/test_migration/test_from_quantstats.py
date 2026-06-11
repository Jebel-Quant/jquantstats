"""Tests ported from the quantstats test suite, adapted for the jquantstats API.

Each class mirrors a class from ``quantstats/tests/test_stats.py``.  Tests that
are already exercised by the parametrised comparisons in ``test_basic.py``,
``test_benchmark.py``, ``test_rolling.py``, ``test_reporting.py``, or
``test_new_stats.py`` are intentionally omitted to avoid duplication.

The focus here is on:
* edge-case inputs not covered elsewhere (all-positive / all-negative returns,
  single observation, empty frame);
* mathematical-property assertions (e.g. ann. vol = sqrt(N) × raw vol,
  CVaR ≤ VaR, Sortino ≠ Sharpe);
* the sign / shape conventions of the drawdown series; and
* multi-column (DataFrame) input handling.

Note on NaN semantics
---------------------
quantstats (pandas) silently drops NaN values before computing statistics.
jquantstats (polars) treats NaN as a valid IEEE-754 float, so NaN propagates
through mean- and std-based statistics.  Edge-case tests below document this
deliberate divergence.

Security note: uses pytest assert statements (S101) which are safe in test code.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from jquantstats import Data

from ..tolerances import TOL_COMPOUNDING, TOL_PINNED

# ── Module-level synthetic fixtures ───────────────────────────────────────────


@pytest.fixture(scope="module")
def sample_data() -> pl.DataFrame:
    """500-period synthetic returns (seed 42), integer-indexed."""
    rng = np.random.default_rng(42)
    n = 500
    vals = (rng.standard_normal(n) * 0.02).tolist()
    return pl.DataFrame({"day": list(range(n)), "strategy": vals})


@pytest.fixture(scope="module")
def sample_stats(sample_data: pl.DataFrame) -> object:
    """Stats object backed by *sample_data*."""
    index = sample_data.select("day")
    returns = sample_data.select("strategy")
    return Data(returns=returns, index=index).stats


@pytest.fixture(scope="module")
def positive_data() -> pl.DataFrame:
    """100-period strictly-positive returns (seed 0)."""
    rng = np.random.default_rng(0)
    n = 100
    vals = (np.abs(rng.standard_normal(n) * 0.01) + 0.001).tolist()
    return pl.DataFrame({"day": list(range(n)), "r": vals})


@pytest.fixture(scope="module")
def negative_data() -> pl.DataFrame:
    """100-period strictly-negative returns (seed 1)."""
    rng = np.random.default_rng(1)
    n = 100
    vals = (-np.abs(rng.standard_normal(n) * 0.01) - 0.001).tolist()
    return pl.DataFrame({"day": list(range(n)), "r": vals})


# ── TestBasicStats ─────────────────────────────────────────────────────────────


class TestBasicStats:
    """Port of quantstats TestBasicStats.

    Covers comp, compsum, exposure, and win_rate, including the all-positive /
    all-negative edge cases that are missing from the parametrised migration
    tests.
    """

    def test_comp(self, sample_stats: object) -> None:
        """comp() returns a finite float."""
        result = sample_stats.comp()  # type: ignore[attr-defined]
        assert np.isfinite(result["strategy"])

    def test_compsum_length(self, sample_stats: object) -> None:
        """compsum() returns a DataFrame whose row count matches the input."""
        result = sample_stats.compsum()  # type: ignore[attr-defined]
        assert result.shape[0] == sample_stats.all.shape[0]  # type: ignore[attr-defined]

    def test_compsum_final_value_matches_comp(self, sample_stats: object) -> None:
        """The final value of compsum() equals comp()."""
        result = sample_stats.compsum()  # type: ignore[attr-defined]
        final = result["strategy"][-1]
        expected = sample_stats.comp()["strategy"]  # type: ignore[attr-defined]
        assert final == pytest.approx(expected, rel=TOL_PINNED)

    def test_exposure(self, sample_stats: object) -> None:
        """exposure() is in [0, 1]."""
        result = sample_stats.exposure()  # type: ignore[attr-defined]
        assert 0 <= result["strategy"] <= 1

    def test_win_rate_in_range(self, sample_stats: object) -> None:
        """win_rate() is in [0, 1] for random returns."""
        result = sample_stats.win_rate()  # type: ignore[attr-defined]
        assert 0 <= result["strategy"] <= 1

    def test_win_rate_all_positive(self, positive_data: pl.DataFrame) -> None:
        """win_rate() is 1.0 for a series consisting entirely of positive returns."""
        d = Data(returns=positive_data.select("r"), index=positive_data.select("day"))
        assert d.stats.win_rate()["r"] == pytest.approx(1.0)

    def test_win_rate_all_negative(self, negative_data: pl.DataFrame) -> None:
        """win_rate() is 0.0 for a series consisting entirely of negative returns."""
        d = Data(returns=negative_data.select("r"), index=negative_data.select("day"))
        assert d.stats.win_rate()["r"] == pytest.approx(0.0)


# ── TestRiskMetrics ────────────────────────────────────────────────────────────


class TestRiskMetrics:
    """Port of quantstats TestRiskMetrics.

    Focuses on properties not already verified by test_basic.py migration tests:
    the annualisation ratio for volatility and the CVaR ≤ VaR relationship.
    """

    def test_volatility_positive_and_finite(self, sample_stats: object) -> None:
        """volatility() returns a positive, finite number."""
        result = sample_stats.volatility()  # type: ignore[attr-defined]
        assert result["strategy"] > 0
        assert np.isfinite(result["strategy"])

    def test_volatility_annualized_ratio(self, sample_stats: object) -> None:
        """Annualised vol / non-annualised vol equals sqrt(periods).

        This mirrors the quantstats test_volatility_annualized test.
        """
        periods = 252
        raw = sample_stats.volatility(periods=periods, annualize=False)["strategy"]  # type: ignore[attr-defined]
        ann = sample_stats.volatility(periods=periods, annualize=True)["strategy"]  # type: ignore[attr-defined]
        assert ann / raw == pytest.approx(np.sqrt(periods), rel=TOL_COMPOUNDING)

    def test_max_drawdown_in_range(self, sample_stats: object) -> None:
        """max_drawdown() is in [-1, 0]."""
        result = sample_stats.max_drawdown()  # type: ignore[attr-defined]
        assert -1 <= result["strategy"] <= 0

    def test_var_negative(self, sample_stats: object) -> None:
        """VaR at 5 % confidence is a negative number (loss)."""
        result = sample_stats.value_at_risk(alpha=0.05)  # type: ignore[attr-defined]
        assert result["strategy"] < 0

    def test_cvar_at_least_as_extreme_as_var(self, sample_stats: object) -> None:
        """CVaR (expected shortfall) is at least as negative as VaR.

        This corresponds to the quantstats assertion ``cvar <= var``.
        """
        var = sample_stats.value_at_risk(alpha=0.05)["strategy"]  # type: ignore[attr-defined]
        cvar = sample_stats.conditional_value_at_risk(alpha=0.05)["strategy"]  # type: ignore[attr-defined]
        assert cvar <= var


# ── TestRatios ─────────────────────────────────────────────────────────────────


class TestRatios:
    """Port of quantstats TestRatios.

    Tests that Sharpe, Sortino, Calmar, Omega, and CAGR return valid values,
    and that Sortino ≠ Sharpe (Sortino only penalises downside deviation).
    """

    def test_sharpe_finite(self, sample_stats: object) -> None:
        """sharpe() returns a finite value."""
        result = sample_stats.sharpe(periods=252)  # type: ignore[attr-defined]
        assert np.isfinite(result["strategy"])

    def test_sortino_finite(self, sample_stats: object) -> None:
        """sortino() returns a finite value."""
        result = sample_stats.sortino(periods=252)  # type: ignore[attr-defined]
        assert np.isfinite(result["strategy"])

    def test_sortino_differs_from_sharpe(self, sample_stats: object) -> None:
        """Sortino and Sharpe are different because Sortino only uses downside deviation."""
        sharpe = sample_stats.sharpe(periods=252)["strategy"]  # type: ignore[attr-defined]
        sortino = sample_stats.sortino(periods=252)["strategy"]  # type: ignore[attr-defined]
        assert sharpe != pytest.approx(sortino)

    def test_calmar_finite(self, sample_stats: object) -> None:
        """calmar() returns a finite value for returns with a non-zero drawdown."""
        result = sample_stats.calmar(periods=252)  # type: ignore[attr-defined]
        assert np.isfinite(result["strategy"])

    def test_omega_positive(self, sample_stats: object) -> None:
        """omega() is always positive."""
        result = sample_stats.omega(periods=252)  # type: ignore[attr-defined]
        assert result["strategy"] > 0

    def test_cagr_finite(self, sample_stats: object) -> None:
        """cagr() returns a finite value."""
        result = sample_stats.cagr(periods=252)  # type: ignore[attr-defined]
        assert np.isfinite(result["strategy"])


# ── TestBenchmarkComparison ────────────────────────────────────────────────────


class TestBenchmarkComparison:
    """Port of quantstats TestBenchmarkComparison.

    Uses the real-data ``stats`` fixture (with benchmark) from the shared
    test_migration conftest.  Detailed value comparisons live in
    test_benchmark.py; this file verifies structure and range only.
    """

    def test_greeks_returns_alpha_and_beta(self, stats: object) -> None:
        """greeks() returns a dict containing 'alpha' and 'beta' per asset."""
        result = stats.greeks()  # type: ignore[attr-defined]
        for asset_metrics in result.values():
            assert "alpha" in asset_metrics
            assert "beta" in asset_metrics
            assert np.isfinite(asset_metrics["alpha"])
            assert np.isfinite(asset_metrics["beta"])

    def test_r_squared_in_range(self, stats: object) -> None:
        """r_squared() is in [0, 1]."""
        result = stats.r_squared()  # type: ignore[attr-defined]
        for val in result.values():
            assert 0 <= val <= 1

    def test_information_ratio_finite(self, stats: object) -> None:
        """information_ratio() returns a finite value."""
        result = stats.information_ratio(periods_per_year=252)  # type: ignore[attr-defined]
        for val in result.values():
            assert np.isfinite(val)


# ── TestDrawdown ───────────────────────────────────────────────────────────────


class TestDrawdown:
    """Port of quantstats TestDrawdown.

    Note on sign convention: quantstats ``to_drawdown_series`` returns values
    ≤ 0 (drawdown expressed as a negative fraction).  jquantstats ``drawdown()``
    returns values ≥ 0 (drawdown expressed as a positive fraction below the
    high-water mark).  The tests below reflect the jquantstats convention.
    """

    def test_drawdown_series_is_dataframe(self, stats: object) -> None:
        """drawdown() returns a Polars DataFrame."""
        dd = stats.drawdown()  # type: ignore[attr-defined]
        assert isinstance(dd, pl.DataFrame)

    def test_drawdown_series_length(self, stats: object) -> None:
        """drawdown() has the same number of rows as the input."""
        dd = stats.drawdown()  # type: ignore[attr-defined]
        assert dd.shape[0] == stats.all.shape[0]  # type: ignore[attr-defined]

    def test_drawdown_series_non_negative(self, stats: object) -> None:
        """All drawdown values are ≥ 0 (jquantstats uses positive-fraction convention)."""
        dd = stats.drawdown()  # type: ignore[attr-defined]
        for col in stats.assets:  # type: ignore[attr-defined]
            assert (dd[col] >= 0).all(), f"Negative drawdown value found in column {col!r}"

    def test_drawdown_details_structure(self, stats: object) -> None:
        """drawdown_details() returns a DataFrame with start, end, and max_drawdown columns."""
        result = stats.drawdown_details()  # type: ignore[attr-defined]
        for col, df in result.items():
            assert isinstance(df, pl.DataFrame), f"Expected DataFrame for {col!r}"
            assert "start" in df.columns, f"'start' missing from {col!r}"
            assert "end" in df.columns, f"'end' missing from {col!r}"
            assert "max_drawdown" in df.columns, f"'max_drawdown' missing from {col!r}"


# ── TestConsecutive ────────────────────────────────────────────────────────────


class TestConsecutive:
    """Port of quantstats TestConsecutive."""

    def test_consecutive_wins_non_negative_int(self, sample_stats: object) -> None:
        """consecutive_wins() returns a non-negative integer."""
        result = sample_stats.consecutive_wins()  # type: ignore[attr-defined]
        val = result["strategy"]
        assert isinstance(val, int)
        assert val >= 0

    def test_consecutive_losses_non_negative_int(self, sample_stats: object) -> None:
        """consecutive_losses() returns a non-negative integer."""
        result = sample_stats.consecutive_losses()  # type: ignore[attr-defined]
        val = result["strategy"]
        assert isinstance(val, int)
        assert val >= 0


# ── TestEdgeCases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Port of quantstats TestEdgeCases, adapted for jquantstats semantics.

    quantstats raises on empty input, accepts single-row input, and silently
    drops NaN values.  jquantstats requires at least two observations and
    propagates NaN through statistics (Polars IEEE-754 semantics).
    """

    def test_empty_returns_raises(self) -> None:
        """Data.from_returns with a zero-row DataFrame raises ValueError."""
        empty = pl.DataFrame({"Date": pl.Series([], dtype=pl.Date), "asset": pl.Series([], dtype=pl.Float64)})
        with pytest.raises(ValueError, match="at least two timestamps"):
            Data.from_returns(returns=empty)

    def test_comp_with_minimal_series(self) -> None:
        """comp() is finite for a two-observation series (minimum valid input)."""
        from datetime import date

        two_row = pl.DataFrame({"Date": [date(2020, 1, 1), date(2020, 1, 2)], "asset": [0.01, 0.02]})
        data = Data.from_returns(returns=two_row)
        result = data.stats.comp()
        assert np.isfinite(result["asset"])

    def test_nan_propagates_through_sharpe(self) -> None:
        """NaN in returns propagates through sharpe() in jquantstats.

        In quantstats (pandas) NaN is dropped before computation, so sharpe()
        returns a finite number.  In jquantstats (polars) NaN is a valid float
        that propagates through mean() and std(), so sharpe() returns NaN.
        Users who need NaN-safe statistics should sanitise their data first.
        """
        from datetime import date

        returns = pl.DataFrame(
            {
                "Date": [date(2020, 1, i) for i in range(1, 11)],
                "asset": [0.01, 0.02, float("nan"), -0.01, 0.03, -0.02, 0.01, 0.015, -0.005, 0.02],
            }
        )
        data = Data.from_returns(returns=returns)
        result = data.stats.sharpe(periods=252)
        # NaN propagates in Polars; this differs from quantstats/pandas behaviour.
        assert np.isnan(result["asset"])

    def test_multi_column_sharpe_returns_dict(self, stats: object) -> None:
        """sharpe() returns a dict with one finite entry per asset column.

        This is the jquantstats equivalent of the quantstats test that verifies
        DataFrame input produces a Series result.
        """
        result = stats.sharpe(periods=252)  # type: ignore[attr-defined]
        assert isinstance(result, dict)
        assert len(result) == len(stats.assets)  # type: ignore[attr-defined]
        for val in result.values():
            assert np.isfinite(val)
