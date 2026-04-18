"""Tests for edge cases in the jquantstats package."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jquantstats.data import Data


@pytest.fixture
def data_no_benchmark(returns):
    """Fixture that returns a Data object with no benchmark.

    Args:
        returns: The returns fixture containing asset returns.

    Returns:
        _Data: A Data object with returns but no benchmark.

    """
    return Data.from_returns(returns=returns)


def test_copy_no_benchmark(data_no_benchmark):
    """Tests that the copy() method works correctly when benchmark is None.

    Args:
        data_no_benchmark: Fixture that returns a Data object with no benchmark.

    Verifies:
        1. The return value is a Data object.
        2. The copied object has the same returns as the original.
        3. The benchmark is None in both the original and the copy.
        4. Modifying the copied object does not affect the original.

    """
    # Create a copy of the data object
    data_copy = data_no_benchmark.copy()

    # Verify the copy has the same returns as the original
    assert_frame_equal(data_copy.returns, data_no_benchmark.returns)

    # Verify the benchmark is None in both the original and the copy
    assert data_no_benchmark.benchmark is None
    assert data_copy.benchmark is None

    # Verify that modifying the copy doesn't affect the original
    assert data_copy is not data_no_benchmark
    assert data_copy.returns is not data_no_benchmark.returns


def test_r_squared_no_benchmark(data_no_benchmark):
    """Tests that the r_squared() method raises an AttributeError when benchmark is None.

    Args:
        data_no_benchmark: Fixture that returns a Data object with no benchmark.

    Verifies:
        1. Calling r_squared() raises an AttributeError with the expected message.

    """
    # Verify that calling r_squared() raises an AttributeError
    with pytest.raises(AttributeError, match="No benchmark data available"):
        data_no_benchmark.stats.r_squared()


def test_non_overlapping_dates():
    """Tests that Data.from_returns raises a ValueError when returns and benchmark have non-overlapping dates.

    Verifies:
        1. Calling Data.from_returns with non-overlapping dates raises a ValueError with the expected message.
    """
    # Create returns data with dates in 2020
    returns_dates = [f"2020-01-{i:02d}" for i in range(1, 11)]
    returns = pl.DataFrame({"Date": returns_dates, "Asset": [0.01] * len(returns_dates)}).with_columns(
        pl.col("Date").str.to_date()
    )

    # Create benchmark data with dates in 2010
    benchmark_dates = [f"2010-01-{i:02d}" for i in range(1, 11)]
    benchmark = pl.DataFrame({"Date": benchmark_dates, "Benchmark": [0.01] * len(benchmark_dates)}).with_columns(
        pl.col("Date").str.to_date()
    )

    # Verify that calling Data.from_returns raises a ValueError
    with pytest.raises(ValueError, match=r"No overlapping dates between returns and benchmark\."):
        Data.from_returns(returns=returns, benchmark=benchmark)


def test_periods_per_year_non_date_index():
    """Integer-indexed Data falls back to 252 trading days per year."""
    index = pl.DataFrame({"i": [1, 2, 3, 4, 5]})
    returns = pl.DataFrame({"asset": [0.01, -0.02, 0.03, 0.01, 0.02]})
    data = Data(returns=returns, index=index)
    assert data._periods_per_year == pytest.approx(252.0)


def test_volatility_invalid_periods():
    """Tests that volatility raises TypeError when periods is not numeric."""
    from datetime import date

    returns = pl.DataFrame(
        {
            "Date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "asset": [0.01, -0.02, 0.03],
        }
    )
    data = Data.from_returns(returns=returns)
    with pytest.raises(TypeError):
        data.stats.volatility(periods="bad")


def test_sharpe_variance_nan_zero_std(edge):
    """Tests that sharpe_variance returns NaN when std is zero (all-zero returns)."""
    result = edge.stats.sharpe_variance()
    assert np.isnan(result["returns"])


def test_sharpe_variance_nan_short_series():
    """Tests that sharpe_variance returns NaN when kurtosis cannot be computed (< 4 obs)."""
    from datetime import date

    returns = pl.DataFrame(
        {
            "Date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "asset": [1.0, -0.5, 0.25],
        }
    )
    data = Data.from_returns(returns=returns)
    result = data.stats.sharpe_variance()
    assert np.isnan(result["asset"])


def test_prob_sharpe_ratio_nan_zero_std(edge):
    """Tests that prob_sharpe_ratio returns NaN when std is zero (all-zero returns)."""
    result = edge.stats.probabilistic_sharpe_ratio()
    assert np.isnan(result["returns"])


def test_prob_sharpe_ratio_nan_short_series():
    """Tests that prob_sharpe_ratio returns NaN when kurtosis cannot be computed (< 4 obs)."""
    from datetime import date

    returns = pl.DataFrame(
        {
            "Date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "asset": [1.0, -0.5, 0.25],
        }
    )
    data = Data.from_returns(returns=returns)
    result = data.stats.probabilistic_sharpe_ratio()
    assert np.isnan(result["asset"])


def test_hhi_positive_nan_few_positives(edge):
    """Tests that hhi_positive returns NaN when there are <= 2 positive returns."""
    result = edge.stats.hhi_positive()
    assert np.isnan(result["returns"])


def test_hhi_negative_nan_few_negatives(edge):
    """Tests that hhi_negative returns NaN when there are <= 2 negative returns."""
    result = edge.stats.hhi_negative()
    assert np.isnan(result["returns"])


def test_subtract_rf_invalid_type():
    """Tests that Data.from_returns raises TypeError when rf is not a float or DataFrame."""
    from datetime import date

    returns = pl.DataFrame(
        {
            "Date": [date(2023, 1, 1), date(2023, 1, 2)],
            "asset": [0.01, -0.02],
        }
    )
    with pytest.raises(TypeError, match="rf must be a float or DataFrame"):
        Data.from_returns(returns=returns, rf=1)  # int is not float


# ── Empty returns ─────────────────────────────────────────────────────────────


def test_from_returns_empty_returns_raises():
    """Data.from_returns with a zero-row DataFrame raises ValueError.

    Verifies that an empty returns frame is rejected before any statistics
    are attempted.
    """
    empty = pl.DataFrame({"Date": pl.Series([], dtype=pl.Date), "asset": pl.Series([], dtype=pl.Float64)})
    with pytest.raises(ValueError, match="at least two timestamps"):
        Data.from_returns(returns=empty)


# ── Single-row frame ──────────────────────────────────────────────────────────


def test_from_returns_single_row_raises():
    """Data.from_returns with a single-row DataFrame raises ValueError.

    A meaningful time series requires at least two observations to compute
    any meaningful statistics (e.g. variance, drawdown).
    """
    from datetime import date

    single = pl.DataFrame({"Date": [date(2023, 1, 1)], "asset": [0.01]})
    with pytest.raises(ValueError, match="at least two timestamps"):
        Data.from_returns(returns=single)


# ── All-zero returns — additional statistics ──────────────────────────────────


def test_all_zero_volatility(edge):
    """Volatility of all-zero returns is 0.0 (no variation)."""
    result = edge.stats.volatility()
    assert result["returns"] == pytest.approx(0.0)


def test_all_zero_max_drawdown(edge):
    """max_drawdown of all-zero returns is 0 (price curve never declines)."""
    result = edge.stats.max_drawdown()
    assert result["returns"] == pytest.approx(0.0)


def test_all_zero_avg_drawdown(edge):
    """avg_drawdown of all-zero returns is 0.0 (no underwater periods)."""
    result = edge.stats.avg_drawdown()
    assert result["returns"] == pytest.approx(0.0)


def test_all_zero_calmar_is_nan(edge):
    """Calmar of all-zero returns is NaN because max_drawdown is zero."""
    result = edge.stats.calmar()
    assert np.isnan(result["returns"])


def test_all_zero_recovery_factor_is_nan(edge):
    """recovery_factor of all-zero returns is NaN because max_drawdown is zero."""
    result = edge.stats.recovery_factor()
    assert np.isnan(result["returns"])


def test_all_zero_gain_to_pain_is_nan(edge):
    """gain_to_pain_ratio of all-zero returns is NaN (no losses to divide by)."""
    result = edge.stats.gain_to_pain_ratio()
    assert np.isnan(result["returns"])


def test_all_zero_exposure(edge):
    """Exposure of all-zero returns is 0.0 (no non-zero periods)."""
    result = edge.stats.exposure()
    assert result["returns"] == pytest.approx(0.0)


# ── NaN-containing frames ─────────────────────────────────────────────────────


def test_from_returns_with_nan_succeeds():
    """Data.from_returns accepts returns containing NaN values without raising.

    NaN is a valid IEEE-754 float and must not be mistaken for a missing row.
    """
    from datetime import date

    returns = pl.DataFrame(
        {
            "Date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "asset": [0.01, float("nan"), 0.03],
        }
    )
    data = Data.from_returns(returns=returns)
    assert data.returns.shape[0] == 3


def test_nan_avg_return_propagates_nan():
    """avg_return propagates NaN when the returns series contains NaN.

    In Polars, NaN is not null and is not filtered by is_not_null(), so it
    propagates through mean().
    """
    from datetime import date

    returns = pl.DataFrame(
        {
            "Date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "asset": [0.01, float("nan"), 0.03],
        }
    )
    data = Data.from_returns(returns=returns)
    result = data.stats.avg_return()
    assert np.isnan(result["asset"])


def test_nan_volatility_propagates_nan():
    """Volatility propagates NaN when the returns series contains NaN.

    Polars std() follows IEEE-754 semantics and returns NaN when any element
    in the series is NaN.
    """
    from datetime import date

    returns = pl.DataFrame(
        {
            "Date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "asset": [0.01, float("nan"), 0.03],
        }
    )
    data = Data.from_returns(returns=returns)
    result = data.stats.volatility()
    assert np.isnan(result["asset"])


# ── Partial date overlap (misaligned dates) ───────────────────────────────────


def test_partial_date_overlap_aligns_correctly():
    """Data.from_returns keeps only dates present in both returns and benchmark.

    Returns cover Jan 1–10; benchmark covers Jan 5–15.  The inner join must
    produce 6 rows (Jan 5–10) in both returns and benchmark.
    """
    returns_dates = [f"2020-01-{i:02d}" for i in range(1, 11)]
    benchmark_dates = [f"2020-01-{i:02d}" for i in range(5, 16)]

    returns = pl.DataFrame({"Date": returns_dates, "Asset": [0.01] * 10}).with_columns(pl.col("Date").str.to_date())
    benchmark = pl.DataFrame({"Date": benchmark_dates, "Benchmark": [0.01] * 11}).with_columns(
        pl.col("Date").str.to_date()
    )

    data = Data.from_returns(returns=returns, benchmark=benchmark)

    assert data.returns.shape[0] == 6
    assert data.benchmark is not None
    assert data.benchmark.shape[0] == 6
    assert data.index.shape[0] == 6


# ── null_strategy parameter ───────────────────────────────────────────────────


def _make_returns_with_null(n: int = 4) -> pl.DataFrame:
    """Build a minimal returns frame with one null in the middle."""
    from datetime import date, timedelta

    base = date(2023, 1, 1)
    return pl.DataFrame(
        {
            "Date": [base + timedelta(days=i) for i in range(n)],
            "asset": [0.01, None, 0.03, 0.02],
        }
    )


def test_null_strategy_none_preserves_nulls():
    """Default null_strategy=None leaves null values in place (backward compat)."""
    returns = _make_returns_with_null()
    data = Data.from_returns(returns=returns)
    # null is preserved — returns frame has all 4 rows
    assert data.returns.shape[0] == 4
    assert data.returns["asset"].null_count() == 1


def test_null_strategy_raise_on_null():
    """null_strategy='raise' raises NullsInReturnsError when nulls are found."""
    from jquantstats.exceptions import NullsInReturnsError

    returns = _make_returns_with_null()
    with pytest.raises(NullsInReturnsError, match="'asset'"):
        Data.from_returns(returns=returns, null_strategy="raise")


def test_null_strategy_raise_no_error_when_clean():
    """null_strategy='raise' does not raise when data has no null values."""
    from datetime import date, timedelta

    returns = pl.DataFrame(
        {
            "Date": [date(2023, 1, 1) + timedelta(days=i) for i in range(3)],
            "asset": [0.01, 0.02, 0.03],
        }
    )
    data = Data.from_returns(returns=returns, null_strategy="raise")
    assert data.returns.shape[0] == 3


def test_null_strategy_drop_removes_null_rows():
    """null_strategy='drop' drops rows that contain at least one null."""
    returns = _make_returns_with_null()
    data = Data.from_returns(returns=returns, null_strategy="drop")
    # row 1 (index 1) was null → 3 rows remain
    assert data.returns.shape[0] == 3
    assert data.returns["asset"].null_count() == 0


def test_null_strategy_forward_fill_fills_nulls():
    """null_strategy='forward_fill' fills each null with the previous non-null value."""
    returns = _make_returns_with_null()
    data = Data.from_returns(returns=returns, null_strategy="forward_fill")
    # All 4 rows are kept, no nulls remain
    assert data.returns.shape[0] == 4
    assert data.returns["asset"].null_count() == 0
    # The null at position 1 was filled with the value at position 0
    assert data.returns["asset"][1] == pytest.approx(0.01)


def test_null_strategy_nan_not_affected_by_null_strategy():
    """NaN (IEEE-754) values are NOT treated as null by null_strategy.

    null_strategy only affects Polars null (missing) values; IEEE-754 NaN
    values are a distinct float value and must be unaffected.
    """
    from datetime import date, timedelta

    returns = pl.DataFrame(
        {
            "Date": [date(2023, 1, 1) + timedelta(days=i) for i in range(3)],
            "asset": [0.01, float("nan"), 0.03],
        }
    )
    # "raise" should not trigger because there are no *null* values (only NaN)
    data = Data.from_returns(returns=returns, null_strategy="raise")
    assert data.returns.shape[0] == 3
    # "drop" should leave the NaN row in place
    data_drop = Data.from_returns(returns=returns, null_strategy="drop")
    assert data_drop.returns.shape[0] == 3


def test_null_strategy_applied_to_benchmark():
    """null_strategy is applied to benchmark as well as returns."""
    from datetime import date, timedelta

    from jquantstats.exceptions import NullsInReturnsError

    base = date(2023, 1, 1)
    dates = [base + timedelta(days=i) for i in range(4)]
    returns = pl.DataFrame({"Date": dates, "asset": [0.01, 0.02, 0.03, 0.04]})
    benchmark = pl.DataFrame({"Date": dates, "bench": [0.01, None, 0.03, 0.04]})

    with pytest.raises(NullsInReturnsError, match="'bench'"):
        Data.from_returns(returns=returns, benchmark=benchmark, null_strategy="raise")

    data = Data.from_returns(returns=returns, benchmark=benchmark, null_strategy="drop")
    assert data.benchmark is not None
    assert data.benchmark.shape[0] == 3


def test_from_prices_null_strategy_forwarded():
    """from_prices passes null_strategy through to from_returns."""
    from datetime import date, timedelta

    from jquantstats.exceptions import NullsInReturnsError

    base = date(2023, 1, 1)
    # Null price in the middle → null pct_change returns
    prices = pl.DataFrame(
        {
            "Date": [base + timedelta(days=i) for i in range(6)],
            "asset": [100.0, None, 102.0, 103.0, 104.0, 105.0],
        }
    )
    with pytest.raises(NullsInReturnsError, match="'asset'"):
        Data.from_prices(prices=prices, null_strategy="raise")

    data = Data.from_prices(prices=prices, null_strategy="drop")
    # pct_change on [100, null, 102, 103, 104, 105] → slice(1) gives 5 rows,
    # 2 of which are null (the null and the row after it); drop_nulls leaves 3 rows
    assert data.returns["asset"].null_count() == 0
    assert data.returns.shape[0] == 3


def test_null_strategy_interpolate_fills_interior_nulls():
    """null_strategy='interpolate' linearly interpolates interior nulls."""
    from datetime import date, timedelta

    base = date(2023, 1, 1)
    returns = pl.DataFrame(
        {
            "Date": [base + timedelta(days=i) for i in range(5)],
            # Interior nulls at positions 1 and 2; known values: 0.01, 0.04
            "asset": [0.01, None, None, 0.04, 0.05],
        }
    )
    data = Data.from_returns(returns=returns, null_strategy="interpolate")
    assert data.returns.shape[0] == 5
    assert data.returns["asset"].null_count() == 0
    # Linear interpolation between 0.01 and 0.04 over 3 steps:
    # position 1 → 0.02, position 2 → 0.03
    assert data.returns["asset"][1] == pytest.approx(0.02)
    assert data.returns["asset"][2] == pytest.approx(0.03)


def test_null_strategy_interpolate_backfills_leading_nulls():
    """null_strategy='interpolate' backfills leading nulls (the backfill trick)."""
    from datetime import date, timedelta

    base = date(2023, 1, 1)
    returns = pl.DataFrame(
        {
            "Date": [base + timedelta(days=i) for i in range(5)],
            # Leading nulls at positions 0 and 1; first known value is 0.03
            "asset": [None, None, 0.03, 0.04, 0.05],
        }
    )
    data = Data.from_returns(returns=returns, null_strategy="interpolate")
    assert data.returns.shape[0] == 5
    assert data.returns["asset"].null_count() == 0
    # Leading nulls are filled with the first known value via backward_fill
    assert data.returns["asset"][0] == pytest.approx(0.03)
    assert data.returns["asset"][1] == pytest.approx(0.03)
