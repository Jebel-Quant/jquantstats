"""Property-based tests using hypothesis for statistical invariants.

These tests verify mathematical properties that must hold for any valid
returns series, regardless of the specific values.  Examples include:

- Sharpe ratio ≥ 0 when all returns are strictly positive
- Sharpe ratio is scale-invariant: sharpe(k·r) = sharpe(r) for k > 0
- Max drawdown is always in [-1, 0] (signed drawdown in return space)
- Max drawdown = 0 when all returns are non-negative (no peak-to-trough loss)
- Max drawdown is bounded below by −1 (cannot lose more than 100 %)
- Win rate is always in [0, 1]
- Win rate = 1 when every return is strictly positive
- Win rate = 0 when every return is strictly negative
- Volatility is always ≥ 0
- Volatility scales linearly: vol(k·r) = k·vol(r) for k > 0
- Sortino ratio ≥ 0 when all returns are strictly positive
- Average return is positive when every return is strictly positive
- CAGR is non-negative when all returns are non-negative
- lag(0) is a no-op on Portfolio: cashposition is unchanged
"""

import math
from datetime import date, timedelta

import polars as pl
import polars.testing as pt
import pytest

from jquantstats import Data, Portfolio

from .tolerances import TOL_PARITY, TOL_PINNED

hypothesis = pytest.importorskip("hypothesis")
assume = hypothesis.assume
given = hypothesis.given
settings = hypothesis.settings
st = hypothesis.strategies


def _make_returns_df(values: list[float], start: date = date(2020, 1, 1)) -> pl.DataFrame:
    """Build a minimal single-asset returns DataFrame with sequential dates.

    Args:
        values: Daily return values for each observation.
        start: Starting date for the date index (default 2020-01-01).

    Returns:
        pl.DataFrame with columns ``Date`` (pl.Date) and ``Asset`` (pl.Float64).

    """
    dates = [start + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"Date": dates, "Asset": pl.Series(values, dtype=pl.Float64)})


# ── Strategy helpers ─────────────────────────────────────────────────────────

# Returns that are strictly positive (capped at 0.1 to keep NAV values
# in a numerically well-behaved range for 50-observation sequences)
_positive_returns = st.floats(min_value=1e-6, max_value=0.1, allow_nan=False, allow_infinity=False)

# Returns that are non-negative (includes zero, same cap for consistency)
_nonneg_returns = st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)

# General returns: both positive and negative
_general_returns = st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False)

# Strictly negative returns
_strictly_negative_returns = st.floats(min_value=-0.1, max_value=-1e-6, allow_nan=False, allow_infinity=False)

# Positive scale factors (used for scale-invariance tests)
_positive_scale_factor = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)

# Returns conditioned for scale-invariance tests: either exactly zero or a
# realistic magnitude in [1e-6, 0.1]. This excludes the denormal-magnitude band
# (|r| ~ 1e-30), where mean/std-ratio metrics like Sharpe become scale-dependent:
# the negligible-std guard in ``_std_is_negligible`` falls back to an absolute
# floor (~eps**2) once |mean| drops below one machine epsilon, so scaling such a
# series can flip the std from "real dispersion" to "rounding noise" (finite
# Sharpe -> NaN). That regime never occurs in real return data; restricting to it
# keeps the scale-invariance property meaningful. See #836.
_scale_safe_returns = st.one_of(
    st.just(0.0),
    st.floats(min_value=1e-6, max_value=0.1, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-0.1, max_value=-1e-6, allow_nan=False, allow_infinity=False),
)

# Positive prices (used to build synthetic Portfolio instances)
_pos_price = st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)

# Cash positions (any finite float)
_any_position = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.property
@given(returns=st.lists(_positive_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_sharpe_nonnegative_for_all_positive_returns(returns: list[float]) -> None:
    """Sharpe ratio is non-negative when all returns are strictly positive.

    When every daily return is positive the mean excess return is positive,
    so the ratio mean/std must be ≥ 0.  The only exception is when all
    returns are identical (std = 0), in which case the implementation
    returns NaN — that case is explicitly allowed here.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    sharpe_val = data.stats.sharpe()["Asset"]
    if not math.isnan(sharpe_val):
        assert sharpe_val >= 0, f"Expected sharpe >= 0 for all-positive returns, got {sharpe_val}"


@pytest.mark.property
@given(returns=st.lists(_general_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_max_drawdown_always_nonnegative(returns: list[float]) -> None:
    """Max drawdown is always ≥ 0, regardless of the returns sign.

    The implementation returns ``-min(price/peak - 1)`` which is the
    non-negative magnitude of the worst trough relative to its preceding
    peak.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    max_dd = data.stats.max_drawdown()["Asset"]
    assert max_dd <= 0, f"Expected max_drawdown <= 0, got {max_dd}"


@pytest.mark.property
@given(returns=st.lists(_nonneg_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_max_drawdown_zero_for_nonnegative_returns(returns: list[float]) -> None:
    """Max drawdown is exactly 0 when every return is non-negative.

    If no return is ever negative the cumulative price series is
    non-decreasing, so the NAV never falls below its running peak and
    the drawdown series is identically zero.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    max_dd = data.stats.max_drawdown()["Asset"]
    assert max_dd == pytest.approx(0.0, abs=TOL_PINNED), (
        f"Expected max_drawdown == 0 for all-non-negative returns, got {max_dd}"
    )


@pytest.mark.property
@given(returns=st.lists(_general_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_win_rate_in_unit_interval(returns: list[float]) -> None:
    """Win rate is always in [0, 1].

    Win rate is defined as (# positive returns) / (# non-zero returns).
    Both counts are non-negative and the numerator cannot exceed the
    denominator, so the result is always in [0, 1].
    """
    assume(any(r != 0.0 for r in returns))  # reject all-zero series to avoid division by zero in win_rate
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    win_rate = data.stats.win_rate()["Asset"]
    assert 0.0 <= win_rate <= 1.0, f"Expected win_rate in [0, 1], got {win_rate}"


@pytest.mark.property
@given(returns=st.lists(_general_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_volatility_nonnegative(returns: list[float]) -> None:
    """Annualised volatility is always ≥ 0.

    Volatility is ``std(returns) * sqrt(periods)``; the standard deviation
    of a real-valued series is always non-negative.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    vol = data.stats.volatility()["Asset"]
    assert vol >= 0, f"Expected volatility >= 0, got {vol}"


@pytest.mark.property
@given(returns=st.lists(_positive_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_sortino_nonnegative_for_all_positive_returns(returns: list[float]) -> None:
    """Sortino ratio is non-negative when all returns are strictly positive.

    With no negative observations the downside deviation is zero, which
    yields an infinite Sortino ratio (positive mean / 0).  Both +inf and
    any finite non-negative value are considered valid here.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    sortino_val = data.stats.sortino()["Asset"]
    if not math.isnan(sortino_val):
        assert sortino_val >= 0, f"Expected sortino >= 0 for all-positive returns, got {sortino_val}"


# ── New invariant tests ───────────────────────────────────────────────────────


@pytest.mark.property
@given(
    returns=st.lists(_scale_safe_returns, min_size=10, max_size=50),
    scale=_positive_scale_factor,
)
@settings(max_examples=100)
def test_sharpe_scale_invariance(returns: list[float], scale: float) -> None:
    """Sharpe ratio is scale-invariant: sharpe(k·r) = sharpe(r) for any k > 0.

    Multiplying every return by a positive constant k scales both the mean
    and the standard deviation by k, so the ratio mean/std — and therefore
    the annualised Sharpe ratio — is unchanged.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    sharpe_original = data.stats.sharpe()["Asset"]

    scaled_returns = [r * scale for r in returns]
    df_scaled = _make_returns_df(scaled_returns)
    data_scaled = Data.from_returns(returns=df_scaled)
    sharpe_scaled = data_scaled.stats.sharpe()["Asset"]

    is_nan_original = math.isnan(sharpe_original)
    is_nan_scaled = math.isnan(sharpe_scaled)

    if is_nan_original or is_nan_scaled:
        # If either Sharpe is NaN, they must both be NaN; otherwise, scaling changed
        # the NaN/finite status, which violates scale invariance of the metric.
        assert is_nan_original == is_nan_scaled, (
            f"Sharpe NaN-handling not scale-invariant: original={sharpe_original}, scaled={sharpe_scaled} (k={scale})"
        )
    else:
        assert sharpe_scaled == pytest.approx(sharpe_original, rel=TOL_PARITY), (
            f"Sharpe not scale-invariant: original={sharpe_original}, scaled={sharpe_scaled} (k={scale})"
        )


@pytest.mark.property
@given(returns=st.lists(_positive_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_win_rate_one_for_all_positive_returns(returns: list[float]) -> None:
    """Win rate equals 1 when every return is strictly positive.

    Win rate is (# returns > 0) / (# non-zero returns).  When every return
    is positive both numerator and denominator equal the series length, so
    the ratio is exactly 1.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    win_rate = data.stats.win_rate()["Asset"]
    assert win_rate == pytest.approx(1.0), f"Expected win_rate == 1 for all-positive returns, got {win_rate}"


@pytest.mark.property
@given(returns=st.lists(_strictly_negative_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_win_rate_zero_for_all_negative_returns(returns: list[float]) -> None:
    """Win rate equals 0 when every return is strictly negative.

    Win rate is (# returns > 0) / (# non-zero returns).  When every return
    is negative the numerator is zero, so the ratio is exactly 0.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    win_rate = data.stats.win_rate()["Asset"]
    assert win_rate == pytest.approx(0.0), f"Expected win_rate == 0 for all-negative returns, got {win_rate}"


@pytest.mark.property
@given(returns=st.lists(_positive_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_avg_return_positive_for_all_positive_returns(returns: list[float]) -> None:
    """Average return is positive when every return is strictly positive.

    The average of a set of strictly positive numbers is itself strictly
    positive.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    avg = data.stats.avg_return()["Asset"]
    assert avg > 0, f"Expected avg_return > 0 for all-positive returns, got {avg}"


@pytest.mark.property
@given(returns=st.lists(_general_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_max_drawdown_bounded_below_minus_one(returns: list[float]) -> None:
    """Max drawdown is always ≥ −1 (a portfolio cannot lose more than 100 %).

    Because the NAV starts at 1 and each multiplicative factor (1 + r) is
    positive for r > −1, the NAV never becomes negative, so the drawdown
    fraction is always in [0, 1] and the signed max-drawdown in [−1, 0].
    The ``_general_returns`` strategy is bounded to [−0.1, 0.1], so every
    generated value already satisfies r > −1.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    max_dd = data.stats.max_drawdown()["Asset"]
    assert max_dd >= -1.0, f"Expected max_drawdown >= -1.0, got {max_dd}"


@pytest.mark.property
@given(
    returns=st.lists(_general_returns, min_size=10, max_size=50),
    scale=_positive_scale_factor,
)
@settings(max_examples=100)
def test_volatility_scale_linearity(returns: list[float], scale: float) -> None:
    """Annualised volatility scales linearly with the magnitude of returns.

    ``std(k·r) = k·std(r)`` for any k > 0, so the annualised volatility
    satisfies ``vol(k·r) = k·vol(r)``.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    vol_original = data.stats.volatility()["Asset"]

    scaled_returns = [r * scale for r in returns]
    df_scaled = _make_returns_df(scaled_returns)
    data_scaled = Data.from_returns(returns=df_scaled)
    vol_scaled = data_scaled.stats.volatility()["Asset"]

    assert vol_scaled == pytest.approx(vol_original * scale, rel=TOL_PARITY), (
        f"Volatility not linearly scaled: vol(r)={vol_original}, vol({scale}·r)={vol_scaled}"
    )


@pytest.mark.property
@given(returns=st.lists(_nonneg_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_cagr_nonnegative_for_nonneg_returns(returns: list[float]) -> None:
    """CAGR is non-negative when all returns are non-negative.

    When every period return is ≥ 0 the compounded total return is ≥ 0,
    so the geometric annualisation is also ≥ 0.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    cagr = data.stats.cagr()["Asset"]
    assert cagr >= 0, f"Expected cagr >= 0 for all-non-negative returns, got {cagr}"


@pytest.mark.property
@given(
    data_tuple=st.integers(min_value=5, max_value=20).flatmap(
        lambda n: st.tuples(
            st.lists(_pos_price, min_size=n, max_size=n),
            st.lists(_any_position, min_size=n, max_size=n),
        )
    )
)
@settings(max_examples=50)
def test_lag_zero_is_identity_property(data_tuple: tuple[list[float], list[float]]) -> None:
    """lag(0) is a no-op: the Portfolio's cashposition and prices are unchanged.

    Shifting a cash-position series by zero rows must leave every element in
    place, so the resulting Portfolio is data-identical to the original.
    """
    prices_list, positions_list = data_tuple
    n = len(prices_list)
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n)]

    prices_df = pl.DataFrame({"date": dates, "Asset": pl.Series(prices_list, dtype=pl.Float64)})
    positions_df = pl.DataFrame({"date": dates, "Asset": pl.Series(positions_list, dtype=pl.Float64)})

    portfolio = Portfolio(prices=prices_df, cashposition=positions_df, aum=1e5)
    lagged = portfolio.lag(0)

    pt.assert_frame_equal(lagged.cashposition, portfolio.cashposition)
    pt.assert_frame_equal(lagged.prices, portfolio.prices)
    assert lagged.aum == portfolio.aum
