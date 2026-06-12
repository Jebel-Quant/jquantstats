"""Mutation-kill tests for PortfolioUtils default arguments and __slots__.

Each test pins the *documented* default of a `PortfolioUtils` accessor by
asserting that a no-argument call is elementwise identical to a call with the
default passed explicitly, on portfolio data crafted so any perturbed default
yields a different result.  A sentinel assertion in each test proves the data
actually discriminates between the documented default and its mutated value,
so the equality check is guaranteed to fail under the mutant.

Targets the surviving mutmut mutants in
``src/jquantstats/_utils/_portfolio.py``: the ``__slots__`` removal and the
default-argument mutations on ``to_prices``, ``rebase``, ``group_returns``,
``aggregate_returns``, ``to_excess_returns``,
``to_volatility_adjusted_returns``, ``exponential_stdev``, ``winsorise``
and ``exponential_cov``.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jquantstats import Portfolio

from ..tolerances import TOL_PINNED

# Row index of the injected outlier return (see fixture below).
_OUTLIER_ROW = 40

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def portfolio() -> Portfolio:
    """90-day single-asset Portfolio whose returns discriminate mutated defaults.

    Daily returns are non-constant seeded noise (std ~1 %) with one large
    outlier (+20 %) at row ``_OUTLIER_ROW``.  With ``cash_position == aum``
    the portfolio-level return series equals the asset's percentage change,
    so:

    - 90 rows of varying returns make rolling window 60 vs 61 and EWMA
      span/half-life 30 vs 31 differ on later rows;
    - the outlier exceeds any 3-sigma rolling band, so winsorise clips it
      differently for window 7 vs 8 and for n_sigma 3 vs 4;
    - returns are nonzero, so subtracting a mutated risk-free rate changes
      every value.
    """
    n = 90
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0, 0.01, n)
    rets[_OUTLIER_ROW] = 0.20  # far beyond 3 rolling sigmas of the noise
    prices = 100.0 * np.cumprod(1.0 + rets)

    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    aum = 1_000.0
    return Portfolio.from_cash_position(
        prices=pl.DataFrame({"date": dates, "A": pl.Series(prices, dtype=pl.Float64)}),
        cash_position=pl.DataFrame({"date": dates, "A": pl.Series([aum] * n, dtype=pl.Float64)}),
        aum=aum,
    )


def _assert_cov_dicts_equal(a: dict, b: dict) -> None:
    """Assert two exponential_cov results have identical keys and matrices (NaN-aware)."""
    assert list(a.keys()) == list(b.keys())
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def _cov_dicts_differ(a: dict, b: dict) -> bool:
    """Return True if two exponential_cov results differ in keys or any cell."""
    if list(a.keys()) != list(b.keys()):
        return True
    return any(not np.array_equal(a[k], b[k], equal_nan=True) for k in a)


# ─── __slots__ ────────────────────────────────────────────────────────────────


def test_slots_rejects_unknown_attribute(portfolio):
    """PortfolioUtils declares __slots__, so undeclared attributes must raise AttributeError."""
    utils = portfolio.utils
    with pytest.raises(AttributeError):
        utils.not_a_declared_slot = object()


# ─── to_prices ────────────────────────────────────────────────────────────────


def test_to_prices_default_base_is_1e5(portfolio):
    """to_prices() must equal to_prices(base=1e5) and start at exactly 1e5."""
    default = portfolio.utils.to_prices()
    explicit = portfolio.utils.to_prices(base=1e5)
    assert_frame_equal(default, explicit, check_exact=True)
    # First portfolio return is 0.0, so the first price equals the base exactly.
    assert default["returns"][0] == pytest.approx(1e5, abs=TOL_PINNED)


# ─── rebase ───────────────────────────────────────────────────────────────────


def test_rebase_default_base_is_100(portfolio):
    """rebase() must equal rebase(base=100.0) and anchor the first price at 100."""
    default = portfolio.utils.rebase()
    explicit = portfolio.utils.rebase(base=100.0)
    assert_frame_equal(default, explicit, check_exact=True)
    assert default["returns"][0] == pytest.approx(100.0, abs=TOL_PINNED)


# ─── group_returns / aggregate_returns ───────────────────────────────────────


def test_group_returns_default_period_is_1mo(portfolio):
    """group_returns() must equal group_returns(period="1mo"); a mangled period must raise."""
    default = portfolio.utils.group_returns()
    explicit = portfolio.utils.group_returns(period="1mo")
    assert_frame_equal(default, explicit, check_exact=True)
    # Sentinel: the mutated default ("XX1moXX") is not a valid Polars interval.
    with pytest.raises(pl.exceptions.PolarsError):
        portfolio.utils.group_returns(period="XX1moXX")


def test_aggregate_returns_default_period_is_1mo(portfolio):
    """aggregate_returns() must equal aggregate_returns(period="1mo"); a mangled period must raise."""
    default = portfolio.utils.aggregate_returns()
    explicit = portfolio.utils.aggregate_returns(period="1mo")
    assert_frame_equal(default, explicit, check_exact=True)
    with pytest.raises(pl.exceptions.PolarsError):
        portfolio.utils.aggregate_returns(period="XX1moXX")


# ─── to_excess_returns ────────────────────────────────────────────────────────


def test_to_excess_returns_default_rf_is_zero(portfolio):
    """to_excess_returns() must equal to_excess_returns(rf=0.0); rf=1.0 must differ."""
    default = portfolio.utils.to_excess_returns()
    explicit = portfolio.utils.to_excess_returns(rf=0.0)
    assert_frame_equal(default, explicit, check_exact=True)
    # Sentinel: the mutated default rf=1.0 shifts every (finite, nonzero) return.
    perturbed = portfolio.utils.to_excess_returns(rf=1.0)
    assert perturbed["returns"][-1] != explicit["returns"][-1]


# ─── to_volatility_adjusted_returns ──────────────────────────────────────────


def test_to_volatility_adjusted_returns_default_window_is_60(portfolio):
    """to_volatility_adjusted_returns() must equal the explicit window=60 call; 61 must differ."""
    default = portfolio.utils.to_volatility_adjusted_returns()
    explicit = portfolio.utils.to_volatility_adjusted_returns(window=60)
    assert_frame_equal(default, explicit, check_exact=True)
    # Sentinel: with 90 rows of varying returns, rolling_std(60) != rolling_std(61).
    perturbed = portfolio.utils.to_volatility_adjusted_returns(window=61)
    assert perturbed["returns"][-1] != explicit["returns"][-1]


# ─── exponential_stdev ────────────────────────────────────────────────────────


def test_exponential_stdev_default_window_is_30(portfolio):
    """exponential_stdev() must equal exponential_stdev(window=30); span 31 must differ."""
    default = portfolio.utils.exponential_stdev()
    explicit = portfolio.utils.exponential_stdev(window=30)
    assert_frame_equal(default, explicit, check_exact=True)
    perturbed = portfolio.utils.exponential_stdev(window=31)
    assert perturbed["returns"][-1] != explicit["returns"][-1]


def test_exponential_stdev_default_is_halflife_false(portfolio):
    """exponential_stdev() must equal exponential_stdev(is_halflife=False); half-life mode must differ."""
    default = portfolio.utils.exponential_stdev()
    explicit = portfolio.utils.exponential_stdev(is_halflife=False)
    assert_frame_equal(default, explicit, check_exact=True)
    # Sentinel: span-30 and half-life-30 decays differ on non-constant data.
    perturbed = portfolio.utils.exponential_stdev(is_halflife=True)
    assert perturbed["returns"][-1] != explicit["returns"][-1]


# ─── winsorise ────────────────────────────────────────────────────────────────


def test_winsorise_default_window_is_7(portfolio):
    """winsorise() must equal winsorise(window=7); window=8 must clip the outlier differently."""
    default = portfolio.utils.winsorise()
    explicit = portfolio.utils.winsorise(window=7)
    assert_frame_equal(default, explicit, check_exact=True)
    # Sentinel: the trailing 7- and 8-row bands around the outlier differ.
    perturbed = portfolio.utils.winsorise(window=8)
    assert perturbed["returns"][_OUTLIER_ROW] != explicit["returns"][_OUTLIER_ROW]


def test_winsorise_default_n_sigma_is_3(portfolio):
    """winsorise() must equal winsorise(n_sigma=3.0); n_sigma=4.0 must clip the outlier differently."""
    default = portfolio.utils.winsorise()
    explicit = portfolio.utils.winsorise(n_sigma=3.0)
    assert_frame_equal(default, explicit, check_exact=True)
    # Sentinel: the outlier exceeds 3 rolling sigmas, so 3- and 4-sigma clips differ.
    perturbed = portfolio.utils.winsorise(n_sigma=4.0)
    assert perturbed["returns"][_OUTLIER_ROW] != explicit["returns"][_OUTLIER_ROW]


# ─── exponential_cov ─────────────────────────────────────────────────────────


def test_exponential_cov_default_window_is_30(portfolio):
    """exponential_cov() must equal exponential_cov(window=30); span 31 must differ."""
    default = portfolio.utils.exponential_cov()
    explicit = portfolio.utils.exponential_cov(window=30)
    _assert_cov_dicts_equal(default, explicit)
    perturbed = portfolio.utils.exponential_cov(window=31)
    assert _cov_dicts_differ(perturbed, explicit)


def test_exponential_cov_default_is_halflife_false(portfolio):
    """exponential_cov() must equal exponential_cov(is_halflife=False); half-life mode must differ."""
    default = portfolio.utils.exponential_cov()
    explicit = portfolio.utils.exponential_cov(is_halflife=False)
    _assert_cov_dicts_equal(default, explicit)
    perturbed = portfolio.utils.exponential_cov(is_halflife=True)
    assert _cov_dicts_differ(perturbed, explicit)
