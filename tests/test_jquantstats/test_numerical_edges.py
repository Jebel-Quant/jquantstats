"""Numerical edge-case tests for drawdown math and zero-variance guards.

Covers the behaviors documented in `_stats/_core.py` and `_stats/_basic.py`:

- Drawdown series stay within [0, 1] and finite for any valid returns,
  including a (near-)total wipeout where the high-water mark approaches the
  ``1e-10`` division floor.
- Without a baseline, an exact -100 % first return yields a drawdown of 0
  (the first observation is its own high-water mark); with the phantom
  baseline it yields the expected full drawdown of 1.0.
- Mean/std ratio metrics (Sharpe, trading_cost_impact) report NaN — not an
  absurdly large value — when the dispersion is numerically zero.
"""

import math
from datetime import date, timedelta

import polars as pl
import pytest

from jquantstats import Data, Portfolio
from jquantstats._stats._basic import _BasicStatsMixin
from jquantstats._stats._core import _drawdown_series, _std_is_negligible

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
settings = hypothesis.settings
st = hypothesis.strategies

# Valid simple returns: a position cannot lose more than 100 %.
_valid_returns = st.floats(min_value=-1.0, max_value=0.5, allow_nan=False, allow_infinity=False)


@pytest.mark.property
@given(returns=st.lists(_valid_returns, min_size=1, max_size=50))
@settings(max_examples=100)
def test_drawdown_series_bounded_and_finite(returns: list[float]) -> None:
    """Drawdown values are finite and within [0, 1] for any valid returns series.

    This includes series whose first return is (effectively) -100 %, where the
    high-water mark falls below the 1e-10 division floor.
    """
    dd = _drawdown_series(pl.Series(returns, dtype=pl.Float64))
    for value in dd.to_list():
        assert math.isfinite(value), f"drawdown must be finite, got {value}"
        assert 0.0 <= value <= 1.0, f"drawdown must be in [0, 1], got {value}"


@pytest.mark.property
@given(returns=st.lists(_valid_returns, min_size=1, max_size=50))
@settings(max_examples=100)
def test_drawdown_with_baseline_bounded_and_finite(returns: list[float]) -> None:
    """Baseline drawdown values are finite and within [0, 1] for any valid returns series."""
    dd = _BasicStatsMixin._drawdown_with_baseline(pl.Series(returns, dtype=pl.Float64))
    for value in dd.to_list():
        assert math.isfinite(value), f"drawdown must be finite, got {value}"
        assert 0.0 <= value <= 1.0, f"drawdown must be in [0, 1], got {value}"


def test_total_wipeout_first_return_conventions() -> None:
    """A -100 % first return: dd = 0 without baseline, dd = 1 with the phantom baseline.

    Without a baseline the first observation is its own high-water mark
    (nav == hwm == 0), so no drawdown is reported.  The baseline variant pins
    the initial capital at 1.0 and reports the full loss.
    """
    series = pl.Series([-1.0, 0.0], dtype=pl.Float64)
    assert _drawdown_series(series).to_list() == [0.0, 0.0]
    assert _BasicStatsMixin._drawdown_with_baseline(series).to_list() == [1.0, 1.0]


def test_near_wipeout_drawdown_stays_bounded() -> None:
    """A first return a hair above -100 % keeps the drawdown defined and within [0, 1]."""
    series = pl.Series([-1.0 + 1e-12, 0.5, -0.5], dtype=pl.Float64)
    dd = _drawdown_series(series)
    assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in dd.to_list())


def test_std_is_negligible_handles_none_zero_and_real_dispersion() -> None:
    """The shared zero-std guard: None and 0.0 are negligible, genuine dispersion is not."""
    assert _std_is_negligible(None, 1.0)
    assert _std_is_negligible(0.0, 0.05)
    assert _std_is_negligible(1e-18, 0.05)  # rounding noise relative to the mean
    assert not _std_is_negligible(0.01, 0.05)


def test_sharpe_is_nan_for_constant_nonzero_returns() -> None:
    """Constant non-zero returns have zero dispersion, so Sharpe must be NaN, not huge."""
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(20)]
    df = pl.DataFrame({"Date": dates, "Asset": [0.01] * 20})
    data = Data.from_returns(returns=df)
    assert math.isnan(data.stats.sharpe()["Asset"])


def test_trading_cost_impact_is_nan_for_zero_variance_returns() -> None:
    """trading_cost_impact reports NaN Sharpe at every cost level when returns have no dispersion."""
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(20)]
    prices = pl.DataFrame({"date": dates, "A": [100.0] * 20})
    pos = pl.DataFrame({"date": dates, "A": [1000.0] * 20})
    pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)
    impact = pf.trading_cost_impact(max_bps=3)
    assert impact["sharpe"].is_nan().all()
