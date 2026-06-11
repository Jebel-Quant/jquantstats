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


# ── Mutation-testing killers (issue #808) ─────────────────────────────────────
# These pin behaviors that line coverage alone could not distinguish from
# mutants: exact values, boundary inclusivity, and decorator error paths.


def test_drawdown_series_exact_values():
    """Drawdown values are exact, not just bounded — kills sign/operator mutants."""
    dd = _drawdown_series(pl.Series([0.0, -0.1, 0.2], dtype=pl.Float64))
    assert [round(x, 10) for x in dd.to_list()] == [0.0, 0.1, 0.0]


def test_to_float_conversions():
    """_to_float maps None to 0.0, timedeltas to seconds, and floats through."""
    from jquantstats._stats._core import _to_float

    assert _to_float(None) == 0.0
    assert _to_float(2.5) == 2.5
    assert _to_float(timedelta(seconds=90)) == 90.0


def test_mean_of_empty_series_is_nan():
    """_mean returns NaN (not an error, not 0.0) for empty input."""
    from jquantstats._stats._core import _mean

    assert math.isnan(_mean(pl.Series([], dtype=pl.Float64)))


def test_std_is_negligible_threshold_boundaries():
    """The 10-epsilon threshold is exact: scale factor, mean scaling, and inclusive boundary."""
    import sys

    eps = sys.float_info.epsilon
    # threshold scales with |mean|: cutoff is 10 * eps * |mean|
    assert _std_is_negligible(9.9 * eps * 0.05, 0.05)
    assert not _std_is_negligible(10.5 * eps * 0.05, 0.05)
    # well above the relative threshold but below an eps/|mean| mis-scaling
    assert not _std_is_negligible(1e-15, 0.05)
    # absolute floor for zero mean, boundary is inclusive (<=)
    assert _std_is_negligible(eps * eps * 10.0, 0.0)


def test_decorators_raise_for_hosts_without_data_attr():
    """columnwise_stat/to_frame fail loudly when the host lacks the data attribute."""
    from jquantstats._stats._core import columnwise_stat, to_frame

    class _NoData:
        """Host class deliberately missing the '_data' attribute."""

        @columnwise_stat
        def metric(self, series):
            """Dummy metric."""
            return 0.0

        @to_frame
        def framed(self, series):
            """Dummy per-column frame builder."""
            return series

    obj = _NoData()
    with pytest.raises(AttributeError, match=r"columnwise_stat requires host object to define '_data'"):
        obj.metric()
    with pytest.raises(AttributeError, match=r"to_frame requires host object to define '_data'"):
        obj.framed()


def test_decorated_methods_preserve_metadata(data):
    """@wraps keeps the wrapped method's name; to_frame builds a full-height frame."""
    assert data.stats.sharpe.__name__ == "sharpe"
    assert data.stats.compsum.__name__ == "compsum"
    frame = data.stats.compsum()
    assert frame.height == data.returns.height
