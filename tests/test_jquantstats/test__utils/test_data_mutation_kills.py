"""Mutation-killing tests for ``jquantstats._utils._data.DataUtils``.

Each test pins an exact behaviour (default argument values, null-fill
semantics, arithmetic identities, error-message wording, period-alias
resolution) that mutation testing showed was previously unasserted.

Security note: Test code uses pytest assertions (S101), which are intentional
and safe in the test context.
"""

from __future__ import annotations

import math
import statistics
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from jquantstats import Data
from jquantstats.exceptions import MissingDateColumnError

from ..tolerances import TOL_FLOAT64

# ─── Helpers / fixtures ───────────────────────────────────────────────────────


def _make_data(values: dict[str, list[float | None]], start: date = date(2020, 1, 1)) -> Data:
    """Build a daily-indexed Data object from per-asset return lists."""
    n = len(next(iter(values.values())))
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    frame = pl.DataFrame({"Date": dates, **{k: pl.Series(v, dtype=pl.Float64) for k, v in values.items()}})
    return Data.from_returns(returns=frame)


def _varied(n: int) -> list[float]:
    """Deterministic non-constant returns with a small trend.

    The trend guarantees that any two distinct rolling windows (and any two
    distinct EWM decay parameters) produce different statistics.
    """
    return [(((i * 37) % 19) - 9) / 1000.0 + i * 1e-5 for i in range(n)]


@pytest.fixture
def varied_data() -> Data:
    """80-day single-asset Data with non-constant returns."""
    return _make_data({"A": _varied(80)})


@pytest.fixture
def varied_two_asset_data() -> Data:
    """80-day two-asset Data with non-constant returns."""
    return _make_data({"A": _varied(80), "B": [v * 0.5 + 0.001 for v in _varied(80)]})


@pytest.fixture
def year_span_data() -> Data:
    """400-day single-asset Data spanning two calendar years."""
    return _make_data({"A": _varied(400)})


@pytest.fixture
def month_null_data() -> Data:
    """5-day single-asset Data within one month containing a null return."""
    return _make_data({"A": [0.01, None, 0.02, -0.005, 0.0]})


@pytest.fixture
def int_index_data() -> Data:
    """Integer-indexed (non-temporal) Data."""
    returns = pl.DataFrame({"A": pl.Series([0.01, 0.02, -0.01, 0.0, 0.03], dtype=pl.Float64)})
    index = pl.DataFrame({"index": list(range(5))})
    return Data(returns=returns, index=index)


# ─── Period aliases ───────────────────────────────────────────────────────────


def test_period_alias_daily_matches_1d(year_span_data):
    """'daily' must resolve to the Polars interval '1d' and give one row per day."""
    via_alias = year_span_data.utils.group_returns(period="daily")
    via_native = year_span_data.utils.group_returns(period="1d")
    assert via_alias.height == 400
    assert via_alias.equals(via_native)


def test_period_alias_weekly_matches_1w(year_span_data):
    """'weekly' must resolve to the Polars interval '1w'."""
    via_alias = year_span_data.utils.group_returns(period="weekly")
    via_native = year_span_data.utils.group_returns(period="1w")
    assert via_alias.height > 50
    assert via_alias.equals(via_native)


def test_period_alias_quarterly_matches_1q(year_span_data):
    """'quarterly' must resolve to the Polars interval '1q'."""
    via_alias = year_span_data.utils.group_returns(period="quarterly")
    via_native = year_span_data.utils.group_returns(period="1q")
    assert via_alias.height == 5  # Q1-Q4 2020 + Q1 2021
    assert via_alias.equals(via_native)


def test_period_alias_annual_matches_1y(year_span_data):
    """'annual' must resolve to the Polars interval '1y'."""
    via_alias = year_span_data.utils.group_returns(period="annual")
    via_native = year_span_data.utils.group_returns(period="1y")
    assert via_alias.height == 2  # 2020 and 2021
    assert via_alias.equals(via_native)


def test_period_alias_yearly_matches_1y(year_span_data):
    """'yearly' must resolve to the Polars interval '1y'."""
    via_alias = year_span_data.utils.group_returns(period="yearly")
    via_native = year_span_data.utils.group_returns(period="1y")
    assert via_alias.height == 2
    assert via_alias.equals(via_native)


# ─── __slots__ ────────────────────────────────────────────────────────────────


def test_data_utils_rejects_undeclared_attributes(varied_data):
    """DataUtils declares __slots__, so undeclared attributes must raise AttributeError."""
    utils = varied_data.utils
    with pytest.raises(AttributeError):
        utils.not_a_declared_slot = 123


# ─── to_prices / to_log_returns null handling ────────────────────────────────


def test_to_prices_null_return_keeps_price_flat():
    """A null return must be treated as 0.0 (price unchanged), not as +100 %."""
    data = _make_data({"A": [0.02, None, 0.01]})
    prices = data.utils.to_prices(base=100.0)
    assert prices["A"][0] == pytest.approx(102.0, abs=TOL_FLOAT64)
    assert prices["A"][1] == pytest.approx(102.0, abs=TOL_FLOAT64)
    assert prices["A"][2] == pytest.approx(103.02, abs=TOL_FLOAT64)


def test_to_log_returns_null_maps_to_zero():
    """A null return must become log(1 + 0) = 0.0, not log(1 + 1)."""
    data = _make_data({"A": [0.02, None, 0.01]})
    log_rets = data.utils.to_log_returns()
    assert log_rets["A"][0] == pytest.approx(math.log(1.02), abs=TOL_FLOAT64)
    assert log_rets["A"][1] == pytest.approx(0.0, abs=TOL_FLOAT64)


# ─── to_volatility_adjusted_returns ──────────────────────────────────────────


def test_volatility_adjusted_default_window_is_60(varied_data):
    """Calling with no arguments must equal an explicit window=60 call elementwise."""
    default = varied_data.utils.to_volatility_adjusted_returns()
    explicit = varied_data.utils.to_volatility_adjusted_returns(window=60)
    assert default.equals(explicit)


def test_volatility_adjusted_divides_by_lagged_vol():
    """The return must be divided (not multiplied) by the lagged rolling std."""
    data = _make_data({"A": [0.01, 0.03, 0.02, 0.01]})
    result = data.utils.to_volatility_adjusted_returns(window=2)
    expected = 0.02 / statistics.stdev([0.01, 0.03])  # r_2 / std(r_0, r_1)
    assert result["A"][2] == pytest.approx(expected, rel=TOL_FLOAT64)


# ─── rebase ───────────────────────────────────────────────────────────────────


def test_rebase_default_base_is_100(varied_data):
    """rebase() with no arguments must anchor the series at exactly 100.0."""
    rebased = varied_data.utils.rebase()
    assert float(rebased["A"][0]) == pytest.approx(100.0, abs=TOL_FLOAT64)


def test_rebase_survives_huge_price_levels():
    """Rebase must stay finite near float64 max.

    The internal price conversion uses base 1.0; any larger internal base
    would push a price level of ~1e308 into overflow (inf/inf = NaN).
    """
    data = _make_data({"A": [1e308, 0.0]})
    rebased = data.utils.rebase(base=100.0)
    assert math.isfinite(rebased["A"][0])
    assert rebased["A"][0] == pytest.approx(100.0, abs=TOL_FLOAT64)
    assert rebased["A"][1] == pytest.approx(100.0, abs=TOL_FLOAT64)


# ─── winsorise ────────────────────────────────────────────────────────────────


def test_winsorise_default_window_and_sigma():
    """winsorise() must equal winsorise(window=7, n_sigma=3.0) elementwise.

    The data contains an outlier whose clipped value depends on both the
    rolling window length and the sigma multiplier, so any perturbation of
    either default changes the output.
    """
    vals = [(((i * 53) % 17) - 8) / 1000.0 for i in range(20)]
    vals[12] = 0.8  # outlier clipped against the rolling bounds
    data = _make_data({"A": vals})
    default = data.utils.winsorise()
    explicit = data.utils.winsorise(window=7, n_sigma=3.0)
    assert default["A"][12] < 0.8  # the outlier is actually clipped
    assert default.equals(explicit)


def test_winsorise_clips_negative_outlier_to_exact_lower_bound():
    """A negative outlier must be clipped to exactly mean - n_sigma * std.

    The bounds at row 7 are the rolling mean/std of rows 0-6 (window=7,
    lagged by one row to avoid look-ahead).
    """
    head = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.0]
    vals = [*head, -0.6, 0.01, -0.01]
    data = _make_data({"A": vals})
    result = data.utils.winsorise(window=7, n_sigma=3.0)
    expected_lower = statistics.mean(head) - 3.0 * statistics.stdev(head)
    assert result["A"][7] == pytest.approx(expected_lower, abs=TOL_FLOAT64)


# ─── group_returns / aggregate_returns ───────────────────────────────────────


def test_group_returns_default_period_is_monthly(year_span_data):
    """group_returns() with no arguments must aggregate monthly ('1mo')."""
    default = year_span_data.utils.group_returns()
    explicit = year_span_data.utils.group_returns(period="1mo")
    assert default.height == 14  # Jan 2020 .. Feb 2021
    assert default.equals(explicit)


def test_aggregate_returns_default_period_is_monthly(year_span_data):
    """aggregate_returns() with no arguments must aggregate monthly ('1mo')."""
    default = year_span_data.utils.aggregate_returns()
    explicit = year_span_data.utils.aggregate_returns(period="1mo")
    assert default.equals(explicit)


def test_group_returns_integer_index_error_message(int_index_data):
    """The error for integer-indexed data must name the group_returns method."""
    with pytest.raises(
        MissingDateColumnError,
        match=r"^DataFrame 'group_returns' is missing the required 'date' column\.$",
    ):
        int_index_data.utils.group_returns()


def test_group_returns_compounded_with_nulls_exact_value(month_null_data):
    """Compounded aggregation must be prod(1 + r) - 1 with nulls treated as 0."""
    grouped = month_null_data.utils.group_returns(period="1mo", compounded=True)
    expected = 1.01 * 1.0 * 1.02 * 0.995 * 1.0 - 1.0
    assert grouped.height == 1
    assert grouped["A"][0] == pytest.approx(expected, abs=TOL_FLOAT64)


def test_group_returns_sum_with_nulls_exact_value(month_null_data):
    """Non-compounded aggregation must sum the returns with nulls treated as 0."""
    grouped = month_null_data.utils.group_returns(period="1mo", compounded=False)
    expected = 0.01 + 0.0 + 0.02 - 0.005 + 0.0
    assert grouped.height == 1
    assert grouped["A"][0] == pytest.approx(expected, abs=TOL_FLOAT64)


# ─── to_excess_returns ────────────────────────────────────────────────────────


def test_to_excess_returns_default_rf_is_zero(varied_data):
    """to_excess_returns() with no arguments must leave returns unchanged."""
    excess = varied_data.utils.to_excess_returns()
    assert excess["A"].to_list() == varied_data.returns["A"].to_list()


def test_to_excess_returns_nperiods_exact_deannualisation():
    """With nperiods, rf must be de-annualised as (1 + rf)^(1/nperiods) - 1."""
    data = _make_data({"A": [0.01, 0.02]})
    excess = data.utils.to_excess_returns(rf=0.05, nperiods=252)
    rf_per_period = (1.0 + 0.05) ** (1.0 / 252) - 1.0
    assert excess["A"][0] == pytest.approx(0.01 - rf_per_period, abs=TOL_FLOAT64)
    assert excess["A"][1] == pytest.approx(0.02 - rf_per_period, abs=TOL_FLOAT64)


# ─── exponential_stdev ────────────────────────────────────────────────────────


def test_exponential_stdev_defaults_span_30(varied_data):
    """exponential_stdev() must equal an explicit span-30 call elementwise."""
    default = varied_data.utils.exponential_stdev()
    explicit = varied_data.utils.exponential_stdev(window=30, is_halflife=False)
    assert default.equals(explicit)


def test_exponential_stdev_halflife_first_row_defined(varied_data):
    """min_samples=1 must yield a value already at the first row (half-life mode)."""
    result = varied_data.utils.exponential_stdev(window=5, is_halflife=True)
    assert result["A"][0] is not None


def test_exponential_stdev_span_first_row_defined(varied_data):
    """min_samples=1 must yield a value already at the first row (span mode)."""
    result = varied_data.utils.exponential_stdev(window=5, is_halflife=False)
    assert result["A"][0] is not None


def test_exponential_stdev_halflife_constant_returns_zero():
    """Half-life mode must compute an actual EWM std: zero for constant returns."""
    data = _make_data({"A": [0.01] * 30})
    result = data.utils.exponential_stdev(window=5, is_halflife=True)
    values = result["A"].to_list()
    assert len(values) == 30
    for v in values:
        assert v is not None
        assert abs(v) <= TOL_FLOAT64


# ─── exponential_cov ─────────────────────────────────────────────────────────


def test_exponential_cov_default_window_is_30(varied_two_asset_data):
    """exponential_cov() must equal an explicit window=30 call for every date."""
    default = varied_two_asset_data.utils.exponential_cov()
    explicit = varied_two_asset_data.utils.exponential_cov(window=30)
    assert list(default.keys()) == list(explicit.keys())
    for key, mat in default.items():
        assert np.array_equal(mat, explicit[key], equal_nan=True)


def test_exponential_cov_warmup_type_error_message(varied_data):
    """The TypeError for a float warmup must carry the exact message."""
    with pytest.raises(TypeError, match=r"^warmup must be an integer, got float$"):
        varied_data.utils.exponential_cov(warmup=1.5)


def test_exponential_cov_warmup_negative_error_message(varied_data):
    """The ValueError for a negative warmup must carry the exact message."""
    with pytest.raises(ValueError, match=r"^warmup must be a non-negative integer, got -3$"):
        varied_data.utils.exponential_cov(warmup=-3)


def test_exponential_cov_names_do_not_affect_values():
    """Covariance values must not depend on asset column names.

    Uses a column name shaped like an internal pair alias to ensure the
    pair-column naming scheme never collides with a real asset column.
    """
    values_a = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02]
    values_b = [0.005, 0.01, -0.005, 0.02, 0.015, -0.01]
    plain = _make_data({"A": values_a, "B": values_b})
    weird = _make_data({"A": values_a, "XXA_AXX": values_b})
    cov_plain = plain.utils.exponential_cov(window=3)
    cov_weird = weird.utils.exponential_cov(window=3)
    assert list(cov_plain.keys()) == list(cov_weird.keys())
    for key, mat in cov_plain.items():
        assert mat.shape == (2, 2)
        assert np.array_equal(mat, cov_weird[key], equal_nan=True)
