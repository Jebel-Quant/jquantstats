"""Tests for the shared ``truncate`` bound contract (issues #926, #927).

`Data.truncate` and `Portfolio.truncate` route through the same resolver, so the
contract is tested once against both. The rule under test: **the bound type picks
the axis, the index type says what is legal.**
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from jquantstats import Data, Portfolio
from jquantstats._truncate import resolve_bounds
from jquantstats.exceptions import (
    IntegerIndexBoundError,
    InvalidTruncateBoundError,
    JQuantStatsError,
    MixedTruncateBoundsError,
)

N = 80
START = date(2020, 1, 1)
DATES = [START + timedelta(days=i) for i in range(N)]

# The 11th row, so a date bound and the row index 10 select the same rows.
ROW_10_DATE = START + timedelta(days=10)


def _prices(dated: bool = True) -> pl.DataFrame:
    """Build a one-asset price frame, with or without a date column."""
    cols: dict[str, object] = {"A": pl.Series([100.0 + i for i in range(N)], dtype=pl.Float64)}
    return pl.DataFrame({"date": DATES, **cols} if dated else cols)


def _cash(dated: bool = True) -> pl.DataFrame:
    """Build a one-asset cash-position frame, with or without a date column."""
    cols: dict[str, object] = {"A": pl.Series([1000.0] * N, dtype=pl.Float64)}
    return pl.DataFrame({"date": DATES, **cols} if dated else cols)


@pytest.fixture
def dated_pf() -> Portfolio:
    """An 80-row Portfolio with a temporal date column."""
    return Portfolio.from_cash_position(prices=_prices(), cash_position=_cash(), aum=1e6)


@pytest.fixture
def int_pf() -> Portfolio:
    """An 80-row Portfolio with no date column."""
    return Portfolio.from_cash_position(prices=_prices(dated=False), cash_position=_cash(dated=False), aum=1e6)


@pytest.fixture
def dated_data() -> Data:
    """An 80-row Data with a temporal index."""
    return Data.from_returns(pl.DataFrame({"Date": DATES, "R": pl.Series([0.001] * N, dtype=pl.Float64)}))


@pytest.fixture
def int_data() -> Data:
    """An 80-row Data with an integer index."""
    return Data.from_returns(pl.DataFrame({"Date": list(range(N)), "R": pl.Series([0.001] * N, dtype=pl.Float64)}))


# ─── #926: string bounds are parsed ───────────────────────────────────────────
# The signature advertised `str` but `pl.lit` wrapped it unparsed, so Polars
# refused to compare Utf8 against a temporal column.


@pytest.mark.parametrize("bound", ["2020-01-11", "2020-01-11T00:00:00"])
def test_iso_string_start_matches_the_equivalent_date(dated_pf, dated_data, bound):
    """An ISO-8601 string selects exactly what the equivalent date object selects."""
    assert dated_pf.truncate(start=bound).prices.height == dated_pf.truncate(start=ROW_10_DATE).prices.height == 70
    assert dated_data.truncate(start=bound).returns.height == 70


def test_iso_string_end_and_both_bounds(dated_pf):
    """String bounds work as an upper bound and as a closed range."""
    assert dated_pf.truncate(end="2020-01-10").prices.height == 10
    assert dated_pf.truncate(start="2020-01-11", end="2020-01-20").prices.height == 10


def test_unparseable_string_raises_a_typed_error(dated_pf, dated_data):
    """A non-ISO string is reported by the library, not by Polars."""
    with pytest.raises(InvalidTruncateBoundError, match="ISO-8601"):
        dated_pf.truncate(start="last tuesday")
    with pytest.raises(InvalidTruncateBoundError, match="ISO-8601"):
        dated_data.truncate(end="not a date")


# ─── #927: integer bounds slice positionally on a temporal axis ───────────────
# Routing used to be on index dtype, so an int bound was compared against the
# date column, produced an all-true mask, and truncated nothing.


def test_integer_bound_slices_a_dated_object(dated_pf, dated_data):
    """An int bound is a row index even when the index is temporal."""
    assert dated_pf.truncate(start=10).prices.height == 70
    assert dated_data.truncate(start=10).returns.height == 70


def test_integer_bounds_agree_with_the_equivalent_dates(dated_pf):
    """Row 10 and the date on row 10 select the same rows — the old bug's tell."""
    by_row = dated_pf.truncate(start=10)
    by_date = dated_pf.truncate(start=ROW_10_DATE)
    assert by_row.prices.equals(by_date.prices)


def test_integer_range_and_end_only(dated_pf):
    """Row indices are inclusive on both ends."""
    assert dated_pf.truncate(start=10, end=19).prices.height == 10
    assert dated_pf.truncate(end=9).prices.height == 10


def test_inverted_integer_range_is_empty_not_negative(dated_pf):
    """An inverted range clamps to an empty frame rather than a negative slice."""
    assert dated_pf.truncate(start=50, end=10).prices.height == 0


# ─── Mixing the two axes is rejected ──────────────────────────────────────────
# Previously the int was ignored and the date silently applied, so the caller got
# a range they never asked for.


def test_mixed_bounds_raise(dated_pf, dated_data):
    """A row index paired with a date names two different axes."""
    with pytest.raises(MixedTruncateBoundsError, match="start as a row index and end as a date"):
        dated_pf.truncate(start=10, end=ROW_10_DATE)
    with pytest.raises(MixedTruncateBoundsError, match="end as a row index and start as a date"):
        dated_data.truncate(start=ROW_10_DATE, end=20)


# ─── Unsupported bound types ──────────────────────────────────────────────────


@pytest.mark.parametrize("bound", [3.5, object(), (1, 2)])
def test_unsupported_type_on_a_temporal_axis_raises(dated_pf, bound):
    """A value that is neither a row index nor a date is refused."""
    with pytest.raises(InvalidTruncateBoundError):
        dated_pf.truncate(start=bound)


def test_bool_is_not_accepted_as_a_row_index(dated_pf, int_pf):
    """``bool`` subclasses ``int``, so it would otherwise mean row 1 silently."""
    with pytest.raises(InvalidTruncateBoundError):
        dated_pf.truncate(start=True)
    with pytest.raises(IntegerIndexBoundError, match="got bool"):
        int_pf.truncate(start=True)


# ─── Preserved: an integer-indexed object accepts row indices only ────────────


def test_integer_indexed_rejects_non_integer_bounds(int_pf, int_data):
    """Unchanged behaviour, including the message, for a non-temporal index."""
    with pytest.raises(IntegerIndexBoundError, match="start must be an integer, got str"):
        int_pf.truncate(start="2020-01-01")
    with pytest.raises(IntegerIndexBoundError, match="end must be an integer, got float"):
        int_pf.truncate(end=3.5)
    with pytest.raises(IntegerIndexBoundError, match="start must be an integer, got date"):
        int_data.truncate(start=START)


def test_integer_indexed_slices_and_passes_through(int_pf):
    """Row bounds and no bounds both behave as before on an integer index."""
    assert int_pf.truncate(start=10).prices.height == 70
    assert int_pf.truncate(start=10, end=19).prices.height == 10
    assert int_pf.truncate().prices.height == N


# ─── Unchanged behaviour on the date axis ─────────────────────────────────────


def test_date_bounds_and_no_bounds(dated_pf, dated_data):
    """Date objects and the no-bound case are untouched by the new routing."""
    assert dated_pf.truncate().prices.height == N
    assert dated_data.truncate().returns.height == N
    assert dated_pf.truncate(start=START, end=START + timedelta(days=9)).prices.height == 10
    assert dated_pf.truncate(start=datetime(2020, 1, 11)).prices.height == 70


def test_truncate_preserves_aum_and_benchmark(dated_pf, dated_data):
    """Truncation carries construction state across, as it always did."""
    assert dated_pf.truncate(start=10).aum == dated_pf.aum
    assert dated_data.truncate(start=10).benchmark is None


# ─── The resolver in isolation ────────────────────────────────────────────────


def test_resolve_bounds_reports_the_mode():
    """The resolver names the axis it chose and coerces string bounds."""
    assert resolve_bounds(None, None, temporal=True) == ("none", None, None)
    assert resolve_bounds(None, None, temporal=False) == ("none", None, None)
    assert resolve_bounds(5, None, temporal=True) == ("rows", 5, None)
    assert resolve_bounds(None, 5, temporal=True) == ("rows", None, 5)
    assert resolve_bounds("2020-01-11", None, temporal=True) == ("dates", date(2020, 1, 11), None)
    assert resolve_bounds(None, "2020-01-11T06:30:00", temporal=True) == (
        "dates",
        None,
        datetime(2020, 1, 11, 6, 30),
    )


def test_new_exceptions_are_part_of_the_domain_family():
    """Both new errors are catchable via ``except JQuantStatsError``."""
    assert issubclass(InvalidTruncateBoundError, JQuantStatsError)
    assert issubclass(MixedTruncateBoundsError, JQuantStatsError)
    # And keep their builtin flavours, so existing ``except ValueError`` still works.
    assert issubclass(InvalidTruncateBoundError, ValueError)
    assert issubclass(MixedTruncateBoundsError, TypeError)


def test_exception_payloads_are_introspectable():
    """The errors carry the offending parameter and value for programmatic use."""
    err = InvalidTruncateBoundError("start", "last tuesday")
    assert err.param == "start"
    assert err.value == "last tuesday"

    mixed = MixedTruncateBoundsError("start", "end")
    assert mixed.row_param == "start"
    assert mixed.date_param == "end"
