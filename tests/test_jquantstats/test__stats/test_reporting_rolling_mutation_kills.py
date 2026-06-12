"""Mutation-kill tests for `_stats/_reporting.py` and `_stats/_rolling.py`.

Targets the surviving mutmut mutants recorded in
``.mutation-sweep/D2.json`` (key ``src/jquantstats/_stats/_reporting.py``) and
``.mutation-sweep/F2.json`` (key ``src/jquantstats/_stats/_rolling.py``).

Strategy: exact-value pins on shared code paths so that one test kills many
mutants.  Expected values mirror the implementation's arithmetic step by step
(same operations, same order) so they hold to float64 round-off, while every
targeted mutant moves the result by many orders of magnitude more than the
assertion tolerance.  Error-message tests use fully anchored regexes so the
``XX…XX`` string mutants fail.  Comments inside each test name the mutant ids
it kills.
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jquantstats.data import Data

from ..tolerances import TOL_COMPOUNDING, TOL_FLOAT64, TOL_PINNED

# ── Helpers ───────────────────────────────────────────────────────────────────


def _days(n: int, start: date = date(2023, 1, 2)) -> list[date]:
    """Return *n* consecutive calendar days starting at *start*."""
    return [start + timedelta(days=i) for i in range(n)]


def _stats_for(columns, *, dates=None, idx=None, benchmark=None):
    """Build a Stats object from raw return columns and a date or integer index."""
    returns = pl.DataFrame(columns)
    if dates is not None:
        index = pl.DataFrame({"Date": dates})
    else:
        index = pl.DataFrame({"idx": list(range(returns.height)) if idx is None else idx})
    bench = None if benchmark is None else pl.DataFrame(benchmark)
    return Data(returns=returns, index=index, benchmark=bench).stats


# ── _reporting: avg_drawdown ──────────────────────────────────────────────────


def test_avg_drawdown_averages_only_underwater_periods():
    """avg_drawdown must average strictly positive drawdowns only.

    Kills _reporting mutants 4 (``dd > 0`` -> ``dd >= 0``: zeros dilute the
    mean) and 5 (``dd > 0`` -> ``dd > 1``: filter empties, returning 0.0).
    """
    rets = [0.0, 0.10, -0.10, 0.05, 0.07]
    stats = _stats_for({"A": rets}, dates=_days(5))

    # Mirror the implementation: compound NAV, running high-water mark,
    # drawdown fractions, then the mean over underwater rows only.
    nav, acc = [], 1.0
    for r in rets:
        acc *= 1.0 + r
        nav.append(acc)
    hwm = [max(nav[: i + 1]) for i in range(len(nav))]
    dd = [(h - v) / h for h, v in zip(hwm, nav, strict=True)]
    underwater = [x for x in dd if x > 0.0]
    assert len(underwater) == 2  # guard: 2 underwater rows and 3 at-peak rows
    expected = -sum(underwater) / len(underwater)

    assert stats.avg_drawdown()["A"] == pytest.approx(expected, abs=TOL_PINNED)


# ── _reporting: cagr ──────────────────────────────────────────────────────────


def test_cagr_deannualises_risk_free_rate():
    """Cagr must subtract the per-period risk-free rate ``rf / periods``.

    Kills _reporting mutants 18 (``- rf`` -> ``+ rf``) and 19
    (``rf / raw_periods`` -> ``rf * raw_periods``).
    """
    rets = [0.01, 0.02, -0.005, 0.015, 0.0, 0.01]
    stats = _stats_for({"A": rets}, dates=_days(6))
    rf, periods = 0.0252, 252

    excess = [r - rf / periods for r in rets]
    total = math.prod(1.0 + e for e in excess) - 1.0
    years = len(rets) / periods
    expected = abs(1.0 + total) ** (1.0 / years) - 1.0

    assert stats.cagr(rf=rf, periods=periods)["A"] == pytest.approx(expected, rel=TOL_PINNED)


# ── _reporting: expected_return ───────────────────────────────────────────────

# Jan has two returns (product != sum) and Feb one, so compounded, summed and
# raw-geomean results are all measurably different.
_ER_DATES = [date(2023, 1, 10), date(2023, 1, 20), date(2023, 2, 10)]
_ER_RETS = [0.1, 0.1, 0.05]


def test_expected_return_monthly_compounds_by_default():
    """expected_return(aggregate=...) must compound within calendar months by default.

    Kills _reporting mutants 31 (``compounded=True`` default -> ``False``),
    59/60 (date-column / temporal-check logic: both fall back to the raw
    geometric mean, which differs), and 65 (``(1 + ret).product()`` ->
    ``(1 - ret).product()``).
    """
    stats = _stats_for({"A": _ER_RETS}, dates=_ER_DATES)
    jan = (1.0 + 0.1) * (1.0 + 0.1) - 1.0
    expected = ((1.0 + jan) * (1.0 + 0.05)) ** 0.5 - 1.0

    assert stats.expected_return(aggregate="monthly")["A"] == pytest.approx(expected, abs=TOL_PINNED)


def test_expected_return_monthly_summed_aggregation():
    """expected_return(compounded=False) must sum returns within each month.

    Kills _reporting mutant 69 (``pl.col("ret")`` -> ``pl.col("XXretXX")`` in
    the sum branch, which raises ColumnNotFound when executed).
    """
    stats = _stats_for({"A": _ER_RETS}, dates=_ER_DATES)
    expected = ((1.0 + (0.1 + 0.1)) * (1.0 + 0.05)) ** 0.5 - 1.0

    result = stats.expected_return(aggregate="monthly", compounded=False)
    assert result["A"] == pytest.approx(expected, abs=TOL_PINNED)


def test_expected_return_rejects_unknown_aggregate():
    """The aggregate validation error must render the exact frequency list.

    Kills _reporting mutant 56 (XX-wrapped message).
    """
    stats = _stats_for({"A": _ER_RETS}, dates=_ER_DATES)
    msg = re.escape("aggregate must be one of ['weekly', 'monthly', 'quarterly', 'annual', 'yearly'], got 'bogus'")
    with pytest.raises(ValueError, match=f"^{msg}$"):
        stats.expected_return(aggregate="bogus")


def test_expected_return_integer_index_falls_back_to_raw_geomean():
    """With a non-temporal index, aggregation falls back to the raw geometric mean.

    Documents the integer-index branch of the ``date_col_name is None or not
    temporal`` guard (the dated test above kills the logic mutants 59/60).
    """
    stats = _stats_for({"A": _ER_RETS})  # integer index 0..2
    expected = ((1.0 + 0.1) * (1.0 + 0.1) * (1.0 + 0.05)) ** (1.0 / 3.0) - 1.0

    assert stats.expected_return(aggregate="monthly")["A"] == pytest.approx(expected, abs=TOL_PINNED)


# ── _reporting: max_drawdown_duration ─────────────────────────────────────────

# Additive NAV (1 + cumsum) gives in-drawdown rows 1..3 and row 6; the
# weekend gap inside the first run makes calendar days (5) differ from row
# counts (3).
_DD_RETS = [0.01, -0.01, -0.01, 0.005, 0.02, 0.0, -0.005]
_DD_DATES = [
    date(2023, 1, 5),  # Thu
    date(2023, 1, 6),  # Fri
    date(2023, 1, 9),  # Mon (weekend gap inside the drawdown run)
    date(2023, 1, 10),
    date(2023, 1, 11),
    date(2023, 1, 12),
    date(2023, 1, 13),
]


def test_max_drawdown_duration_counts_calendar_days():
    """Temporal indices must measure drawdown duration in calendar days.

    The longest underwater run spans Jan 6 - Jan 10 (5 calendar days, only 3
    rows).  Kills _reporting mutants 112/113/115 (date-column and ``has_date``
    logic: all fall back to row counting -> 3), 120 (``1.0 - cum_sum`` flips
    the run -> 4), 123 (``nav <= hwm`` makes every row underwater -> 9),
    126 (no-drawdown result ``0`` -> ``1`` on the FLAT column), 128
    (``continue`` -> ``break`` drops the DD column entirely), and 150/151
    (``total_days() + 1`` -> ``- 1`` / ``+ 2``).
    """
    stats = _stats_for({"FLAT": [0.01, 0.0, 0.02, 0.0, 0.01, 0.0, 0.03], "DD": _DD_RETS}, dates=_DD_DATES)
    result = stats.max_drawdown_duration()
    assert result["FLAT"] == 0  # never underwater (kills 126; setup for 128)
    assert result["DD"] == 5  # Jan 6 -> Jan 10 inclusive


def test_max_drawdown_duration_integer_index_counts_rows():
    """Integer indices must count rows via ``range(len)``, not index values.

    The non-contiguous index [0, 10, ...] distinguishes row positions from
    index values.  Kills _reporting mutants 130 (``and`` -> ``or`` routes the
    raw integer index into the frame -> 21), 155 (``end + start + 1`` -> 5),
    157 (``- 1`` -> 1), and 158 (``+ 2`` -> 4).
    """
    stats = _stats_for({"DD": _DD_RETS}, idx=[0, 10, 20, 30, 40, 50, 60])
    assert stats.max_drawdown_duration()["DD"] == 3  # rows 1..3


# ── _reporting: monthly_win_rate ──────────────────────────────────────────────


def test_monthly_win_rate_counts_strictly_positive_months():
    """Only strictly positive compounded months count as wins.

    Jan is positive, Feb compounds to exactly 0.0, Mar is negative -> 1/3.
    Kills _reporting mutants 176 (``(col + 1).product()`` -> ``(col - 1)``:
    every month flips non-positive -> 0.0) and 190 (``> 0`` -> ``>= 0``: the
    zero month counts -> 2/3).
    """
    dates = [date(2023, 1, 5), date(2023, 1, 15), date(2023, 2, 5), date(2023, 2, 15), date(2023, 3, 5)]
    stats = _stats_for({"A": [0.1, 0.1, 0.0, 0.0, -0.05]}, dates=dates)
    assert stats.monthly_win_rate()["A"] == pytest.approx(1.0 / 3.0, abs=TOL_FLOAT64)


def test_monthly_win_rate_is_nan_for_all_null_asset():
    """An all-null asset has zero months and must yield ``float("nan")``.

    Kills _reporting mutants 186 (``n_total == 0`` -> ``== 1``: falls through
    to ``0 / 0`` -> ZeroDivisionError), 187 (``float("XXnanXX")`` raises
    ValueError), and 188 (``None`` instead of NaN -> ``math.isnan`` raises).
    """
    nulls = pl.Series("N", [None, None, None], dtype=pl.Float64)
    stats = _stats_for({"N": nulls}, dates=_days(3))
    assert math.isnan(stats.monthly_win_rate()["N"])


# ── _reporting: monthly_returns ───────────────────────────────────────────────


def test_monthly_returns_default_compounds_and_includes_eoy():
    """Defaults must be ``eoy=True`` and ``compounded=True``; absent months fill 0.0.

    Kills _reporting mutants 195 (``eoy=True`` -> ``False``: no EOY column),
    196 (``compounded=True`` -> ``False``: JAN becomes 0.2 instead of the
    compounded 0.21), and 256 (missing-month fill ``lit(0.0)`` -> ``lit(1.0)``).
    """
    stats = _stats_for({"A": _ER_RETS}, dates=_ER_DATES)
    mr = stats.monthly_returns()["A"]

    assert "EOY" in mr.columns  # 195
    jan = (1.0 + 0.1) * (1.0 + 0.1) - 1.0
    assert mr["JAN"][0] == pytest.approx(jan, abs=TOL_PINNED)  # 196
    assert mr["DEC"][0] == 0.0  # 256
    eoy = (1.0 + 0.1) * (1.0 + 0.1) * (1.0 + 0.05) - 1.0
    assert mr["EOY"][0] == pytest.approx(eoy, abs=TOL_PINNED)


def test_monthly_returns_summed_aggregation():
    """``compounded=False`` must sum returns within each month.

    Kills _reporting mutant 240 (``pl.col("XXretXX").sum()`` raises
    ColumnNotFound when the sum branch executes).
    """
    stats = _stats_for({"A": _ER_RETS}, dates=_ER_DATES)
    mr = stats.monthly_returns(compounded=False)["A"]
    assert mr["JAN"][0] == pytest.approx(0.1 + 0.1, abs=TOL_PINNED)


# ── _reporting: distribution ──────────────────────────────────────────────────


def test_distribution_iqr_split_pins_fence_boundaries():
    """The IQR split must keep points exactly on both 1.5*IQR fences as inliers.

    All values are dyadic so quantiles and fences are exact: q1 = -1/64,
    q3 = 1/64, IQR = 1/32, fences at -1/16 and +1/16, with one point exactly
    on each fence and one strict outlier on each side.  Kills _reporting
    mutants 290 (``q3 + q1``), 292 (``>=`` -> ``>`` at the lower fence), 293
    (lower bound ``q1 + 1.5*iqr``), 294 (``2.5 *``), 295 (``1.5 /``), 296
    (``&`` -> ``|``: outliers vanish), 297 (``<=`` -> ``<`` at the upper
    fence), 298 (upper bound ``q3 - 1.5*iqr``), 299 (``2.5 *``), and 300
    (``1.5 /``).
    """
    rets = [-0.078125, -0.0625, -0.015625, -0.0078125, 0.0, 0.0078125, 0.015625, 0.0625, 0.078125]
    stats = _stats_for({"A": rets}, dates=_days(9))
    split = stats.distribution()["A"]["Daily"]

    assert split["values"] == [-0.0625, -0.015625, -0.0078125, 0.0, 0.0078125, 0.015625, 0.0625]
    assert split["outliers"] == [-0.078125, 0.078125]


def test_distribution_compounds_by_default():
    """distribution() must compound within periods by default.

    Kills _reporting mutant 273 (``compounded=True`` -> ``False``: the
    two-return January aggregates to 0.2 instead of 0.21).
    """
    stats = _stats_for({"A": _ER_RETS}, dates=_ER_DATES)
    monthly = stats.distribution()["A"]["Monthly"]
    jan = (1.0 + 0.1) * (1.0 + 0.1) - 1.0
    values = sorted(monthly["values"] + monthly["outliers"])
    assert values == pytest.approx([0.05, jan], abs=TOL_COMPOUNDING)


def test_distribution_summed_aggregation():
    """distribution(compounded=False) must sum returns within periods.

    Kills _reporting mutant 282 (``pl.col("XXretXX").sum()`` raises
    ColumnNotFound when the sum branch executes).
    """
    stats = _stats_for({"A": _ER_RETS}, dates=_ER_DATES)
    monthly = stats.distribution(compounded=False)["A"]["Monthly"]
    values = sorted(monthly["values"] + monthly["outliers"])
    assert values == pytest.approx([0.05, 0.1 + 0.1], abs=TOL_COMPOUNDING)


# ── _reporting: compare ───────────────────────────────────────────────────────

# Two rows each in Dec-2022, Jan-2023 (asset returns null) and Apr-2023.  The
# all-null January separates the asset and benchmark period sets, which makes
# the full-join ``coalesce=True`` observable through the sort order.
_CMP_DATES = [
    date(2022, 12, 26),  # Mon
    date(2022, 12, 27),
    date(2023, 1, 2),  # Mon
    date(2023, 1, 3),
    date(2023, 4, 3),  # Mon
    date(2023, 4, 4),
]
_CMP_RETS = [0.1, 0.1, None, None, 0.05, 0.05]
_CMP_BENCH = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]


def _compare_stats():
    """Stats fixture data for the compare() aggregation tests."""
    return _stats_for({"A": _CMP_RETS}, dates=_CMP_DATES, benchmark={"B": _CMP_BENCH})


def test_compare_requires_benchmark_message():
    """compare() without a benchmark must raise the exact AttributeError message.

    Kills _reporting mutant 335 (XX-wrapped message).
    """
    stats = _stats_for({"A": [0.01, -0.02, 0.03]}, dates=_days(3))
    with pytest.raises(AttributeError, match=r"^No benchmark data available$"):
        stats.compare()


def test_compare_monthly_aggregation_pins_compounded_values():
    """compare(aggregate="ME") must compound by month and coalesce join keys.

    Kills _reporting mutants 333 (``compounded=True`` default -> ``False``:
    December becomes 20.0 instead of 21.0), 341 (``"ME"`` key XX-mutant falls
    through to 6 daily rows), 350-353 (compounded aggregation expression
    constants), 359-360 (aggregate-condition logic falls through to daily
    rows), and 375 (``coalesce=True`` -> ``False``: the null-keyed January row
    sorts first, reordering the Benchmark column).
    """
    out = _compare_stats().compare(aggregate="ME")["A"]

    assert out.height == 3  # 341, 359, 360 (daily fallback would give 6)
    dec_ret = ((1.0 + 0.1) * (1.0 + 0.1) - 1.0) * 100
    assert out["Returns"][0] == pytest.approx(dec_ret, abs=TOL_COMPOUNDING)  # 333, 350-353
    assert out["Returns"][1] is None  # January asset returns are all null
    dec_bench = ((1.0 + 0.01) * (1.0 + 0.02) - 1.0) * 100
    jan_bench = ((1.0 + 0.03) * (1.0 + 0.04) - 1.0) * 100
    assert out["Benchmark"][0] == pytest.approx(dec_bench, abs=TOL_COMPOUNDING)  # 375
    assert out["Benchmark"][1] == pytest.approx(jan_bench, abs=TOL_COMPOUNDING)


def test_compare_quarterly_yearly_weekly_aggregate_row_counts():
    """Every _freq_map entry must map its key to the right truncation period.

    Kills _reporting mutants 343/345/347 (key XX-mutants fall through to 6
    daily rows) and 344/346/348 (value XX-mutants make ``dt.truncate`` raise).
    """
    stats = _compare_stats()
    assert stats.compare(aggregate="QE")["A"].height == 3  # 2022Q4, 2023Q1, 2023Q2
    assert stats.compare(aggregate="YE")["A"].height == 2  # 2022, 2023
    assert stats.compare(aggregate="W")["A"].height == 3  # three Monday-anchored weeks


def test_compare_unknown_aggregate_falls_back_to_daily_rows():
    """An unknown aggregate string must fall back to daily rows, not KeyError.

    Kills _reporting mutant 361 (``and`` -> ``or`` short-circuits into
    ``_freq_map["bogus"]`` -> KeyError).
    """
    assert _compare_stats().compare(aggregate="bogus")["A"].height == 6


def test_compare_daily_multiplier_and_won_flags():
    """Daily compare() must pin Multiplier division, zero-guard, and Won ties.

    Kills _reporting mutants 397 (``Returns / Benchmark`` -> ``*``), 399
    (``replace(0.0, None)`` -> ``replace(1.0, None)``: division by zero gives
    inf instead of null), 402 (``>=`` -> ``>``: the tied row flips to "-"),
    and 404/405 (XX-wrapped "+" / "-" literals).
    """
    stats = _stats_for(
        {"A": [0.02, 0.01, 0.03, 0.05]},
        dates=_days(4),
        benchmark={"B": [0.02, 0.02, 0.0, 0.04]},
    )
    out = stats.compare()["A"]

    assert out["Won"].to_list() == ["+", "-", "+", "+"]  # 402 (tie), 404, 405
    assert out["Multiplier"][0] == pytest.approx(1.0, abs=TOL_FLOAT64)
    assert out["Multiplier"][1] == pytest.approx((0.01 * 100) / (0.02 * 100), abs=TOL_FLOAT64)  # 397
    assert out["Multiplier"][2] is None  # 399: zero benchmark -> null, not inf
    assert out["Multiplier"][3] == pytest.approx((0.05 * 100) / (0.04 * 100), abs=TOL_FLOAT64)


# ── _reporting: worst_n_periods ───────────────────────────────────────────────


def test_worst_n_periods_defaults_to_five():
    """worst_n_periods() must return exactly the 5 smallest returns by default.

    Kills _reporting mutant 414 (``n=5`` default -> ``6``).
    """
    stats = _stats_for({"A": [0.03, -0.02, 0.01, -0.05, 0.04, -0.01, 0.0]}, dates=_days(7))
    assert stats.worst_n_periods()["A"] == [-0.05, -0.02, -0.01, 0.0, 0.01]


# ── _reporting: capture ratios ────────────────────────────────────────────────


def test_up_capture_pins_geometric_mean_ratio():
    """up_capture must divide the strategy's up-market geometric mean by the benchmark's.

    Kills _reporting mutants 430/431/433 (benchmark geometric-mean formula),
    445/446/448 (strategy geometric-mean formula), and 451 (``/`` -> ``*``).
    """
    stats = _stats_for({"A": [0.10, -0.04, 0.06, -0.02]}, dates=_days(4))
    bench = pl.Series([0.05, -0.03, 0.02, -0.01])

    strat_geom = ((1.0 + 0.10) * (1.0 + 0.06)) ** 0.5 - 1.0
    bench_geom = ((1.0 + 0.05) * (1.0 + 0.02)) ** 0.5 - 1.0
    assert stats.up_capture(bench)["A"] == pytest.approx(strat_geom / bench_geom, abs=TOL_PINNED)


def test_up_capture_with_unit_benchmark_geometric_mean():
    """A benchmark up-market geometric mean of exactly 1.0 is a valid divisor.

    Two +100% benchmark periods give ``bench_geom == 1.0`` exactly (4.0**0.5
    is exact).  Kills _reporting mutant 437 (``bench_geom == 0.0`` ->
    ``== 1.0``, which would return NaN here).
    """
    stats = _stats_for({"A": [0.5, -0.1, 0.125]}, dates=_days(3))
    bench = pl.Series([1.0, -0.5, 1.0])

    result = stats.up_capture(bench)["A"]
    expected = ((1.0 + 0.5) * (1.0 + 0.125)) ** 0.5 - 1.0  # divided by bench_geom == 1.0
    assert not math.isnan(result)
    assert result == pytest.approx(expected, abs=TOL_PINNED)


def test_down_capture_pins_geometric_mean_ratio():
    """down_capture must divide the strategy's down-market geometric mean by the benchmark's.

    Kills _reporting mutants 458-464 (benchmark geometric-mean formula),
    473/475-479 (strategy geometric-mean formula), and 481 (``/`` -> ``*``).
    """
    stats = _stats_for({"A": [0.10, -0.04, 0.06, -0.02]}, dates=_days(4))
    bench = pl.Series([0.05, -0.03, 0.02, -0.01])

    strat_geom = ((1.0 + -0.04) * (1.0 + -0.02)) ** 0.5 - 1.0
    bench_geom = ((1.0 + -0.03) * (1.0 + -0.01)) ** 0.5 - 1.0
    assert stats.down_capture(bench)["A"] == pytest.approx(strat_geom / bench_geom, abs=TOL_PINNED)


# ── _reporting: annual_breakdown ──────────────────────────────────────────────


def test_annual_breakdown_integer_chunks_start_at_row_zero():
    """Integer-index chunking must start at row 0, keep 63-row tails, and carry the benchmark.

    315 rows -> one full 252-row chunk plus a 63-row tail; the tail sits
    exactly on the ``max(5, 252 // 4)`` threshold so it must be kept.  Row 0
    holds the global maximum so the year-1 ``best`` pins the chunk boundary.
    Kills _reporting mutants 493 (``range(0, ...)`` -> ``range(1, ...)``),
    496 (``<`` -> ``<=`` drops the 63-row tail), and 504
    (``chunk_benchmark`` forced to ``None`` drops the benchmark column).
    """
    rng = np.random.default_rng(7)
    vals = rng.uniform(-0.05, 0.05, 315)
    vals[0] = 0.5  # unique global maximum, only in the [0, 252) chunk
    bench_vals = rng.uniform(-0.05, 0.05, 315)
    stats = _stats_for({"A": vals}, idx=list(range(315)), benchmark={"B": bench_vals})

    bd = stats.annual_breakdown()
    assert set(bd["year"].to_list()) == {1, 2}  # 496
    assert "B" in bd.columns  # 504
    best = bd.filter((pl.col("year") == 1) & (pl.col("metric") == "best"))["A"][0]
    assert best == pytest.approx(0.5, abs=TOL_FLOAT64)  # 493


def test_annual_breakdown_integer_skips_short_tail_chunk():
    """Integer-index tails below ``max(5, chunk // 4)`` rows must be skipped.

    312 rows leave a 60-row tail: 60 < 63 so only chunk 1 survives.  Kills
    _reporting mutant 499 (``chunk // 4`` -> ``chunk // 5``: threshold 50
    would keep the tail as year 2).
    """
    rng = np.random.default_rng(11)
    stats = _stats_for({"A": rng.uniform(-0.05, 0.05, 312)}, idx=list(range(312)))
    assert set(stats.annual_breakdown()["year"].to_list()) == {1}


def test_annual_breakdown_temporal_keeps_benchmark():
    """Calendar-year breakdown must pass the benchmark into each year's summary.

    Kills _reporting mutant 529 (``year_benchmark`` forced to ``None``).
    """
    dates = _days(5, date(2022, 1, 3)) + _days(5, date(2023, 1, 3))
    stats = _stats_for(
        {"A": [0.01, -0.02, 0.015, 0.0, 0.005, 0.01, -0.01, 0.02, 0.0, -0.005]},
        dates=dates,
        benchmark={"B": [0.005, 0.01, -0.01, 0.02, 0.0, -0.005, 0.01, 0.0, 0.015, -0.02]},
    )
    bd = stats.annual_breakdown()
    assert sorted(set(bd["year"].to_list())) == [2022, 2023]
    assert "B" in bd.columns


def test_annual_breakdown_empty_schema_columns():
    """When every year is too short, the empty result must keep the exact schema.

    Each year contains a single row (< 2), so the typed empty frame is
    returned.  Kills _reporting mutant 537 (``"metric"`` schema key ->
    ``"XXmetricXX"``).
    """
    stats = _stats_for({"A": [0.01, 0.02]}, dates=[date(2020, 6, 1), date(2021, 6, 1)])
    bd = stats.annual_breakdown()
    assert bd.height == 0
    assert bd.columns == ["year", "metric", "A"]


# ── _reporting: summary ───────────────────────────────────────────────────────


def test_summary_metric_labels_exact():
    """summary() must emit the exact metric labels in declaration order.

    Kills _reporting mutants 550-559 and 561-569 (XX-wrapped metric keys).
    """
    stats = _stats_for({"A": [0.01, -0.02, 0.015, 0.0, -0.005, 0.02]}, dates=_days(6))
    assert stats.summary()["metric"].to_list() == [
        "avg_return",
        "avg_win",
        "avg_loss",
        "win_rate",
        "profit_factor",
        "payoff_ratio",
        "monthly_win_rate",
        "best",
        "worst",
        "volatility",
        "sharpe",
        "skew",
        "kurtosis",
        "value_at_risk",
        "conditional_value_at_risk",
        "max_drawdown",
        "avg_drawdown",
        "max_drawdown_duration",
        "calmar",
        "recovery_factor",
    ]


# ── _rolling: default arguments ───────────────────────────────────────────────


@pytest.fixture
def rolling_stats():
    """300 rows of seeded pseudo-random returns with a correlated benchmark."""
    rng = np.random.default_rng(42)
    a = rng.normal(0.0005, 0.01, 300)
    b = 0.5 * a + rng.normal(0.0, 0.005, 300)
    stats = _stats_for({"A": a}, dates=_days(300, date(2020, 1, 1)), benchmark={"B": b})
    return stats, a


def test_rolling_defaults_match_explicit_arguments(rolling_stats):
    """Every rolling default must equal the explicitly spelled-out default.

    With 300 varied rows, an off-by-one window changes the null warm-up
    pattern (and the values), so frame equality kills each default mutant.
    Kills _rolling mutants 1 (``periods=252`` -> ``253``), 2
    (``annualize=True`` -> ``False``: returns a dict), 28 (``window=60`` ->
    ``61``), 38/59/72/112 (``rolling_period=126`` -> ``127``), and 80/81
    (``ppy`` collapses to ``None``: alpha would be all-null).
    """
    stats, _ = rolling_stats

    iv = stats.implied_volatility()
    assert isinstance(iv, pl.DataFrame)  # 2
    assert_frame_equal(iv, stats.implied_volatility(periods=252, annualize=True))  # 1

    assert_frame_equal(stats.pct_rank(), stats.pct_rank(window=60))  # 28
    assert_frame_equal(stats.rolling_sortino(), stats.rolling_sortino(rolling_period=126))  # 38
    assert_frame_equal(stats.rolling_sharpe(), stats.rolling_sharpe(rolling_period=126))  # 59
    assert_frame_equal(stats.rolling_volatility(), stats.rolling_volatility(rolling_period=126))  # 112

    greeks = stats.rolling_greeks()
    assert_frame_equal(greeks, stats.rolling_greeks(rolling_period=126))  # 72
    alpha = greeks["A_alpha"][-1]
    assert alpha is not None  # 80, 81: ppy=None would null out every alpha
    assert math.isfinite(alpha)


# ── _rolling: validation ──────────────────────────────────────────────────────


def test_pct_rank_accepts_window_of_one():
    """``window=1`` is a valid positive window and ranks every price at 100.

    Kills _rolling mutant 31 (``window <= 0`` -> ``<= 1`` would reject 1).
    """
    stats = _stats_for({"A": [0.01, -0.02, 0.03, 0.0]}, dates=_days(4))
    assert stats.pct_rank(window=1)["A"].to_list() == [100.0, 100.0, 100.0, 100.0]


def test_rolling_validators_accept_window_of_one():
    """Every rolling validator must accept a window of exactly 1.

    Kills _rolling mutants 41, 65, 77, and 119 (``<= 0`` -> ``<= 1``).
    """
    stats = _stats_for(
        {"A": [0.01, 0.02, -0.01, 0.03]},
        dates=_days(4),
        benchmark={"B": [0.02, 0.01, 0.0, 0.02]},
    )
    assert isinstance(stats.rolling_sortino(rolling_period=1), pl.DataFrame)  # 41
    assert isinstance(stats.rolling_sharpe(rolling_period=1), pl.DataFrame)  # 65
    assert isinstance(stats.rolling_volatility(rolling_period=1), pl.DataFrame)  # 119
    assert isinstance(stats.rolling_greeks(rolling_period=1), pl.DataFrame)  # 77


def test_rolling_validators_reject_zero_with_exact_message():
    """A zero window must raise ValueError with the exact validation message.

    Kills _rolling mutants 33, 43, 67, 79, and 121 (XX-wrapped messages) and
    76 (``<= 0`` -> ``< 0`` lets 0 through to a non-ValueError polars failure).
    """
    stats = _stats_for(
        {"A": [0.01, 0.02, -0.01, 0.03]},
        dates=_days(4),
        benchmark={"B": [0.02, 0.01, 0.0, 0.02]},
    )
    period_msg = "^rolling_period must be a positive integer$"
    with pytest.raises(ValueError, match=r"^window must be a positive integer$"):
        stats.pct_rank(window=0)  # 33
    with pytest.raises(ValueError, match=period_msg):
        stats.rolling_sortino(rolling_period=0)  # 43
    with pytest.raises(ValueError, match=period_msg):
        stats.rolling_sharpe(rolling_period=0)  # 67
    with pytest.raises(ValueError, match=period_msg):
        stats.rolling_volatility(rolling_period=0)  # 121
    with pytest.raises(ValueError, match=period_msg):
        stats.rolling_greeks(rolling_period=0)  # 76, 79


def test_rolling_greeks_requires_benchmark_message():
    """rolling_greeks without a benchmark must raise the exact message.

    Kills _rolling mutant 74 (XX-wrapped message).
    """
    stats = _stats_for({"A": [0.01, -0.02, 0.03]}, dates=_days(3))
    with pytest.raises(AttributeError, match=r"^No benchmark data available$"):
        stats.rolling_greeks()


# ── _rolling: pinned formulas ─────────────────────────────────────────────────


def test_rolling_greeks_pins_alpha_formula():
    """Rolling alpha must equal ``(mean_x - beta * mean_y) * ppy`` exactly.

    Hand-computes the OLS quantities for the final 3-row window.  Kills
    _rolling mutants 105 (``mean_x + beta * mean_y``), 106
    (``cov * var``), 107 (``beta / mean_y``), 108 (``* ppy`` -> ``/ ppy``),
    and 109 (XX-wrapped ``_alpha`` column alias -> ColumnNotFound).
    """
    x = [0.01, 0.02, -0.01, 0.03]
    y = [0.02, 0.01, 0.0, 0.02]
    stats = _stats_for({"A": x}, dates=_days(4), benchmark={"B": y})
    res = stats.rolling_greeks(rolling_period=3, periods_per_year=252)

    wx, wy = x[1:], y[1:]
    mean_x = sum(wx) / 3.0
    mean_y = sum(wy) / 3.0
    mean_xy = sum(xv * yv for xv, yv in zip(wx, wy, strict=True)) / 3.0
    mean_y2 = sum(yv * yv for yv in wy) / 3.0
    var = mean_y2 - mean_y * mean_y
    cov = mean_xy - mean_x * mean_y
    beta = cov / var
    alpha = (mean_x - beta * mean_y) * 252.0

    assert res["A_beta"][3] == pytest.approx(beta, rel=TOL_COMPOUNDING, abs=TOL_COMPOUNDING)
    assert res["A_alpha"][3] == pytest.approx(alpha, rel=TOL_COMPOUNDING, abs=TOL_COMPOUNDING)


def test_rolling_volatility_unannualized_is_sample_std(rolling_stats):
    """``annualize=False`` must scale by exactly 1.0 (raw sample std).

    Kills _rolling mutant 124 (``else 1.0`` -> ``else 2.0``).
    """
    stats, a = rolling_stats
    res = stats.rolling_volatility(rolling_period=126, annualize=False)
    expected = float(np.std(a[-126:], ddof=1))
    assert res["A"][-1] == pytest.approx(expected, rel=TOL_COMPOUNDING)
