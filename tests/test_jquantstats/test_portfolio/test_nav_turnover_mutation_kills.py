"""Mutation-killing tests for the NAV/returns and turnover mixins.

Targets surviving ``mutmut`` mutants in ``_portfolio_nav.py`` and
``_portfolio_turnover.py``.  Each test is constructed so its assertion fails
under the targeted mutation:

- ``profits`` fill mutants (7, 10) — crafted nulls in prices / positions push a
  non-zero value through exactly one of the two ``fill_null`` calls.
- ``profit`` / ``monthly`` error-message string mutants (19, 41) — anchored
  ``match`` regexes pin the exact runtime text.
- ``profit`` column selection (25) — the ``'date'`` column must survive.
- ``monthly`` compounding arithmetic (57, 58, 61, 62) — exact compounded
  monthly return on a hand-computed frame.
- ``drawdown_pct`` formula (92, 94) — exact percentage on a peak-then-dip NAV.
- ``all`` date-join branch (99) — duplicate dates expose join vs hstack.
- turnover asset predicate (5, 6), ``fill_nan`` (9), weekly ``rolling_sum``
  window (26), and ``turnover_summary`` fallbacks (35, 38, 44, 46).
"""

from __future__ import annotations

import math
import re
from datetime import date

import polars as pl
import pytest

from jquantstats import Portfolio
from jquantstats.exceptions import MissingDateColumnError, NoAssetColumnsError

from ..tolerances import TOL_COMPOUNDING, TOL_FLOAT64

# ══ NAV / returns mixin ═══════════════════════════════════════════════════════

# ── profits fill_null mutants (7, 10) ─────────────────────────────────────────


def test_profits_fill_null_on_pct_change_is_zero():
    """A null pct_change must contribute zero profit (kills pct_change fill_null(1.0)).

    With a null price in the middle row, every pct_change is null and is filled
    with 0.0; multiplied by the (non-zero) shifted positions the profit is all
    zeros.  Mutant 7 fills with 1.0 → profit becomes [0, 1000, 1000].
    """
    prices = pl.DataFrame({"A": pl.Series([100.0, None, 121.0], dtype=pl.Float64)})
    pos = pl.DataFrame({"A": pl.Series([1000.0, 1000.0, 1000.0], dtype=pl.Float64)})
    pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)

    assert pf.profit["profit"].to_list() == pytest.approx([0.0, 0.0, 0.0], abs=TOL_FLOAT64)


def test_profits_fill_null_on_shifted_position_is_zero():
    """A null shifted position must contribute zero profit (kills shift fill_null(1.0)).

    A null position at row 1 makes ``shift(1)`` null at row 2; filled with 0.0
    the profit there is 0.  Mutant 10 fills with 1.0 → row 2 profit becomes
    ``pct_change * 1`` = 0.1 instead of 0.
    """
    prices = pl.DataFrame({"A": pl.Series([100.0, 110.0, 121.0], dtype=pl.Float64)})
    pos = pl.DataFrame({"A": pl.Series([1000.0, None, 1000.0], dtype=pl.Float64)})
    pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)

    # pct_change = [0, 0.1, 0.1]; shift(pos) filled = [0, 1000, 0] → [0, 100, 0].
    assert pf.profit["profit"].to_list() == pytest.approx([0.0, 100.0, 0.0], abs=TOL_FLOAT64)


# ── profit / monthly error-message mutants (19, 41) ───────────────────────────


def test_profit_no_asset_columns_message_is_exact():
    """``profit`` must raise ``NoAssetColumnsError('profits')`` with the exact text.

    A prices/positions frame containing only a (non-numeric) ``'date'`` column
    has no numeric assets, so ``profit`` raises.  The anchored match pins the
    embedded frame name and kills the ``"XXprofitsXX"`` mutant (19).
    """
    dates = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
    prices = pl.DataFrame({"date": dates})
    pos = pl.DataFrame({"date": dates})
    pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)

    expected = (
        "DataFrame 'profits' contains no numeric asset columns; at least one numeric column besides 'date' is required."
    )
    with pytest.raises(NoAssetColumnsError, match=rf"^{re.escape(expected)}$"):
        _ = pf.profit


def test_monthly_missing_date_message_is_exact(int_portfolio):
    """``monthly`` must raise ``MissingDateColumnError('monthly')`` with the exact text.

    The anchored match pins the embedded frame name ``monthly`` and kills the
    ``"XXmonthlyXX"`` mutant (41); a bare-substring match would not.
    """
    expected = "DataFrame 'monthly' is missing the required 'date' column."
    with pytest.raises(MissingDateColumnError, match=rf"^{re.escape(expected)}$"):
        _ = int_portfolio.monthly


# ── profit column selection (25) ──────────────────────────────────────────────


def test_profit_preserves_date_column(portfolio):
    """``profit`` must keep the non-asset ``'date'`` column.

    Mutant 25 corrupts the ``[*non_assets, ...]`` selection; if applied it
    either drops the date column or fails to import the module, so asserting
    the ``'date'`` column is present (and the row count is preserved) kills it.
    """
    df = portfolio.profit
    assert "date" in df.columns
    assert "profit" in df.columns
    assert df.height == portfolio.prices.height


# ── monthly compounding arithmetic (57, 58, 61, 62) ───────────────────────────


def test_monthly_compounded_return_is_exact():
    """Monthly return must be ``prod(1 + r) - 1`` on a hand-computed frame.

    Three days, +0.1% each (returns [0, 0.001, 0.001]) → gross = 1.001² and
    monthly return = 1.001² − 1.  This single value distinguishes all four
    arithmetic mutants:
    57 ``prod(r - 1)``, 58 ``prod(r + 2)``, 61 ``gross + 1``, 62 ``gross - 2``.
    """
    dates = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
    prices = pl.DataFrame({"date": dates, "A": pl.Series([100.0, 110.0, 121.0])})
    pos = pl.DataFrame({"date": dates, "A": pl.Series([1000.0, 1000.0, 1000.0])})
    pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)

    monthly = pf.monthly
    jan = monthly.filter((pl.col("year") == 2020) & (pl.col("month") == 1))
    assert jan.height == 1
    expected = (1.0 + 0.001) * (1.0 + 0.001) - 1.0
    assert float(jan["returns"][0]) == pytest.approx(expected, rel=TOL_COMPOUNDING)


# ── drawdown_pct formula (92, 94) ─────────────────────────────────────────────


def test_drawdown_pct_is_relative_drop_from_highwater():
    """``drawdown_pct`` must equal ``(highwater - NAV) / highwater``.

    A peak-then-dip NAV gives a non-zero drawdown on the last row.  The result
    is cross-checked against the recomputed formula (whose inputs are produced
    by unmutated code) and against the exact hand value, killing the
    ``highwater + NAV`` mutant (92) and the ``* highwater`` mutant (94).
    """
    dates = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
    prices = pl.DataFrame({"date": dates, "A": pl.Series([100.0, 110.0, 99.0])})
    pos = pl.DataFrame({"date": dates, "A": pl.Series([1000.0, 1000.0, 1000.0])})
    pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)

    dd = pf.drawdown
    recomputed = ((dd["highwater"] - dd["NAV_accumulated"]) / dd["highwater"]).to_list()
    actual = dd["drawdown_pct"].to_list()
    assert actual == pytest.approx(recomputed, rel=TOL_COMPOUNDING, abs=TOL_FLOAT64)

    # NAV path: [100000, 100100, 100000]; highwater [100000, 100100, 100100].
    assert actual[0] == pytest.approx(0.0, abs=TOL_FLOAT64)
    assert actual[2] == pytest.approx(100.0 / 100100.0, rel=TOL_COMPOUNDING)


# ── all date-join branch (99) ─────────────────────────────────────────────────


def test_all_uses_date_join_path():
    """``all`` must take the date-join branch when a ``'date'`` column exists.

    Duplicate dates make the inner join on date explode rows (2×2 for the
    repeated date), while the ``"XXdateXX"`` mutant (99) takes the ``hstack``
    branch and keeps the original row count.
    """
    dates = [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 2)]
    prices = pl.DataFrame({"date": dates, "A": pl.Series([100.0, 110.0, 121.0])})
    pos = pl.DataFrame({"date": dates, "A": pl.Series([1000.0, 1000.0, 1000.0])})
    pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)

    # Inner join: 2020-01-01 (x2 both sides) → 4, 2020-01-02 → 1 → 5 rows.
    assert pf.all.height == 5


# ══ turnover mixin ════════════════════════════════════════════════════════════

# ── turnover asset predicate (5, 6) ───────────────────────────────────────────


def test_turnover_excludes_numeric_date_column():
    """The turnover asset predicate must exclude a *numeric* ``'date'`` column.

    A numeric ``Int64`` ``'date'`` column exercises ``c != "date" and
    is_numeric()``: only asset ``'A'`` should contribute.  The ``!= "XXdateXX"``
    mutant (5) and the ``or`` mutant (6) both include the date deltas, changing
    every turnover row.
    """
    aum = 1000.0
    cashposition = pl.DataFrame(
        {
            "date": pl.Series([10, 20, 30], dtype=pl.Int64),
            "A": pl.Series([0.0, 1000.0, 700.0], dtype=pl.Float64),
        }
    )
    prices = pl.DataFrame(
        {
            "date": pl.Series([10, 20, 30], dtype=pl.Int64),
            "A": pl.Series([100.0, 110.0, 121.0], dtype=pl.Float64),
        }
    )
    pf = Portfolio(prices=prices, cashposition=cashposition, aum=aum)

    turn = pf.turnover["turnover"].to_list()
    # |Δ A| / aum = [0, 1000, 300] / 1000; mutants would add date |Δ| [0, 10, 10].
    assert turn == pytest.approx([0.0, 1000.0 / aum, 300.0 / aum], abs=TOL_FLOAT64)


# ── turnover fill_nan (9) ─────────────────────────────────────────────────────


def test_turnover_fill_nan_is_zero():
    """NaN position deltas must be filled with 0.0 in turnover (kills fill_nan(1.0)).

    A NaN position makes the diff NaN on rows 1 and 2; filled with 0.0 the
    turnover is all zeros.  Mutant 9 fills with 1.0 → [0, 0.001, 0.001].
    """
    prices = pl.DataFrame({"A": pl.Series([100.0, 110.0, 121.0])})
    pos = pl.DataFrame({"A": pl.Series([1000.0, float("nan"), 1000.0], dtype=pl.Float64)})
    pf = Portfolio(prices=prices, cashposition=pos, aum=1000.0)

    assert pf.turnover["turnover"].to_list() == pytest.approx([0.0, 0.0, 0.0], abs=TOL_FLOAT64)


# ── weekly rolling_sum window (26) ────────────────────────────────────────────


def test_turnover_weekly_rolling_window_is_five():
    """Date-free ``turnover_weekly`` must use a 5-period rolling sum.

    Seven rows with strictly increasing turnover [0, .1, .2, .3, .4, .5, .6]:
    the window-5 sum at the last row is rows 2..6 = 2.0.  Mutant 26 widens the
    window to 6 (rows 1..6) → 2.1.
    """
    n = 7
    prices = pl.DataFrame({"A": pl.Series([100.0] * n)})
    pos = pl.DataFrame({"A": pl.Series([0.0, 100.0, 300.0, 600.0, 1000.0, 1500.0, 2100.0])})
    pf = Portfolio(prices=prices, cashposition=pos, aum=1000.0)

    weekly = pf.turnover_weekly["turnover"].to_list()
    assert weekly[:4] == [None, None, None, None]  # min_samples=5
    assert weekly[4] == pytest.approx(1.0, abs=TOL_FLOAT64)
    assert weekly[5] == pytest.approx(1.5, abs=TOL_FLOAT64)
    assert weekly[6] == pytest.approx(2.0, abs=TOL_FLOAT64)  # window-6 mutant → 2.1


# ── turnover_summary fallbacks (35, 38) ───────────────────────────────────────


def test_turnover_summary_empty_uses_zero_fallbacks():
    """Empty turnover → mean/std fall back to 0.0 (kills the 1.0 fallback mutants).

    A zero-row portfolio makes ``mean()``/``std()`` return ``None``, exercising
    the ``else 0.0`` branches.  Mutant 35 (mean → 1.0) and mutant 38
    (std → 1.0) change these summary values.
    """
    prices = pl.DataFrame({"A": pl.Series([], dtype=pl.Float64)})
    pos = pl.DataFrame({"A": pl.Series([], dtype=pl.Float64)})
    pf = Portfolio(prices=prices, cashposition=pos, aum=1000.0)

    summary = pf.turnover_summary()
    values = dict(zip(summary["metric"].to_list(), summary["value"].to_list(), strict=True))
    assert values["mean_daily_turnover"] == pytest.approx(0.0, abs=TOL_FLOAT64)
    assert values["turnover_std"] == pytest.approx(0.0, abs=TOL_FLOAT64)


# ── turnover_summary weekly mean branch (44) ──────────────────────────────────


def test_turnover_summary_single_weekly_value_is_used():
    """A single weekly value must be used (kills the ``len() > 1`` mutant 44).

    A 5-row date-free portfolio yields exactly one non-null weekly rolling sum
    (= 0.4).  The original guard ``weekly_col.len() > 0`` uses it; mutant 44
    (``> 1``) discards it and returns NaN.
    """
    prices = pl.DataFrame({"A": pl.Series([100.0] * 5)})
    pos = pl.DataFrame({"A": pl.Series([0.0, 100.0, 200.0, 300.0, 400.0])})
    pf = Portfolio(prices=prices, cashposition=pos, aum=1000.0)

    summary = pf.turnover_summary()
    values = dict(zip(summary["metric"].to_list(), summary["value"].to_list(), strict=True))
    mean_weekly = values["mean_weekly_turnover"]
    assert not math.isnan(mean_weekly)
    assert mean_weekly == pytest.approx(0.4, rel=TOL_COMPOUNDING)


# ── turnover_summary NaN-literal else branch (46) ─────────────────────────────


def test_turnover_summary_empty_weekly_returns_nan():
    """Empty weekly turnover must yield NaN, not a crash (kills the ``"XXnanXX"`` mutant).

    A 3-row date-free portfolio produces no complete 5-period window, so
    ``weekly_col`` is empty and the ``else float("nan")`` branch runs.  Mutant
    46 changes it to ``float("XXnanXX")``, which raises ``ValueError`` when that
    branch executes — so simply obtaining the (NaN) value kills it.
    """
    prices = pl.DataFrame({"A": pl.Series([100.0, 110.0, 121.0])})
    pos = pl.DataFrame({"A": pl.Series([0.0, 1000.0, 700.0])})
    pf = Portfolio(prices=prices, cashposition=pos, aum=1000.0)

    summary = pf.turnover_summary()
    values = dict(zip(summary["metric"].to_list(), summary["value"].to_list(), strict=True))
    assert math.isnan(values["mean_weekly_turnover"])
