"""Mutation-killing tests for ``PortfolioCostMixin`` (``_portfolio_cost.py``).

Each test targets a specific surviving mutant found by ``mutmut`` and is
designed so the assertion *fails* when the mutation is applied:

- ``position_delta_costs`` asset predicate (``!= "date"`` / ``and``) — a
  cashposition with a *numeric* ``'date'`` column distinguishes the original
  predicate from the ``!= "XXdateXX"`` and ``or`` mutants.
- ``net_cost_nav`` ``"date" in ...`` branch — duplicate dates make the
  date-join path (row explosion) observably differ from the ``hstack`` path.
- Error-message string mutants — anchored ``match`` regexes pin the exact
  runtime-formatted text.
- ``cost_adjusted_returns`` arithmetic (``/ 10_000.0``) — an exact-formula
  cross-check on a non-trivial turnover series.
- ``trading_cost_impact`` ``max_bps`` boundary (``< 1``) — calling with
  ``max_bps=1`` must succeed.
"""

from __future__ import annotations

import re
from datetime import date

import polars as pl
import pytest

from jquantstats import Portfolio

from ..tolerances import TOL_FLOAT64

# ── position_delta_costs asset predicate (mutants 3, 4) ───────────────────────


def test_position_delta_costs_excludes_numeric_date_column():
    """The asset predicate must exclude a *numeric* ``'date'`` column.

    With a cashposition whose ``'date'`` column is ``Int64`` (numeric), the
    predicate ``c != "date" and is_numeric()`` keeps only the genuine asset
    ``'A'``.  Both the ``!= "XXdateXX"`` mutant (3) and the ``or`` mutant (4)
    would pull the numeric ``'date'`` column into the asset list, adding its
    own |Δ| = [0, 10, 10] to every cost row.
    """
    cpu = 0.01
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
    pf = Portfolio(prices=prices, cashposition=cashposition, aum=1e5, cost_per_unit=cpu)

    costs = pf.position_delta_costs["cost"].to_list()
    # Genuine asset-only |Δ| = [0, 1000, 300]; the mutants would add the
    # numeric date deltas [0, 10, 10] → [0, 1010, 310].
    assert costs == pytest.approx([0.0, 1000.0 * cpu, 300.0 * cpu], abs=TOL_FLOAT64)


# ── net_cost_nav "date" branch (mutant 20) ────────────────────────────────────


def test_net_cost_nav_uses_date_join_path():
    """``net_cost_nav`` must take the date-join branch when a ``'date'`` column exists.

    Duplicate dates make the join path observable: a left join on a date that
    appears twice on both sides yields a 2×2 row explosion, whereas the
    ``"XXdateXX"`` mutant (20) falls through to the ``hstack`` branch and
    leaves the row count unchanged.
    """
    dates = [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 2)]
    prices = pl.DataFrame({"date": dates, "A": pl.Series([100.0, 110.0, 121.0])})
    pos = pl.DataFrame({"date": dates, "A": pl.Series([1000.0, 1000.0, 1000.0])})
    pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)

    # Join path: date 2020-01-01 (x2 on both sides) → 4 rows, 2020-01-02 → 1 → 5.
    # The hstack mutant keeps the original 3 rows.
    assert pf.net_cost_nav.height == 5


# ── cost_adjusted_returns error messages (mutants 37, 40) ─────────────────────


def test_cost_adjusted_returns_non_numeric_message_is_exact(turnover_portfolio):
    """The ``TypeError`` text must be exactly ``cost_bps must be a number, got <type>``.

    Anchoring the regex at both ends kills the ``XX...XX`` string mutant (37),
    which a bare-substring match would miss.
    """
    expected = "cost_bps must be a number, got str"
    with pytest.raises(TypeError, match=rf"^{re.escape(expected)}$"):
        turnover_portfolio.cost_adjusted_returns("5")


def test_cost_adjusted_returns_non_finite_message_is_exact(turnover_portfolio):
    """The ``ValueError`` text must be exactly ``cost_bps must be finite, got inf``.

    Anchored match kills the ``XX...XX`` string mutant (40).
    """
    expected = "cost_bps must be finite, got inf"
    with pytest.raises(ValueError, match=rf"^{re.escape(expected)}$"):
        turnover_portfolio.cost_adjusted_returns(float("inf"))


# ── cost_adjusted_returns arithmetic (mutants 46, 47) ─────────────────────────


def test_cost_adjusted_returns_exact_bps_formula(turnover_portfolio):
    """Per-period deduction must equal ``turnover * (cost_bps / 10_000)`` exactly.

    The implied cost (``base − adjusted``) is cross-checked against the
    independently computed expected cost at a very tight tolerance.  The
    ``* 10_000`` mutant (46) explodes the cost; the ``/ 10001`` mutant (47)
    shifts it by ~1e-9 per row — both exceed the 1e-12 relative tolerance.
    """
    bps = 7.0
    base = turnover_portfolio.returns["returns"].to_list()
    adj = turnover_portfolio.cost_adjusted_returns(bps)["returns"].to_list()
    turn = turnover_portfolio.turnover["turnover"].to_list()

    implied_cost = [b - a for b, a in zip(base, adj, strict=True)]
    expected_cost = [t * (bps / 10_000.0) for t in turn]
    assert implied_cost == pytest.approx(expected_cost, rel=1e-12, abs=1e-15)


# ── trading_cost_impact max_bps boundary (mutants 54, 55) ─────────────────────


def test_trading_cost_impact_accepts_max_bps_of_one(turnover_portfolio):
    """``trading_cost_impact(max_bps=1)`` must succeed and yield rows ``[0, 1]``.

    The original guard is ``max_bps < 1``; the ``<= 1`` mutant (54) and the
    ``< 2`` mutant (55) both reject ``max_bps=1`` by raising
    ``InvalidMaxBpsError``.
    """
    impact = turnover_portfolio.trading_cost_impact(max_bps=1)
    assert impact.height == 2
    assert impact["cost_bps"].to_list() == [0, 1]
