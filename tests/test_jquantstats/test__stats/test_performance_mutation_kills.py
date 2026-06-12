"""Mutation-kill tests for `jquantstats._stats._performance` (round-3 sweep B2).

Each test targets specific surviving mutmut mutants (ids referenced in the
docstrings).  Expected values are computed *independently* inside each test by
transcribing the ORIGINAL formulas from the source with plain Python floats;
statistical moments (mean/std/skew/kurtosis) are taken from polars so that the
arithmetic of the formula itself is the only thing being pinned.  Identity
comparisons use ``TOL_FLOAT64`` (1e-12 relative).

Mutants accepted as equivalent (not observable through any input):

- ``probabilistic_sharpe_ratio`` (72, 83, 85-91, 93-94, 97, 99-100):
  ``benchmark_sr`` is hard-coded to ``0.0``, so every term of the
  ``var_bench_sr`` formula that those mutants alter is multiplied by zero.
  ``var_bench_sr`` always equals ``1 / t`` (strictly positive, t >= 2 when
  reached), the ``<= 0`` guard is dead code, and ``observed_sr - 0.0`` equals
  ``observed_sr + 0.0``.  ``test_probabilistic_sharpe_ratio_matches_formula``
  pins the formula anyway as a regression guard.
- ``sharpe`` (9): ``series.mean() is None`` implies ``series.std(ddof=1) is
  None``, so ``_std_is_negligible(None, ...)`` returns NaN before the
  ``mean_f`` fallback value is ever used in arithmetic.
- ``sortino`` (148): the ``mean_f < 0`` branch is unreachable — zero downside
  deviation means no negative returns, hence a non-negative (or NaN) mean.
- ``omega`` (184, 188): elements exactly at the threshold become exactly
  ``0.0`` in ``returns_less_thresh`` and contribute ``0.0`` to either sum, so
  widening ``>``/``<`` to ``>=``/``<=`` is unobservable.
- ``drawdown_details`` (243): the ``row_idx`` column is created but never
  referenced afterwards; renaming it is unobservable.
- ``_probabilistic_ratio_from_base`` (332): at ``n == 2`` the bias-corrected
  skew/kurtosis are NaN, so the original already returns NaN through the NaN
  variance; (333): polars skew and kurtosis are ``None`` under identical
  conditions (empty/all-null input), which is already caught by ``n <= 1``;
  (355): ``variance == 0.0`` exactly would require exact float cancellation
  of moment terms and is not robustly constructible.
- ``_sortino_base`` (387): ``s <= 0`` only adds ``0.0 ** 2`` terms to
  ``downside_sum``.
- alias renames (430, 479, 514): the aliased benchmark column is consumed
  strictly positionally via ``.to_numpy()``; the alias string is observable
  only through a duplicate-column-name collision with a user asset literally
  named ``"benchmark"`` / ``"_bench"`` (pathological, not worth pinning).
- ``treynor_ratio`` (537, 538): ``n == 0`` is unreachable because ``Data``
  validation requires at least two index rows (branch is ``pragma: no cover``).
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from jquantstats import Data
from jquantstats._stats import Stats

from ..tolerances import TOL_FLOAT64

# Small, asymmetric returns series: non-zero mean, std, skew and excess
# kurtosis, so every coefficient of the probabilistic-ratio formulas matters.
RETURNS_10 = [0.01, -0.02, 0.035, -0.005, 0.0125, -0.03, 0.0425, 0.02, -0.01, 0.015]


def _int_data(columns: dict, benchmark: dict | None = None) -> Data:
    """Build a Data object over a plain integer index (no temporal frequency)."""
    n = len(next(iter(columns.values())))
    return Data(
        returns=pl.DataFrame(columns),
        benchmark=pl.DataFrame(benchmark) if benchmark is not None else None,
        index=pl.DataFrame({"idx": list(range(n))}),
    )


def _expected_probabilistic(base: float, s: pl.Series) -> float:
    """Transcribe `_probabilistic_ratio_from_base` with plain Python floats.

    variance = (1 + 0.5*base^2 - skew*base + ((kurt - 3)/4)*base^2) / (n - 1)
    result   = norm.cdf(base / sqrt(variance))
    """
    n = s.count()
    skew = float(s.skew(bias=False))
    kurt = float(s.kurtosis(bias=False))
    variance = (1 + 0.5 * base**2 - skew * base + ((kurt - 3) / 4) * base**2) / (n - 1)
    assert variance > 0  # sanity: chosen inputs must keep the estimate valid
    return float(norm.cdf(base / np.sqrt(variance)))


# ── sharpe / sharpe_variance: `factor = periods or 1` fallback ───────────────


def test_sharpe_and_variance_factor_fallback_is_one(monkeypatch):
    """When periods-per-year is falsy the annualization factor must be 1, not 2.

    Kills mutants 14 (`sharpe`: ``periods or 1 -> periods or 2`` would scale by
    sqrt(2)) and 55 (`sharpe_variance`: would scale by 2).  The ``or 1`` branch
    is reachable only when ``Data._periods_per_year`` is falsy, so it is
    patched to 0.0.
    """
    data = _int_data({"x": RETURNS_10})
    monkeypatch.setattr(Data, "_periods_per_year", property(lambda self: 0.0))

    s = pl.Series(RETURNS_10)
    mean = float(s.mean())
    std = float(s.std(ddof=1))
    sr = mean / std

    # factor == 1 -> sharpe is the raw, unannualized mean/std ratio.
    assert data.stats.sharpe()["x"] == pytest.approx(sr, rel=TOL_FLOAT64)

    skew = float(s.skew(bias=False))
    kurt = float(s.kurtosis(bias=False))
    expected_variance = (1 + (skew * sr) / 2 + ((kurt - 3) / 4) * sr**2) / 10
    assert data.stats.sharpe_variance()["x"] == pytest.approx(expected_variance, rel=TOL_FLOAT64)


# ── probabilistic_sharpe_ratio: formula pin ───────────────────────────────────


def test_probabilistic_sharpe_ratio_matches_formula():
    """Pin PSR to its documented formula, transcribed with Python floats.

    Note: mutants 72, 83, 85-91, 93-94, 97, 99-100 are *equivalent* because
    ``benchmark_sr`` is hard-coded to 0.0 (every mutated term is multiplied by
    zero and ``var_bench_sr == 1/t > 0`` always).  This test pins the formula
    so the terms become guarded if ``benchmark_sr`` is ever parameterized.
    """
    data = _int_data({"x": RETURNS_10})
    s = pl.Series(RETURNS_10)
    t = 10
    sr = float(s.mean()) / float(s.std(ddof=1))
    skew = float(s.skew(bias=False))
    kurt = float(s.kurtosis(bias=False))
    benchmark_sr = 0.0
    var_bench_sr = (1 + (skew * benchmark_sr) / 2 + ((kurt - 3) / 4) * benchmark_sr**2) / t
    expected = float(norm.cdf((sr - benchmark_sr) / np.sqrt(var_bench_sr)))
    assert data.stats.probabilistic_sharpe_ratio()["x"] == pytest.approx(expected, rel=TOL_FLOAT64)


# ── HHI concentration guards ──────────────────────────────────────────────────


def test_hhi_positive_exact_value_with_three_positive_returns():
    """Exactly three positive returns must produce a value, not NaN.

    Kills mutant 107 (``<= 2 -> <= 3`` would return NaN for three positives).
    """
    data = _int_data({"x": [0.1, 0.2, 0.3, -0.05]})
    pos = [0.1, 0.2, 0.3]
    total = sum(pos)
    weights = [p / total for p in pos]
    sq = sum(w * w for w in weights)
    expected = (3 * sq - 1) / (3 - 1)  # == 1/12 up to float round-off
    assert data.stats.hhi_positive()["x"] == pytest.approx(expected, rel=TOL_FLOAT64)


def test_hhi_positive_is_nan_with_two_positive_returns():
    """Two positive returns are indeterminate and must yield NaN.

    Kills mutant 106 (``<= 2 -> < 2`` would compute a value for two positives).
    """
    data = _int_data({"x": [0.1, 0.2, -0.05]})
    assert math.isnan(data.stats.hhi_positive()["x"])


def test_hhi_negative_exact_value_with_three_negative_returns():
    """Exactly three negative returns must produce a value, not NaN.

    Kills mutant 124 (``<= 2 -> <= 3``).
    """
    data = _int_data({"x": [-0.1, -0.2, -0.3, 0.05]})
    neg = [-0.1, -0.2, -0.3]
    total = sum(neg)
    weights = [n / total for n in neg]
    sq = sum(w * w for w in weights)
    expected = (3 * sq - 1) / (3 - 1)
    assert data.stats.hhi_negative()["x"] == pytest.approx(expected, rel=TOL_FLOAT64)


def test_hhi_negative_is_nan_with_two_negative_returns():
    """Two negative returns are indeterminate and must yield NaN.

    Kills mutant 123 (``<= 2 -> < 2``).
    """
    data = _int_data({"x": [-0.1, -0.2, 0.05]})
    assert math.isnan(data.stats.hhi_negative()["x"])


# ── sortino: zero-downside branch ─────────────────────────────────────────────


def test_sortino_is_positive_infinity_without_downside():
    """All-positive returns (0 < mean < 1) must give +inf Sortino.

    Kills mutants 144 (``mean_f > 0 -> mean_f > 1`` falls through to NaN) and
    145 (``float("inf") -> float("XXinfXX")`` raises ValueError).
    """
    data = _int_data({"x": [0.01, 0.02, 0.03]})
    assert data.stats.sortino()["x"] == float("inf")


# ── omega ─────────────────────────────────────────────────────────────────────


def test_omega_exact_value_default_threshold():
    """Pin omega = sum(gains) / -sum(losses) for a zero threshold.

    Kills mutant 192 (``denom <= 0.0 -> denom <= 1.0``: the denominator here
    is ~0.065, so the mutant would return NaN instead of the ratio).
    """
    data = _int_data({"x": RETURNS_10})
    numer = sum(x for x in RETURNS_10 if x > 0.0)
    denom = -sum(x for x in RETURNS_10 if x < 0.0)
    assert data.stats.omega()["x"] == pytest.approx(numer / denom, rel=TOL_FLOAT64)


def test_omega_subtracts_risk_free_rate_exactly():
    """A non-zero rf must shift returns by the per-period risk-free rate.

    Kills mutant 163 (``rf != 0.0 -> rf != 1.0``: with rf=1.0 the mutant skips
    the subtraction entirely, changing the ratio materially).
    """
    data = _int_data({"x": RETURNS_10})
    rf_per_period = (1.0 + 1.0) ** (1.0 / 252) - 1.0
    shifted = [x - rf_per_period for x in RETURNS_10]
    numer = sum(y for y in shifted if y > 0.0)
    denom = -sum(y for y in shifted if y < 0.0)
    assert data.stats.omega(rf=1.0, periods=252)["x"] == pytest.approx(numer / denom, rel=TOL_FLOAT64)


def test_omega_is_nan_at_required_return_of_minus_one():
    """required_return == -1 is rejected even when the math would not blow up.

    The series contains a sub-(-100%) return so that, past the guard, the
    threshold of -1 would produce a *finite* ratio — making the guard the only
    thing standing between NaN and a number.  Kills mutants 156
    (``<= -1 -> < -1``) and 158 (``<= -1 -> <= -2``).
    """
    data = _int_data({"x": [0.1, -1.5, 0.2]})
    assert math.isnan(data.stats.omega(required_return=-1.0)["x"])


# ── max_drawdown_single_series ────────────────────────────────────────────────


def test_max_drawdown_single_series_callable_on_instance_with_exact_value():
    """The helper must stay a @staticmethod and return the exact NAV trough.

    Kills mutant 208 (removing ``@staticmethod`` makes the instance call pass
    ``self`` as ``series`` -> TypeError).  NAV path 1.0 -> 0.5 -> 1.0 gives an
    exact dyadic drawdown of -0.5.
    """
    stats = _int_data({"x": [0.0, 0.1]}).stats
    assert stats.max_drawdown_single_series(pl.Series([0.0, -0.5, 1.0])) == -0.5


def test_max_drawdown_single_series_empty_series_returns_zero():
    """An empty series has no drawdown: the None fallback must be 0.0.

    Kills mutant 217 (``else 0.0 -> else 1.0``) and re-kills 208 (instance
    call).
    """
    stats = _int_data({"x": [0.0, 0.1]}).stats
    assert stats.max_drawdown_single_series(pl.Series([], dtype=pl.Float64)) == 0.0


# ── drawdown_details ──────────────────────────────────────────────────────────


def test_drawdown_details_integer_index_uses_positions_and_exact_durations():
    """Integer-indexed data must use 0-based positions, not the index values.

    The index values (10..70) deliberately differ from positions (0..6) so the
    two are distinguishable.  NAV: 1.0, 0.9, 0.81, 1.0125, 1.0125, 0.81, 0.729
    -> a recovered drawdown over positions 1-2 (recovery at 3) and an ongoing
    one over positions 5-6.

    Kills mutants 235 (``has_date and ... -> or`` would use the raw 10..70
    index values), 269/276/277 (recovery-join mutations null out or misalign
    ``end``), 302 (recovered duration ``end - start -> end + start``),
    305/307/308 (ongoing duration ``last - start + 1`` sign/offset mutations),
    and 312 (``recovery_duration = end - valley -> end + valley``).
    """
    returns_b = [0.0, -0.1, -0.1, 0.25, 0.0, -0.2, -0.1]
    data = Data(
        returns=pl.DataFrame({"B": returns_b}),
        index=pl.DataFrame({"idx": [10, 20, 30, 40, 50, 60, 70]}),
    )
    details = data.stats.drawdown_details()["B"]

    assert details["start"].to_list() == [1, 5]
    assert details["valley"].to_list() == [2, 6]
    assert details["end"].to_list() == [3, None]
    assert details["duration"].to_list() == [2, 2]
    assert details["recovery_duration"].to_list() == [1, None]

    # max_drawdown replication: nav / running-high-water-mark - 1.
    nav = []
    acc = 1.0
    for r in returns_b:
        acc = acc * (1.0 + r)
        nav.append(acc)
    expected_dd = [nav[2] / 1.0 - 1, nav[6] / nav[3] - 1]
    assert details["max_drawdown"].to_list() == pytest.approx(expected_dd, rel=TOL_FLOAT64)


def test_drawdown_details_temporal_schema_all_assets_and_ongoing_duration():
    """Temporal data: typed empty frame, all assets reported, day-count +1.

    Asset "A" rises monotonically (empty drawdown table); asset "B" enters an
    ongoing drawdown on Jan 3 lasting through Jan 5 (duration 2 days + 1 = 3).

    Kills mutants 238 (``date_dtype = None`` would make the empty columns Null
    typed), 249-254 (empty-frame column renames), 256 (``continue -> break``
    would drop asset "B" from the result), and 291/292 (ongoing temporal
    duration ``+ 1 -> - 1`` / ``+ 2``).
    """
    returns_b = [0.0, 0.05, -0.1, -0.05, -0.01]
    data = Data(
        returns=pl.DataFrame({"A": [0.01] * 5, "B": returns_b}),
        index=pl.DataFrame({"Date": [date(2023, 1, d) for d in range(1, 6)]}),
    )
    details = data.stats.drawdown_details()

    assert list(details) == ["A", "B"]

    empty = details["A"]
    assert empty.height == 0
    assert empty.columns == ["start", "valley", "end", "duration", "max_drawdown", "recovery_duration"]
    assert empty.schema["start"] == pl.Date
    assert empty.schema["valley"] == pl.Date
    assert empty.schema["end"] == pl.Date
    assert empty.schema["duration"] == pl.Int64
    assert empty.schema["max_drawdown"] == pl.Float64
    assert empty.schema["recovery_duration"] == pl.Int64

    b = details["B"]
    assert b["start"].to_list() == [date(2023, 1, 3)]
    assert b["valley"].to_list() == [date(2023, 1, 5)]
    assert b["end"].to_list() == [None]
    assert b["duration"].to_list() == [3]  # (Jan 5 - Jan 3).days + 1
    assert b["recovery_duration"].to_list() == [None]

    nav = []
    acc = 1.0
    for r in returns_b:
        acc = acc * (1.0 + r)
        nav.append(acc)
    assert b["max_drawdown"].to_list() == pytest.approx([nav[4] / nav[1] - 1], rel=TOL_FLOAT64)


# ── _probabilistic_ratio_from_base / probabilistic_ratio ─────────────────────


def test_probabilistic_ratio_from_base_single_observation_is_nan():
    """N == 1 is insufficient data and must return NaN, not divide by zero.

    Kills mutant 331 (``n <= 1 -> n < 1``: polars skew/kurtosis of a single
    observation are NaN (0/0), so the mutant reaches
    ``variance = (...) / (n - 1)`` and raises ZeroDivisionError).
    """
    assert math.isnan(Stats._probabilistic_ratio_from_base(0.5, pl.Series([0.1])))


def test_probabilistic_ratio_default_base_is_sharpe():
    """Calling with no argument must equal an explicit base="sharpe" call.

    Kills mutant 377 (default ``"sharpe" -> "XXsharpeXX"`` raises ValueError on
    the no-argument call).
    """
    stats = _int_data({"x": RETURNS_10}).stats
    assert stats.probabilistic_ratio() == stats.probabilistic_ratio(base="sharpe")


def test_probabilistic_ratio_sharpe_exact_value():
    """Pin the sharpe-based probabilistic ratio to its transcribed formula.

    Kills mutants 379 (``std(ddof=1) -> std(ddof=2)`` rescales the base by
    sqrt(9/8)) and 386 (``mean / std -> mean * std``; std != 1 here so the two
    differ).
    """
    stats = _int_data({"x": RETURNS_10}).stats
    s = pl.Series(RETURNS_10)
    base = float(s.mean()) / float(s.std(ddof=1))
    expected = _expected_probabilistic(base, s)
    assert stats.probabilistic_ratio(base="sharpe")["x"] == pytest.approx(expected, rel=TOL_FLOAT64)


def test_probabilistic_ratio_sharpe_with_std_exactly_one():
    """A sample std of exactly 1.0 is valid dispersion, not a degenerate case.

    The series has sum of squared deviations exactly 4.0 over n-1 = 4 (all
    dyadic floats), so ``std(ddof=1) == 1.0`` exactly.  Kills mutant 383
    (``std_val == 0 -> std_val == 1`` would return NaN).
    """
    vals = [-1.0, 0.0, 1.0, 1.0, 1.5]
    stats = _int_data({"x": vals}).stats
    s = pl.Series(vals)
    assert float(s.std(ddof=1)) == 1.0  # precondition for the kill
    base = float(s.mean()) / 1.0
    expected = _expected_probabilistic(base, s)
    assert stats.probabilistic_ratio(base="sharpe")["x"] == pytest.approx(expected, rel=TOL_FLOAT64)


def test_probabilistic_ratio_sharpe_single_observation_column_is_nan():
    """A column with one non-null value has no std and must map to NaN.

    polars returns null for ``std(ddof=1)`` of a single observation, so
    ``not std_val`` short-circuits.  Kills mutant 384 (``or -> and`` proceeds
    to ``mean_val / None`` and raises TypeError).
    """
    stats = _int_data({"y": [0.05, None]}).stats
    assert math.isnan(stats.probabilistic_ratio(base="sharpe")["y"])


def test_probabilistic_ratio_sortino_exact_value():
    """Pin the sortino-based probabilistic ratio to its transcribed formula.

    base = mean / sqrt(sum(r^2 for r < 0) / n).  Kills mutants 388
    (``s < 0 -> s < 1`` would also square the sub-1 positive returns), 392
    (``downside_sum / n -> downside_sum * n``), and 397
    (``mean / downside_dev -> mean * downside_dev``).
    """
    stats = _int_data({"x": RETURNS_10}).stats
    s = pl.Series(RETURNS_10)
    downside_sum = sum(x * x for x in RETURNS_10 if x < 0.0)
    downside_dev = float(np.sqrt(downside_sum / 10))
    base = float(s.mean()) / downside_dev
    expected = _expected_probabilistic(base, s)
    assert stats.probabilistic_ratio(base="sortino")["x"] == pytest.approx(expected, rel=TOL_FLOAT64)


def test_probabilistic_ratio_adjusted_sortino_exact_value():
    """Pin the adjusted-sortino base: sortino base divided by sqrt(2).

    Kills mutants 401 (``/ sqrt(2) -> * sqrt(2)``) and 402
    (``sqrt(2) -> sqrt(3)``).
    """
    stats = _int_data({"x": RETURNS_10}).stats
    s = pl.Series(RETURNS_10)
    downside_sum = sum(x * x for x in RETURNS_10 if x < 0.0)
    downside_dev = float(np.sqrt(downside_sum / 10))
    base = (float(s.mean()) / downside_dev) / float(np.sqrt(2))
    expected = _expected_probabilistic(base, s)
    assert stats.probabilistic_ratio(base="adjusted_sortino")["x"] == pytest.approx(expected, rel=TOL_FLOAT64)


def test_probabilistic_ratio_rejects_unknown_base_with_exact_message():
    """An unknown base string must raise the documented ValueError verbatim.

    Kills mutant 406 (``XX``-wrapped message fails the anchored match).
    """
    stats = _int_data({"x": RETURNS_10}).stats
    with pytest.raises(
        ValueError,
        match=r"^base must be one of \['sharpe', 'sortino', 'adjusted_sortino'\], got 'bogus'$",
    ):
        stats.probabilistic_ratio(base="bogus")


# ── benchmark-requiring metrics: error messages ──────────────────────────────


def test_r_squared_error_message_without_benchmark():
    """r_squared without a benchmark raises the exact documented message.

    Kills mutant 425.
    """
    stats = _int_data({"x": [0.01, -0.02]}).stats
    with pytest.raises(AttributeError, match=r"^No benchmark data available$"):
        stats.r_squared()


def test_information_ratio_error_message_without_benchmark():
    """information_ratio without a benchmark raises the exact message.

    Kills mutant 446.
    """
    stats = _int_data({"x": [0.01, -0.02]}).stats
    with pytest.raises(AttributeError, match=r"^No benchmark data available$"):
        stats.information_ratio()


def test_treynor_ratio_error_message_without_benchmark():
    """treynor_ratio without a benchmark raises the exact message.

    Kills mutant 506.
    """
    stats = _int_data({"x": [0.01, -0.02]}).stats
    with pytest.raises(AttributeError, match=r"^No benchmark data available$"):
        stats.treynor_ratio()


# ── information_ratio fallbacks ───────────────────────────────────────────────


def test_information_ratio_single_pair_uses_unit_std_fallback():
    """With one overlapping observation the std fallback must be 1.0, not 2.0.

    Only one non-null strategy/benchmark pair survives ``drop_nulls``; the
    active return is exactly 0.25 (dyadic), its std is null, and the fallback
    makes IR equal the mean.  Kills mutant 463 (``else 1.0 -> else 2.0``).
    """
    data = _int_data({"s": [0.5, None]}, benchmark={"b": [0.25, None]})
    result = data.stats.information_ratio()
    if pl.Series([1.0]).std(ddof=1) is None:
        assert result["s"] == pytest.approx(0.25, rel=TOL_FLOAT64)
    else:  # polars returned NaN std: mean / NaN propagates instead
        assert math.isnan(result["s"])


def test_information_ratio_returns_zero_on_zero_dispersion():
    """Constant active returns divide by exactly 0.0 and must map to 0.0.

    Active returns are exactly [0.25, 0.25] (dyadic), so ``std == 0.0`` and
    ``mean / 0.0`` raises ZeroDivisionError internally.  Kills mutant 470
    (``return 0.0 -> return 1.0``).
    """
    data = _int_data({"s": [0.5, 0.75]}, benchmark={"b": [0.25, 0.5]})
    assert data.stats.information_ratio()["s"] == 0.0


# ── greeks ────────────────────────────────────────────────────────────────────


def test_greeks_beta_with_unit_benchmark_variance():
    """Benchmark variance of exactly 1.0 must still produce a real beta.

    Benchmark [0.0, 1.0, 2.0] has sample variance exactly 1.0 (dyadic), the
    one input where mutant 495 (``var != 0 -> var != 1``) flips the result to
    NaN.  cov = 0.25, so beta = 0.25 and alpha = (0.5 - 0.25*1.0) * 252 = 63.
    """
    data = _int_data({"s": [0.25, 0.5, 0.75]}, benchmark={"b": [0.0, 1.0, 2.0]})
    greeks = data.stats.greeks()["s"]
    assert greeks["beta"] == pytest.approx(0.25, rel=TOL_FLOAT64)
    assert greeks["alpha"] == pytest.approx(63.0, rel=TOL_FLOAT64)


def test_greeks_beta_is_nan_for_constant_benchmark():
    """Zero benchmark variance must yield NaN greeks, not a ValueError.

    Kills mutant 496 (``float("nan") -> float("XXnanXX")`` raises ValueError
    when the zero-variance branch executes).
    """
    data = _int_data({"s": [0.5, 0.25]}, benchmark={"b": [0.125, 0.125]})
    greeks = data.stats.greeks()["s"]
    assert math.isnan(greeks["beta"])
    assert math.isnan(greeks["alpha"])


# ── treynor_ratio ─────────────────────────────────────────────────────────────


def test_treynor_ratio_exact_value_with_unit_benchmark_variance():
    """Benchmark variance of exactly 1.0 must not trip the zero-variance guard.

    Kills mutant 526 (``var == 0 -> var == 1`` returns NaN exactly when the
    variance is 1).  All inputs are dyadic so cov/var/beta/nav are exact.
    """
    data = _int_data({"s": [0.25, 0.5, 0.75]}, benchmark={"b": [0.0, 1.0, 2.0]})
    cov_matrix = np.cov(np.array([0.25, 0.5, 0.75]), np.array([0.0, 1.0, 2.0]))
    beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])  # == 0.25 exactly
    nav_final = (1.0 + 0.25) * (1.0 + 0.5) * (1.0 + 0.75)  # == 3.28125 exactly
    cagr = nav_final ** (252.0 / 3) - 1.0
    assert data.stats.treynor_ratio()["s"] == pytest.approx(cagr / beta, rel=TOL_FLOAT64)


def test_treynor_ratio_exact_value_for_losing_strategy():
    """A losing strategy (0 < final NAV <= 1) pins CAGR power, sign and ratio.

    nav_final = 0.375 exactly; 0.375**84 underflows against 1.0, so
    cagr == -1.0 exactly and treynor == -1.0 / 0.5 == -2.0.  Kills mutants 530
    (``cov / var -> cov * var``: beta 0.03125 -> -32.0), 543
    (``nav <= 0 -> nav <= 1``: NaN), 545 (``nav ** (ppy/n) -> nav * (ppy/n)``:
    61.0), 547 (``- 1.0 -> + 1.0``: 2.0) and 550 (``cagr / beta -> cagr *
    beta``: -0.5).
    """
    data = _int_data({"s": [-0.5, -0.25, 0.0]}, benchmark={"b": [0.0, 0.5, 1.0]})
    cov_matrix = np.cov(np.array([-0.5, -0.25, 0.0]), np.array([0.0, 0.5, 1.0]))
    beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])  # == 0.5 exactly
    nav_final = (1.0 - 0.5) * (1.0 - 0.25) * (1.0 + 0.0)  # == 0.375 exactly
    cagr = nav_final ** (252.0 / 3) - 1.0  # == -1.0 exactly (underflow vs 1.0)
    expected = cagr / beta  # == -2.0
    assert data.stats.treynor_ratio()["s"] == pytest.approx(expected, rel=TOL_FLOAT64)
