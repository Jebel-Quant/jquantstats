"""Mutation-killing tests for the basic-stats and Monte Carlo stats mixins.

Targets surviving ``mutmut`` mutants in ``_stats/_basic.py`` and
``_stats/_montecarlo.py``.  Each test is constructed so its assertion fails
under the targeted mutation:

``_basic.py``
- geometric_mean compound guard (31) — all-zero returns hit ``compound == 1``.
- volatility / autocorr / acf error-message mutants (47, 324, 328, 332, 336) —
  anchored ``match`` regexes pin the exact runtime text.
- ``std() is None`` fallbacks (52, 87, 97, 287) — a single non-null observation
  makes Polars return ``None`` for the sample std.
- VaR / CVaR sigma multiplier and defaults (84, 89, 90, 94) — explicit
  ``sigma=2.0`` pins the doubled-std quantile; no-arg calls pin the defaults.
- CVaR mask boundary (99) — ``alpha=0.5`` makes the VaR threshold equal the
  mean bit-exactly, so a value equal to the threshold separates ``<`` от ``<=``.
- ulcer/UPI/serenity (131, 133, 134, 159) — hand-computed exact values on tiny
  series, including a non-zero ``rf`` to pin the subtraction.
- ``_max_consecutive`` fallback and win predicate (201, 203).
- risk_of_ruin arithmetic (210, 211, 216, 218, 219) — exact ``(1/3)**5``.
- outlier quantile boundaries (249, 255) — a value exactly at the quantile.
- acf defaults and nlags boundary (330, 334, 335).

``_montecarlo.py``
- validation (4, 5, 6, 28, 30, 58, 59, 95, 96) — boundary values succeed and
  anchored messages pin the exact text.
- single-observation bootstrap (10, 40) — one usable value must be resampled,
  not mapped to NaN.
- block bootstrap internals (13, 14, 16, 18, 20, 22, 24) — with
  ``n_obs == block_size`` the simulation is fully deterministic (all block
  starts are 0), so every path must equal the tiled input series.
- draw-stream / block-size arithmetic (11, 15, 32, 33, 34) — seeded replication
  of the exact algorithm; any change to shapes, ranges or assembly changes the
  output frame.
- terminal/sharpe/drawdown/cagr arithmetic (47, 48, 50, 51, 66, 71, 72, 73, 79,
  80, 97, 101, 102, 106, 108-112) — deterministic paths with hand-replicated
  expected values.
- default arguments (44, 45, 53, 54, 75, 76, 90, 91) — no-arg call must equal
  the explicit ``(n=1000, period=252)`` call on data where the perturbed
  default changes the result.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from jquantstats import Data
from jquantstats._stats import Stats

from ..tolerances import TOL_FLOAT64, TOL_PARITY, TOL_PINNED

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_data(values: list[float | None], name: str = "r") -> Data:
    """Build a single-asset Data object with an integer index (252 ppy)."""
    returns = pl.DataFrame({name: pl.Series(name, values, dtype=pl.Float64)})
    index = pl.DataFrame({"idx": list(range(len(values)))})
    return Data(returns=returns, index=index)


def _replicate_paths(values: np.ndarray, n: int, period: int) -> np.ndarray:
    """Replicate the source block-bootstrap algorithm exactly (same RNG calls)."""
    block_size = max(1, round(period**0.5))
    block = min(block_size, values.size)
    n_blocks = math.ceil(period / block)
    max_start = max(1, values.size - block + 1)
    starts = np.random.randint(0, max_start, size=(n, n_blocks))
    idx = starts[:, :, np.newaxis] + np.arange(block)[np.newaxis, np.newaxis, :]
    idx = np.clip(idx, 0, values.size - 1)
    return values[idx].reshape(n, -1)[:, :period]


SIX_VALUES = [0.01, -0.02, 0.03, -0.04, 0.05, -0.06]
THREE_VALUES = [0.01, -0.02, 0.03]
NEG_VALUES = [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06]


# ══ _basic.py ═════════════════════════════════════════════════════════════════

# ── geometric_mean compound guard (31) ────────────────────────────────────────


def test_geometric_mean_compound_in_unit_interval_is_finite():
    """Compound in (0, 1] must NOT hit the NaN guard (kills ``compound <= 1``).

    All-zero returns give ``compound == 1`` exactly → geometric mean 0.0; a
    half-loss/half-gain series gives ``compound == 0.75`` → finite negative.
    Mutant 31 returns NaN for both.
    """
    flat = _make_data([0.0, 0.0, 0.0, 0.0, 0.0])
    result = flat.stats.geometric_mean()
    assert result["r"] == pytest.approx(0.0, abs=TOL_PINNED)

    losing = _make_data([-0.5, 0.5])
    result = losing.stats.geometric_mean()
    assert result["r"] == pytest.approx(math.sqrt(0.75) - 1.0, rel=TOL_PARITY)


# ── volatility (47, 52) ───────────────────────────────────────────────────────


def test_volatility_periods_type_error_message_is_exact():
    """Anchored match pins the exact TypeError text (kills mutant 47)."""
    data = _make_data(THREE_VALUES)
    with pytest.raises(TypeError, match=r"^Expected int or float for periods, got str$"):
        data.stats.volatility(periods="252")  # type: ignore[arg-type]


def test_volatility_with_undefined_std_is_zero():
    """One non-null value → std() is None → volatility 0.0 (kills 52: else 1.0)."""
    data = _make_data([0.01, None])
    result = data.stats.volatility()
    assert result["r"] == 0.0


# ── value_at_risk (84, 87) ────────────────────────────────────────────────────


def test_value_at_risk_sigma_multiplier_doubles_std():
    """``sigma=2.0`` must yield the doubled-std quantile (kills 84: ``sigma = std``)."""
    vals = [0.01, 0.02, 0.03, -0.01, -0.02]
    data = _make_data(vals)
    arr = np.array(vals)
    mu, std = arr.mean(), arr.std(ddof=1)

    result = data.stats.value_at_risk(sigma=2.0)
    assert result["r"] == pytest.approx(float(norm.ppf(0.05, mu, 2.0 * std)), rel=TOL_PARITY)
    # The mutant discards the 2.0 multiplier and lands on the single-std quantile.
    assert result["r"] != pytest.approx(float(norm.ppf(0.05, mu, std)), rel=TOL_PARITY)


def test_value_at_risk_with_undefined_std_is_nan():
    """One non-null value → scale 0 → NaN quantile (kills 87: else 1.0 gives a finite VaR)."""
    data = _make_data([0.01, None])
    result = data.stats.value_at_risk()
    assert math.isnan(result["r"])


# ── _conditional_value_at_risk_impl (89, 90, 94, 97, 99) ──────────────────────


def test_cvar_impl_default_sigma_matches_explicit():
    """No-sigma call equals ``sigma=1.0`` call (kills 89: default 2.0).

    On ``[0.0]*10 + [0.5]`` with ``alpha=0.95`` the threshold μ+1.645σ excludes
    the 0.5 outlier (CVaR 0.0), while the doubled threshold μ+3.29σ includes it
    (CVaR 0.5/11) — so the mutated default produces a different value.
    """
    data = _make_data([0.0] * 10 + [0.5])
    default = data.stats._conditional_value_at_risk_impl(alpha=0.95)
    explicit = data.stats._conditional_value_at_risk_impl(sigma=1.0, alpha=0.95)
    assert default["r"] == pytest.approx(explicit["r"], abs=TOL_PINNED)
    assert default["r"] == pytest.approx(0.0, abs=TOL_PINNED)


def test_cvar_impl_default_alpha_matches_explicit():
    """No-arg call equals explicit defaults (kills 90: alpha 1.05 → NaN quantile)."""
    data = _make_data([0.0] * 10 + [-0.5])
    default = data.stats._conditional_value_at_risk_impl()
    explicit = data.stats._conditional_value_at_risk_impl(sigma=1.0, alpha=0.05)
    assert default["r"] == pytest.approx(-0.5, abs=TOL_PINNED)
    assert explicit["r"] == pytest.approx(-0.5, abs=TOL_PINNED)


def test_cvar_impl_sigma_multiplier_widens_threshold():
    """``sigma=2.0`` widens the mask to include the outlier (kills 94: ``sigma = std``).

    Threshold μ+1.645·(2σ) ≈ 0.541 > 0.5 includes every value → mean 0.5/11.
    The mutant keeps μ+1.645·σ ≈ 0.293 < 0.5 → excludes the outlier → 0.0.
    """
    data = _make_data([0.0] * 10 + [0.5])
    result = data.stats._conditional_value_at_risk_impl(sigma=2.0, alpha=0.95)
    assert result["r"] == pytest.approx(0.5 / 11.0, rel=TOL_PARITY)


def test_cvar_impl_with_undefined_std_returns_lone_value():
    """A single observation degenerates to the observation itself.

    Behaviour pin: with std undefined the threshold collapses onto the lone
    value and the empty tail falls back to it.  (Mutant 97 — ``else 1.0`` —
    is equivalent on this only-reachable input and is accepted in the
    baseline instead.)
    """
    data = _make_data([0.01, None])
    result = data.stats._conditional_value_at_risk_impl(alpha=0.95)
    assert result["r"] == pytest.approx(0.01, abs=TOL_PINNED)


def test_cvar_mask_excludes_value_equal_to_var():
    """A value exactly at the VaR threshold must be excluded (kills 99: ``<=``).

    With ``alpha=0.5`` the threshold is ``norm.ppf(0.5, μ, σ) == μ`` bit-exactly,
    and μ of ``[-0.5, 0.0, 0.5]`` is exactly 0.0 — so the strict ``<`` keeps
    only -0.5 (CVaR -0.5) while ``<=`` would also keep 0.0 (CVaR -0.25).
    """
    data = _make_data([-0.5, 0.0, 0.5])
    result = data.stats._conditional_value_at_risk_impl(alpha=0.5)
    assert result["r"] == pytest.approx(-0.5, abs=TOL_PINNED)


# ── ulcer index family (131, 133, 134) ────────────────────────────────────────


def test_ulcer_performance_index_no_drawdown_is_nan():
    """Ui == 0 → NaN (kills 131: ``float("XXnanXX")`` raises; 133: ``ui == 1`` divides by 0)."""
    data = _make_data([0.01, 0.02])
    result = data.stats.ulcer_performance_index()
    assert math.isnan(result["r"])


def test_ulcer_performance_index_subtracts_rf():
    """Exact UPI with non-zero rf on returns [0.1, -0.5] (kills 134: ``comp + rf``).

    Drawdowns are [0, 0.5] exactly, ulcer index 0.5, comp = 1.1·0.5 − 1 = −0.45,
    so UPI(rf=0.1) = (−0.45 − 0.1)/0.5 = −1.1; the mutant gives −0.7.
    """
    data = _make_data([0.1, -0.5])
    assert data.stats.ulcer_index()["r"] == pytest.approx(0.5, abs=TOL_PINNED)
    comp = (1.1 * 0.5) - 1.0
    result = data.stats.ulcer_performance_index(rf=0.1)
    assert result["r"] == pytest.approx((comp - 0.1) / 0.5, rel=TOL_PARITY)


# ── serenity index (159) ──────────────────────────────────────────────────────


def test_serenity_index_subtracts_rf():
    """Exact serenity with non-zero rf (kills 159: ``sum + rf``).

    Four +1 % returns then −50 % give drawdowns [0,0,0,0,0.5] exactly; the lone
    −0.5 lies below the CVaR threshold μ+1.645σ ≈ −0.468, so CVaR = −0.5,
    pitfall = 0.5/std(returns), ui = 0.25.
    """
    vals = [0.01, 0.01, 0.01, 0.01, -0.5]
    data = _make_data(vals)
    arr = np.array(vals)
    dd_neg = np.array([0.0, 0.0, 0.0, 0.0, -0.5])

    var_threshold = float(norm.ppf(0.05, dd_neg.mean(), dd_neg.std(ddof=1)))
    cvar = dd_neg[dd_neg < var_threshold].mean()
    pitfall = -cvar / arr.std(ddof=1)
    ui = math.sqrt(float((dd_neg**2).sum()) / (len(vals) - 1))
    expected = (arr.sum() - 0.1) / (ui * pitfall)

    result = data.stats.serenity_index(rf=0.1)
    assert result["r"] == pytest.approx(expected, rel=TOL_PARITY)


# ── consecutive runs (201, 203) ───────────────────────────────────────────────


def test_max_consecutive_all_null_mask_is_zero():
    """An all-null mask yields a None aggregate → 0, not 1 (kills 201)."""
    mask = pl.Series([None, None, None], dtype=pl.Boolean)
    assert Stats._max_consecutive(mask) == 0


def test_consecutive_wins_excludes_zero_returns():
    """Zeros are not wins: run of three 0.0 must not count (kills 203: ``>= 0``)."""
    data = _make_data([0.0, 0.0, 0.0, 0.01, -0.02])
    assert data.stats.consecutive_wins()["r"] == 1


# ── risk_of_ruin (210, 211, 216, 218, 219) ────────────────────────────────────


def test_risk_of_ruin_exact_value():
    """Exact ((1−w)/(1+w))^n with w = 1/2, n = 5 (kills 210, 211, 216, 218, 219).

    Series [0.01, −0.02, 0, 0, 0]: 1 win over 2 non-zero periods → w = 0.5 →
    (0.5/1.5)^5 = (1/3)^5.  Every arithmetic/predicate mutant lands elsewhere
    (0.03125, 0.1317, 1.0, 0.2373, 0.00032 respectively).
    """
    data = _make_data([0.01, -0.02, 0.0, 0.0, 0.0])
    result = data.stats.risk_of_ruin()
    assert result["r"] == pytest.approx((1.0 / 3.0) ** 5, rel=TOL_PARITY)


# ── outlier thresholds (249, 255) ─────────────────────────────────────────────


def test_outliers_strictly_above_quantile():
    """A value exactly at the quantile is excluded (kills 249: ``>=``).

    quantile(0.75, linear) of [0.01..0.05] is exactly 0.04 (index 3·1.0), so
    strict ``>`` keeps only 0.05.
    """
    data = _make_data([0.01, 0.02, 0.03, 0.04, 0.05])
    result = data.stats.outliers(quantile=0.75)
    assert result["r"].to_list() == pytest.approx([0.05], abs=TOL_FLOAT64)


def test_remove_outliers_strictly_below_quantile():
    """A value exactly at the quantile is removed (kills 255: ``<=``)."""
    data = _make_data([0.01, 0.02, 0.03, 0.04, 0.05])
    result = data.stats.remove_outliers(quantile=0.75)
    assert result["r"].to_list() == pytest.approx([0.01, 0.02, 0.03], abs=TOL_FLOAT64)


# ── risk_return_ratio fallback (287) ──────────────────────────────────────────


def test_risk_return_ratio_with_undefined_std_uses_unit_denominator():
    """One non-null value → std None → mean / 1.0 (kills 287: else 2.0 halves it)."""
    data = _make_data([0.04, None])
    result = data.stats.risk_return_ratio()
    assert result["r"] == pytest.approx(0.04, abs=TOL_PINNED)


# ── autocorr / acf validation (324, 328, 330, 332, 334, 335, 336) ─────────────


def test_autocorr_lag_error_messages_are_exact():
    """Anchored matches pin the exact lag error texts (kills 324, 328)."""
    data = _make_data(THREE_VALUES)
    with pytest.raises(TypeError, match=r"^lag must be an int, got float$"):
        data.stats.autocorr(lag=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"^lag must be a positive integer, got 0$"):
        data.stats.autocorr(lag=0)


def test_acf_default_nlags_is_twenty():
    """No-arg acf() returns exactly 21 rows (kills 330: default 21 → 22 rows)."""
    rng = np.random.default_rng(3)
    data = _make_data(list(rng.normal(0.0, 0.01, 30)))
    result = data.stats.acf()
    assert result.height == 21
    assert result["lag"].to_list() == list(range(21))


def test_acf_nlags_zero_is_allowed():
    """nlags=0 is valid (non-negative) and yields just lag 0 (kills 334, 335)."""
    data = _make_data(THREE_VALUES)
    result = data.stats.acf(nlags=0)
    assert result.height == 1
    assert result["r"].to_list() == [1.0]


def test_acf_nlags_error_messages_are_exact():
    """Anchored matches pin the exact nlags error texts (kills 332, 336)."""
    data = _make_data(THREE_VALUES)
    with pytest.raises(TypeError, match=r"^nlags must be an int, got float$"):
        data.stats.acf(nlags=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"^nlags must be non-negative, got -1$"):
        data.stats.acf(nlags=-1)


# ══ _montecarlo.py ════════════════════════════════════════════════════════════

# ── validation (4, 5, 6, 28, 30) ──────────────────────────────────────────────


def test_montecarlo_validation_messages_are_exact():
    """Anchored matches pin the exact validation texts (kills 6, 28, 30)."""
    data = _make_data(THREE_VALUES)
    with pytest.raises(ValueError, match=r"^n must be a positive integer$"):
        data.stats.montecarlo(n=0, period=5)
    with pytest.raises(ValueError, match=r"^period must be a positive integer$"):
        data.stats.montecarlo(n=5, period=0)


def test_montecarlo_rejects_bool_n():
    """``n=True`` must be rejected despite bool ⊂ int (kills 5: or → and)."""
    data = _make_data(THREE_VALUES)
    with pytest.raises(ValueError, match=r"^n must be a positive integer$"):
        data.stats.montecarlo(n=True, period=5)


def test_montecarlo_accepts_one():
    """n=1 and period=1 are valid positive integers (kills 4: ``value <= 1``)."""
    data = _make_data(THREE_VALUES)
    result = data.stats.montecarlo(n=1, period=1)
    assert result.shape == (1, 1)


# ── single-observation bootstrap (10, 40) ─────────────────────────────────────


def test_montecarlo_single_observation_bootstraps_that_value():
    """One usable observation is resampled, not NaN (kills 10 and 40: size == 1 → NaN)."""
    data = _make_data([0.01, None])
    result = data.stats.montecarlo(n=3, period=4)
    expected = float(np.prod(np.full(4, 1.01)) - 1.0)
    assert result["r"].to_list() == pytest.approx([expected] * 3, rel=TOL_PARITY)


# ── deterministic full-series blocks (13, 14, 16, 18, 20, 22, 24, 47, 48, 50, 51) ─


def test_montecarlo_is_deterministic_when_nobs_equals_block_size():
    """With n_obs == block_size every path is the tiled series.

    period=9 → block_size=3 == n_obs → max_start = 1 → all 200×3 block starts
    are 0 and every terminal return equals ``∏(1+v)³ − 1`` exactly.  Kills:
    13/16 (max_start ≥ 2 introduces random starts), 14 (max_start 7), 18
    (``randint(1, 1)`` raises), 20 (idx 0−[0,1,2] → [v0,v0,v0] blocks), 22
    (clip lower 1 → [v1,v1,v2]), 24 (clip upper n_obs−2 → [v0,v1,v1]), and the
    terminal-return arithmetic mutants 47, 48, 50, 51.
    """
    data = _make_data(THREE_VALUES)
    np.random.seed(11)
    result = data.stats.montecarlo(n=200, period=9)
    expected = float(np.prod(1.0 + np.tile(np.array(THREE_VALUES), 3)) - 1.0)
    assert result.shape == (200, 1)
    np.testing.assert_allclose(result["r"].to_numpy(), np.full(200, expected), rtol=0.0, atol=TOL_FLOAT64)


# ── seeded replication of the draw stream (11, 15, 33, 34) ────────────────────


def test_montecarlo_matches_seeded_replication():
    """Seeded output equals an exact replica of the algorithm.

    n=32, period=10, n_obs=6 → block 3, n_blocks 4, max_start 4.  Any change to
    the number of blocks (11), the randint range (15), or the block size (33,
    34) alters the seeded draw stream and thus the output frame.
    """
    values = np.array(SIX_VALUES)
    data = _make_data(SIX_VALUES)

    np.random.seed(1234)
    expected = np.prod(1.0 + _replicate_paths(values, n=32, period=10), axis=1) - 1.0
    np.random.seed(1234)
    result = data.stats.montecarlo(n=32, period=10)
    np.testing.assert_allclose(result["r"].to_numpy(), expected, rtol=0.0, atol=TOL_FLOAT64)


def test_montecarlo_small_period_uses_unit_block():
    """period=2 → block_size 1, two independent draws per path (kills 32: max(2, ...)).

    The mutant forces block_size 2 == n_obs, collapsing every path to the
    deterministic full series; the seeded replica has per-path variety.
    """
    values = np.array([0.1, 0.2])
    data = _make_data([0.1, 0.2])

    np.random.seed(7)
    expected = np.prod(1.0 + _replicate_paths(values, n=64, period=2), axis=1) - 1.0
    np.random.seed(7)
    result = data.stats.montecarlo(n=64, period=2)
    np.testing.assert_allclose(result["r"].to_numpy(), expected, rtol=0.0, atol=TOL_FLOAT64)


# ── default arguments (44, 45, 53, 54, 75, 76, 90, 91) ────────────────────────
#
# With 6 observations the default period=252 gives block=6, max_start=1: the
# simulation is fully deterministic (all starts 0), so the no-arg call must
# equal the explicit-default call bit for bit.  Monotonically negative returns
# make the 253rd path element change every statistic (incl. max drawdown).


def test_montecarlo_default_args_match_explicit():
    """montecarlo() equals montecarlo(n=1000, period=252) (kills 44, 45)."""
    data = _make_data(NEG_VALUES)
    default = data.stats.montecarlo()
    explicit = data.stats.montecarlo(n=1000, period=252)
    assert default.shape == (1000, 1)
    np.testing.assert_array_equal(default.to_numpy(), explicit.to_numpy())


def test_montecarlo_sharpe_default_args_match_explicit():
    """montecarlo_sharpe() equals the explicit-default call (kills 53, 54)."""
    data = _make_data(NEG_VALUES)
    default = data.stats.montecarlo_sharpe()
    explicit = data.stats.montecarlo_sharpe(n=1000, period=252)
    assert default.shape == (1000, 1)
    np.testing.assert_array_equal(default.to_numpy(), explicit.to_numpy())


def test_montecarlo_drawdown_default_args_match_explicit():
    """montecarlo_drawdown() equals the explicit-default call (kills 75, 76)."""
    data = _make_data(NEG_VALUES)
    default = data.stats.montecarlo_drawdown()
    explicit = data.stats.montecarlo_drawdown(n=1000, period=252)
    assert default.shape == (1000, 1)
    np.testing.assert_array_equal(default.to_numpy(), explicit.to_numpy())


def test_montecarlo_cagr_default_args_match_explicit():
    """montecarlo_cagr() equals the explicit-default call (kills 90, 91)."""
    data = _make_data(NEG_VALUES)
    default = data.stats.montecarlo_cagr()
    explicit = data.stats.montecarlo_cagr(n=1000, period=252)
    assert default.shape == (1000, 1)
    np.testing.assert_array_equal(default.to_numpy(), explicit.to_numpy())


# ── montecarlo_sharpe (58, 59, 66, 71, 72, 73) ────────────────────────────────


def test_montecarlo_sharpe_ppy_boundary_and_message():
    """periods_per_year=1 is valid; 0 raises the exact message (kills 58, 59)."""
    data = _make_data(THREE_VALUES)
    result = data.stats.montecarlo_sharpe(n=2, period=9, periods_per_year=1)
    assert result.shape == (2, 1)
    with pytest.raises(ValueError, match=r"^periods_per_year must be positive$"):
        data.stats.montecarlo_sharpe(n=2, period=9, periods_per_year=0)


def test_montecarlo_sharpe_exact_value_on_deterministic_paths():
    """Exact mean/std(ddof=1)·√252 on the tiled path (kills 66, 72, 73)."""
    data = _make_data(THREE_VALUES)
    path = np.tile(np.array(THREE_VALUES), 3)
    expected = path.mean() / path.std(ddof=1) * math.sqrt(252.0)
    result = data.stats.montecarlo_sharpe(n=5, period=9, periods_per_year=252)
    assert result["r"].to_list() == pytest.approx([expected] * 5, rel=TOL_PARITY)


def test_montecarlo_sharpe_zero_variance_is_nan():
    """Constant paths (single observation) → std 0 → NaN (kills 71: ``stds == 1.0``)."""
    data = _make_data([0.01, None])
    result = data.stats.montecarlo_sharpe(n=3, period=5, periods_per_year=252)
    assert all(math.isnan(v) for v in result["r"].to_list())


# ── montecarlo_drawdown (79, 80) ──────────────────────────────────────────────


def test_montecarlo_drawdown_exact_value_on_deterministic_paths():
    """Exact max drawdown of the tiled path (kills 79, 80)."""
    data = _make_data(THREE_VALUES)
    path = np.tile(np.array(THREE_VALUES), 3)
    nav = np.cumprod(1.0 + path)
    expected = float(np.min(nav / np.maximum.accumulate(nav) - 1.0))
    result = data.stats.montecarlo_drawdown(n=5, period=9)
    assert result["r"].to_list() == pytest.approx([expected] * 5, rel=TOL_PARITY)


# ── montecarlo_cagr (95, 96, 97, 101, 102, 106, 108-112) ──────────────────────


def test_montecarlo_cagr_ppy_boundary_and_message():
    """periods_per_year=1 is valid; 0 raises the exact message (kills 95, 96)."""
    data = _make_data(THREE_VALUES)
    result = data.stats.montecarlo_cagr(n=2, period=9, periods_per_year=1)
    assert result.shape == (2, 1)
    with pytest.raises(ValueError, match=r"^periods_per_year must be positive$"):
        data.stats.montecarlo_cagr(n=2, period=9, periods_per_year=0)


def test_montecarlo_cagr_exact_value_on_deterministic_paths():
    """Exact totals^(1/years) − 1 on the tiled path (kills 97, 101, 102, 108-112)."""
    data = _make_data(THREE_VALUES)
    path = np.tile(np.array(THREE_VALUES), 3)
    totals = float(np.prod(1.0 + path))
    years = 9 / 252
    expected = totals ** (1.0 / years) - 1.0
    result = data.stats.montecarlo_cagr(n=5, period=9, periods_per_year=252)
    assert result["r"].to_list() == pytest.approx([expected] * 5, rel=TOL_PARITY)


def test_montecarlo_cagr_total_loss_is_nan():
    """A −100 % return makes totals exactly 0 → NaN (kills 106: ``totals >= 0`` → −1.0)."""
    data = _make_data([-1.0, -1.0])
    result = data.stats.montecarlo_cagr(n=3, period=4, periods_per_year=252)
    assert all(math.isnan(v) for v in result["r"].to_list())
