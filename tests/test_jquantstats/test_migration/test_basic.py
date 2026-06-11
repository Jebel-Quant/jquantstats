"""Parametrized migration tests comparing basic stats against quantstats."""

import pytest
import quantstats as qs

from ..tolerances import TOL_FLOAT64


@pytest.mark.parametrize(
    ("method", "kwargs", "tol"),
    [
        ("sharpe", {"periods": 252}, 1e-12),
        ("skew", {}, 1e-12),
        ("kurtosis", {}, 1e-12),
        ("avg_return", {}, 1e-12),
        ("avg_win", {}, 1e-12),
        ("avg_loss", {}, 1e-12),
        ("volatility", {"periods": 252}, 1e-12),
        ("payoff_ratio", {}, 1e-12),
        ("profit_factor", {}, 1e-12),
        ("profit_ratio", {}, 1e-12),
        ("value_at_risk", {}, 1e-12),
        ("conditional_value_at_risk", {}, 1e-12),
        ("win_rate", {}, 1e-12),
        ("gain_to_pain_ratio", {}, 1e-12),
        ("risk_return_ratio", {}, 1e-12),
        ("best", {}, 1e-12),
        ("worst", {}, 1e-12),
        ("exposure", {}, 1e-12),
        ("kelly_criterion", {}, 1e-12),
        ("risk_of_ruin", {}, 1e-12),
        ("consecutive_wins", {}, 1e-12),
        ("consecutive_losses", {}, 1e-12),
        ("autocorr_penalty", {}, 1e-12),
        ("ulcer_index", {}, 1e-12),
        ("ulcer_performance_index", {}, 1e-12),
        ("serenity_index", {}, 1e-12),
        ("tail_ratio", {}, 1e-12),
        ("cpc_index", {}, 1e-12),
        ("common_sense_ratio", {}, 1e-12),
        ("geometric_mean", {}, 1e-12),
        ("comp", {}, 1e-12),
        ("smart_sharpe", {"periods": 252}, 1e-12),
        ("smart_sortino", {"periods": 252}, 1e-12),
        ("probabilistic_sortino_ratio", {}, 1e-12),
        ("probabilistic_adjusted_sortino_ratio", {}, 1e-12),
        ("outlier_win_ratio", {}, 1e-12),
        ("outlier_loss_ratio", {}, 1e-12),
        ("sortino", {"periods": 252}, 1e-12),
        ("adjusted_sortino", {"periods": 252}, 1e-12),
        ("probabilistic_sharpe_ratio", {}, 1e-4),
        ("max_drawdown", {}, 1e-12),
        ("cagr", {"periods": 252}, 1e-12),
        ("rar", {}, 1e-12),
        ("calmar", {"periods": 252}, 1e-12),
        ("recovery_factor", {}, 1e-12),
        ("omega", {"periods": 252}, 1e-12),
        ("omega", {"periods": 252, "required_return": 0.01}, 1e-12),
        ("omega", {"periods": 252, "rf": 0.02}, 1e-12),
    ],
)
def test_migration(stats, method, kwargs, tol):
    """Verify each stat method produces the same result as quantstats."""
    aapl = stats.all.to_pandas().set_index("Date")["AAPL"]
    x = getattr(stats, method)(**kwargs)["AAPL"]
    y = getattr(qs.stats, method)(aapl, **kwargs)
    assert x == pytest.approx(y, abs=tol)


def test_conditional_value_at_risk(stats):
    """Test conditional_value_at_risk matches QuantStats output."""
    returns = stats.all.to_pandas().set_index("Date")["AAPL"]
    x = qs.stats.conditional_value_at_risk(returns, confidence=0.99)
    y = stats.conditional_value_at_risk(confidence=0.99)["AAPL"]
    assert x == pytest.approx(y, abs=TOL_FLOAT64)
