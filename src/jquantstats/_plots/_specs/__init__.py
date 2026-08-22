"""Spec builders: polars in, `~jquantstats._plots._spec.FigureSpec` out.

One module per chart family. Nothing here imports a drawing library, so the
arithmetic behind a chart is written once no matter how many backends render it.
"""

from ._cumulative import compare_spec as compare_spec
from ._cumulative import cumulative_returns_spec as cumulative_returns_spec
from ._cumulative import earnings_spec as earnings_spec
from ._cumulative import log_returns_spec as log_returns_spec
from ._drawdown import compute_drawdown_periods as compute_drawdown_periods
from ._drawdown import drawdown_spec as drawdown_spec
from ._drawdown import drawdowns_periods_spec as drawdowns_periods_spec
from ._periodic import daily_returns_spec as daily_returns_spec
from ._periodic import monthly_heatmap_spec as monthly_heatmap_spec
from ._periodic import monthly_returns_spec as monthly_returns_spec
from ._periodic import period_agg_exprs as period_agg_exprs
from ._periodic import yearly_returns_spec as yearly_returns_spec
from ._rolling import annual_sharpe_spec as annual_sharpe_spec
from ._rolling import portfolio_rolling_sharpe_spec as portfolio_rolling_sharpe_spec
from ._rolling import portfolio_rolling_volatility_spec as portfolio_rolling_volatility_spec
from ._rolling import rolling_beta_expr as rolling_beta_expr
from ._rolling import rolling_beta_spec as rolling_beta_spec
from ._rolling import rolling_sharpe_spec as rolling_sharpe_spec
from ._rolling import rolling_sortino_spec as rolling_sortino_spec
from ._rolling import rolling_volatility_spec as rolling_volatility_spec
from ._rolling import validate_window as validate_window

__all__ = [
    "annual_sharpe_spec",
    "compare_spec",
    "compute_drawdown_periods",
    "cumulative_returns_spec",
    "daily_returns_spec",
    "drawdown_spec",
    "drawdowns_periods_spec",
    "earnings_spec",
    "log_returns_spec",
    "monthly_heatmap_spec",
    "monthly_returns_spec",
    "period_agg_exprs",
    "portfolio_rolling_sharpe_spec",
    "portfolio_rolling_volatility_spec",
    "rolling_beta_expr",
    "rolling_beta_spec",
    "rolling_sharpe_spec",
    "rolling_sortino_spec",
    "rolling_volatility_spec",
    "validate_window",
    "yearly_returns_spec",
]
