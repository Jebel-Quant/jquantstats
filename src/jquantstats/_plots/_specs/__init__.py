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

__all__ = [
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
    "yearly_returns_spec",
]
