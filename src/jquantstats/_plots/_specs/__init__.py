"""Spec builders: polars in, `~jquantstats._plots._spec.FigureSpec` out.

One module per chart family. Nothing here imports a drawing library, so the
arithmetic behind a chart is written once no matter how many backends render it.
"""

from ._cumulative import compare_spec as compare_spec
from ._cumulative import cumulative_returns_spec as cumulative_returns_spec
from ._cumulative import earnings_spec as earnings_spec
from ._cumulative import log_returns_spec as log_returns_spec

__all__ = [
    "compare_spec",
    "cumulative_returns_spec",
    "earnings_spec",
    "log_returns_spec",
]
