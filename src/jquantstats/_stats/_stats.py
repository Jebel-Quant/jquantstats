"""Statistical analysis tools for financial returns data.

This module provides the `Stats` dataclass, which is the public-facing
class that combines five mixin classes:

- `_BasicStatsMixin` — basic statistics,
  volatility, win/loss metrics, and risk metrics (VaR, Sharpe inputs, Kelly).
- `_RiskStatsMixin` — the risk-adjusted ratio family: Sharpe, Sortino, Omega and
  their probabilistic / smart / adjusted variants.
- `_ConcentrationStatsMixin` — Herfindahl-Hirschman concentration of gains and
  losses (`hhi_positive`, `hhi_negative`).
- `_BenchmarkStatsMixin` — benchmark-relative and factor analytics (R², alpha,
  beta, information ratio, Treynor).
- `_DrawdownMixin` — cumulative returns, drawdown series, max drawdown, and
  per-episode drawdown details.
- `_ReportingStatsMixin` — temporal reporting: CAGR, expected return, RAR,
  Calmar, recovery factor, drawdown duration, monthly win rate.
- `_CaptureStatsMixin` — up- and down-market capture ratios.
- `_SummaryStatsMixin` — the tidy `summary` table and its `annual_breakdown`.
- `_PeriodicReportingMixin` — period-bucketed tables: monthly-returns pivot,
  distribution across calendar frequencies, benchmark comparison, worst-N periods.
- `_RollingStatsMixin` — rolling-window
  time-series metrics (rolling Sharpe, Sortino, and volatility).
- `_MonteCarloStatsMixin` — block-bootstrap Monte Carlo simulation distributions
  for total return, Sharpe, max drawdown, and CAGR.

Module-level helpers and the ``columnwise_stat`` / ``to_frame`` decorators are
defined in `jquantstats._stats._core` and re-exported here for backwards
compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ._basic import _BasicStatsMixin
from ._benchmark import _BenchmarkStatsMixin
from ._capture import _CaptureStatsMixin
from ._concentration import _ConcentrationStatsMixin
from ._core import (
    _drawdown_series,
    _mean,
    _to_float,
    columnwise_stat,
    to_frame,
)
from ._drawdown import _DrawdownMixin
from ._internals import (
    _annualization_factor,
    _comp_return,
    _downside_deviation,
    _nav_series,
)
from ._montecarlo import _MonteCarloStatsMixin
from ._performance import _RiskStatsMixin
from ._periodic import _PeriodicReportingMixin
from ._reporting import _ReportingStatsMixin
from ._rolling import _RollingStatsMixin
from ._summary import _SummaryStatsMixin

if TYPE_CHECKING:
    from ..data import Data

__all__ = [
    "Stats",
    "_annualization_factor",
    "_comp_return",
    "_downside_deviation",
    "_drawdown_series",
    "_mean",
    "_nav_series",
    "_to_float",
    "columnwise_stat",
    "to_frame",
]


class Stats(
    _BasicStatsMixin,
    _RiskStatsMixin,
    _ConcentrationStatsMixin,
    _BenchmarkStatsMixin,
    _DrawdownMixin,
    _ReportingStatsMixin,
    _CaptureStatsMixin,
    _SummaryStatsMixin,
    _PeriodicReportingMixin,
    _RollingStatsMixin,
    _MonteCarloStatsMixin,
):
    """Statistical analysis tools for financial returns data.

    Provides a comprehensive set of methods for calculating various financial
    metrics and statistics on returns data, including:

    - Basic statistics (mean, skew, kurtosis)
    - Risk metrics (volatility, value-at-risk, drawdown)
    - Performance ratios (Sharpe, Sortino, information ratio)
    - Win/loss metrics (win rate, profit factor, payoff ratio)
    - Rolling calculations (rolling volatility, rolling Sharpe)
    - Factor analysis (alpha, beta, R-squared)
    - Concentration metrics (``hhi_positive``, ``hhi_negative``) — optional
      Herfindahl-Hirschman Index diagnostics that quantify how concentrated
      gains and losses are across time periods.  These are public API but are
      not included in ``summary()`` by default.

    Metrics are organised into focused modules:

    - `_BasicStatsMixin`
    - `_RiskStatsMixin`
    - `_ConcentrationStatsMixin`
    - `_BenchmarkStatsMixin`
    - `_DrawdownMixin`
    - `_ReportingStatsMixin`
    - `_CaptureStatsMixin`
    - `_SummaryStatsMixin`
    - `_PeriodicReportingMixin`
    - `_RollingStatsMixin`
    - `_MonteCarloStatsMixin`

    Attributes:
        all: A DataFrame combining all data (index, returns, benchmark) for
            easy column selection.
    """

    def __init__(self, data: Data) -> None:
        self._data = data
        self.all: pl.DataFrame = data.all

    def __repr__(self) -> str:
        """Return a string representation of the Stats object."""
        return f"Stats(assets={self._data.assets})"

    @property
    def assets(self) -> list[str]:
        """Asset column names (excludes benchmark and date)."""
        return self._data.assets

    @property
    def returns(self) -> pl.DataFrame:
        """Returns DataFrame (asset columns only, no benchmark)."""
        return self._data.returns

    @property
    def benchmark(self) -> pl.DataFrame | None:
        """Benchmark DataFrame, or None when no benchmark was provided."""
        return self._data.benchmark

    @property
    def date_col(self) -> list[str]:
        """Date column name(s) present in the index, or empty list."""
        return self._data.date_col

    @property
    def index(self) -> pl.DataFrame:
        """Index DataFrame (date or integer range)."""
        return self._data.index
