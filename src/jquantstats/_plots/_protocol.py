"""Protocols describing the minimal interfaces required by the _plots subpackage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

from jquantstats._cost_model import CostModel
from jquantstats._protocol import DataLike

__all__ = ["DataLike", "PortfolioLike", "PortfolioStatsLike"]


class PortfolioStatsLike(Protocol):  # pragma: no cover
    """Structural interface for the stats facade used by `PortfolioPlots`."""

    def sharpe(self) -> dict[str, float]:
        """Annualised Sharpe ratio per asset column."""
        ...

    def rolling_sharpe(
        self,
        rolling_period: int = 126,
        periods_per_year: int | float | None = None,
    ) -> pl.DataFrame:
        """Rolling Sharpe ratio series."""
        ...

    def rolling_volatility(
        self,
        rolling_period: int = 126,
        periods_per_year: int | float | None = None,
        annualize: bool = True,
    ) -> pl.DataFrame:
        """Rolling volatility series."""
        ...

    def annual_breakdown(self) -> pl.DataFrame:
        """Per-year metric breakdown."""
        ...


@runtime_checkable
class PortfolioLike(Protocol):  # pragma: no cover
    """Structural interface required by the `PortfolioPlots` class.

    Any object satisfying this protocol can be passed as ``portfolio`` without a
    concrete dependency on `Portfolio`.
    """

    prices: pl.DataFrame
    aum: float
    cost_model: CostModel

    @property
    def nav_accumulated(self) -> pl.DataFrame:
        """Accumulated NAV series."""
        ...

    @property
    def tilt(self) -> PortfolioLike:
        """Tilt component portfolio."""
        ...

    @property
    def timing(self) -> PortfolioLike:
        """Timing component portfolio."""
        ...

    @property
    def net_cost_nav(self) -> pl.DataFrame:
        """Net-of-cost accumulated NAV series."""
        ...

    @property
    def drawdown(self) -> pl.DataFrame:
        """Drawdown series."""
        ...

    @property
    def assets(self) -> list[str]:
        """Asset names."""
        ...

    @property
    def monthly(self) -> pl.DataFrame:
        """Monthly returns grouped by year and month."""
        ...

    @property
    def profits(self) -> pl.DataFrame:
        """Per-period profit series."""
        ...

    @property
    def stats(self) -> PortfolioStatsLike:
        """Statistics facade (rolling_sharpe, rolling_volatility, annual_breakdown, sharpe)."""
        ...

    def lag(self, n: int) -> PortfolioLike:
        """Return a lagged copy of this portfolio."""
        ...

    def smoothed_holding(self, n: int) -> PortfolioLike:
        """Return a smoothed-holdings copy of this portfolio."""
        ...

    def trading_cost_impact(self, max_bps: int = 20) -> pl.DataFrame:
        """Return a DataFrame of Sharpe vs. one-way trading costs."""
        ...

    def correlation(self, frame: pl.DataFrame, name: str = "portfolio") -> pl.DataFrame:
        """Return the correlation matrix including the portfolio profit series."""
        ...
