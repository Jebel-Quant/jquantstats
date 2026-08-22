"""Spec builders for the portfolio diagnostic charts.

These ask questions about the strategy rather than reporting its returns: how
sensitive is it to execution delay, how correlated are its holdings, which
months carried it, and how much trading cost it can absorb.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

from .._spec import Axis, BarSeries, FigureSpec, HeatmapGrid, LineSeries, Panel, RefLine

if TYPE_CHECKING:
    from .._protocol import PortfolioLike

__all__ = [
    "correlation_heatmap_spec",
    "lead_lag_ir_spec",
    "monthly_returns_heatmap_spec",
    "trading_cost_impact_spec",
]

_MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# The undelayed portfolio is the one being judged, so it is picked out from the
# lagged variants around it.
_FOCUS_COLOR = "red"
_NEUTRAL_COLOR = "#1f77b4"

# A correlation runs from -1 to 1 whatever this particular data happens to span,
# so the ramp is pinned rather than fitted.
_CORRELATION_RANGE = (-1, 1)
_CORRELATION_SIZE = (700, 600)

_MARKER_SIZE = 6


def lead_lag_ir_spec(portfolio: PortfolioLike, start: int, end: int) -> FigureSpec:
    """Describe the Sharpe-by-execution-delay chart.

    Shifting the positions forward and backward shows how much of the
    strategy's edge depends on trading precisely when it says to. A peak away
    from zero suggests the signal is mistimed.

    Args:
        portfolio: The portfolio to plot.
        start: First lag to include.
        end: Last lag to include.

    Returns:
        FigureSpec: One bar per lag, with the undelayed one picked out.

    Raises:
        TypeError: If *start* or *end* is not an integer.

    """
    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError
    if start > end:
        start, end = end, start

    lags = list(range(start, end + 1))
    sharpes: list[float] = []
    for n in lags:
        lagged = portfolio if n == 0 else portfolio.lag(n)
        value = lagged.stats.sharpe().get("returns", float("nan"))
        sharpes.append(float(value) if value is not None else float("nan"))

    panel = Panel(
        bars=(
            BarSeries(
                name="Sharpe by lag",
                x=lags,
                y=sharpes,
                colors=tuple(_FOCUS_COLOR if lag == 0 else _NEUTRAL_COLOR for lag in lags),
            ),
        ),
        xaxis=Axis(title="Lag (steps)"),
        yaxis=Axis(title="Sharpe ratio"),
    )
    return FigureSpec(
        title="Lead/Lag Information Ratio (Sharpe) by Lag",
        panels=(panel,),
        height=None,
        chrome="plain",
        hover_mode="x",
    )


def correlation_heatmap_spec(portfolio: PortfolioLike, frame: pl.DataFrame | None, name: str, title: str) -> FigureSpec:
    """Describe the correlation matrix of the holdings and the portfolio.

    Args:
        portfolio: The portfolio to plot.
        frame: Series to correlate against, or None for the portfolio's prices.
        name: Column name to give the portfolio's own profit series.
        title: Chart title.

    Returns:
        FigureSpec: A square matrix pinned to the full [-1, 1] range.

    """
    frame = portfolio.prices if frame is None else frame
    corr = portfolio.correlation(frame, name=name)
    labels = tuple(corr.columns)
    values = tuple(tuple(row) for row in corr.rows())
    low, high = _CORRELATION_RANGE
    width, height = _CORRELATION_SIZE

    panel = Panel(
        heatmap=HeatmapGrid(
            x_labels=labels,
            y_labels=labels,
            z=values,
            text=tuple(tuple(f"{value:.2f}" for value in row) for row in values),
            colorscale="rdbu_r",
            # A diverging ramp centred on zero: uncorrelated reads as neutral.
            zmid=0,
            zmin=low,
            zmax=high,
            colorbar_title="Correlation",
            # px.imshow generated a tooltip automatically; going to a plain
            # heatmap means asking for one, or the chart silently loses it.
            hover_label="Correlation",
        ),
        # The labels are the series names; repeating them as axis titles would
        # say nothing.
        xaxis=Axis(title=""),
        yaxis=Axis(title=""),
    )
    return FigureSpec(title=title, panels=(panel,), height=height, width=width, chrome="bare")


def monthly_returns_heatmap_spec(portfolio: PortfolioLike) -> FigureSpec:
    """Describe the portfolio's monthly-returns calendar.

    Args:
        portfolio: The portfolio to plot.

    Returns:
        FigureSpec: A year-by-month grid.

    """
    monthly = portfolio.monthly
    years = monthly["year"].unique().sort().to_list()

    z: list[tuple[float | None, ...]] = []
    text: list[tuple[str, ...]] = []
    for year in years:
        rows = monthly.filter(pl.col("year") == year)
        by_month = {int(row["month"]): float(row["returns"]) for row in rows.iter_rows(named=True)}
        percents = [by_month[m] * 100.0 if m in by_month else None for m in range(1, 13)]
        z.append(tuple(percents))
        text.append(tuple("" if value is None else f"{value:.1f}%" for value in percents))

    panel = Panel(
        heatmap=HeatmapGrid(
            x_labels=_MONTH_NAMES,
            y_labels=tuple(str(year) for year in years),
            z=tuple(z),
            text=tuple(text),
            colorscale="rdylgn",
            colorbar_title="Return (%)",
            hover_label="Return",
        ),
        xaxis=Axis(title="Month"),
        # Years are labels, not a continuous scale: 2023 and 2024 are adjacent
        # rows, not a year apart on a number line.
        yaxis=Axis(title="Year", kind="category"),
    )
    return FigureSpec(title="Monthly Returns Heatmap", panels=(panel,), height=None, chrome="bare")


def trading_cost_impact_spec(portfolio: PortfolioLike, max_bps: int) -> FigureSpec:
    """Describe how trading cost erodes the Sharpe ratio.

    Args:
        portfolio: The portfolio to plot.
        max_bps: Highest one-way cost to evaluate, in basis points.

    Returns:
        FigureSpec: Sharpe against cost, with the zero-cost level marked when
        it is finite.

    Raises:
        ValueError: If *max_bps* is not a positive integer.

    """
    impact = portfolio.trading_cost_impact(max_bps=max_bps)
    costs = impact["cost_bps"].to_list()
    sharpes = impact["sharpe"].to_list()

    baseline = float(sharpes[0]) if sharpes and sharpes[0] is not None else float("nan")
    # An unmeasurable baseline gets no line rather than one drawn at NaN.
    ref_lines = () if math.isnan(baseline) else (RefLine(value=baseline, dash="dash", label="0 bps baseline"),)

    panel = Panel(
        lines=(
            LineSeries(
                name="Sharpe (cost-adjusted)",
                x=costs,
                y=sharpes,
                color=_NEUTRAL_COLOR,
                markers=True,
                marker_size=_MARKER_SIZE,
            ),
        ),
        ref_lines=ref_lines,
        # One tick per basis point: the axis spans a couple of dozen integers.
        xaxis=Axis(title="One-way cost (basis points)", dtick=1),
        yaxis=Axis(title="Annualised Sharpe ratio"),
    )
    return FigureSpec(
        # Escaped rather than written literally: the title has always used an en
        # dash, and ruff flags the bare character as visually ambiguous.
        title=f"Trading Cost Impact on Sharpe Ratio (0\u2013{max_bps} bps)",
        panels=(panel,),
        height=None,
        chrome="plain",
        hover_mode="x unified",
    )
