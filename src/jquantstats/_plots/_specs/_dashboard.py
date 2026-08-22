"""Spec builders for the two dashboards and the NAV comparison charts.

A dashboard stacks several views of the same period over a shared time axis, so
a drawdown lines up with the return that caused it. `data_snapshot_spec` builds
the returns-series version and `portfolio_snapshot_spec` the portfolio one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from .._spec import Axis, BarSeries, FigureSpec, HoverSpec, LineSeries, Panel, RefLine
from .._style import hex_to_rgba, ticker_colors

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike

    from .._protocol import PortfolioLike

__all__ = [
    "data_snapshot_spec",
    "lagged_performance_spec",
    "portfolio_snapshot_spec",
    "smoothed_holdings_performance_spec",
]

# A dashboard is tall: three panels of detail need the room.
_DASHBOARD_HEIGHT_PX = 1200
_PANEL_GAP = 0.05

# The headline panel earns the most height; the supporting ones split the rest.
_DATA_PANEL_HEIGHTS = (0.5, 0.25, 0.25)
_PORTFOLIO_PANEL_HEIGHTS = (0.66, 0.33)

# Drawdown fill and the faded half of a two-tone bar.
_LIGHT_ALPHA = 0.5
_MONTHLY_OPACITY = 0.8

# With one asset a bar's colour can carry the sign outright.
_SINGLE_POSITIVE = "green"
_SINGLE_NEGATIVE = "red"

# NAV comparison lines are drawn fine: there is one per lag or window, and the
# point is the spread between them.
_NAV_WIDTH = 1

_DEFAULT_LAGS = [0, 1, 2, 3, 4]


def _split_columns(frame: pl.DataFrame) -> tuple[str, list[str]]:
    """Separate the date column from the value columns.

    Args:
        frame: A frame whose first column is the date axis.

    Returns:
        tuple[str, list[str]]: The date column name and every other column.

    """
    date_col = frame.columns[0]
    return date_col, [c for c in frame.columns if c != date_col]


def data_snapshot_spec(data: DataLike, log_scale: bool) -> FigureSpec:
    """Describe the three-panel returns dashboard.

    Cumulative returns, drawdowns and monthly returns over one shared time
    axis, so a drawdown can be read against the months that produced it.

    Args:
        data: The dataset to plot.
        log_scale: Use a logarithmic scale for cumulative returns.

    Returns:
        FigureSpec: Three stacked panels sharing an x-axis.

    """
    returns = data.all
    date_col, tickers = _split_columns(returns)
    colors = ticker_colors(tickers)
    light = {ticker: hex_to_rgba(colors[ticker], _LIGHT_ALPHA) for ticker in tickers}

    prices = returns.with_columns([((1 + pl.col(t)).cum_prod()).alias(f"{t}_price") for t in tickers])
    monthly = returns.group_by_dynamic(
        index_column=date_col, every="1mo", period="1mo", closed="right", label="right"
    ).agg([((pl.col(t) + 1.0).product() - 1.0).alias(t) for t in tickers])

    cumulative = Panel(
        lines=tuple(
            LineSeries(
                name=ticker,
                x=prices[date_col],
                y=prices[f"{ticker}_price"],
                color=colors[ticker],
                legend_group=ticker,
                show_legend=True,
                hover=HoverSpec(label=ticker, value_format="float2", suffix="x"),
            )
            for ticker in tickers
        ),
        title="Cumulative Returns",
        yaxis=Axis(title="Cumulative Return", tick_format="float2", log=log_scale),
        height_ratio=_DATA_PANEL_HEIGHTS[0],
    )

    drawdowns = Panel(
        lines=tuple(
            LineSeries(
                name=ticker,
                x=prices[date_col],
                y=_drawdown_values(prices, f"{ticker}_price"),
                color=colors[ticker],
                width=1,
                fill=True,
                fill_color=light[ticker],
                legend_group=ticker,
                show_legend=False,
                hover=HoverSpec(label=f"{ticker} Drawdown", value_format="percent2", date_header=False),
            )
            for ticker in tickers
        ),
        ref_lines=(RefLine(value=0),),
        title="Drawdowns",
        yaxis=Axis(title="Drawdown", tick_format="percent0"),
        height_ratio=_DATA_PANEL_HEIGHTS[1],
    )

    single = len(tickers) == 1
    monthly_panel = Panel(
        bars=tuple(
            BarSeries(
                name=ticker,
                x=monthly[date_col],
                y=monthly[ticker],
                colors=tuple(_monthly_colors(monthly[ticker].to_list(), colors[ticker], light[ticker], single=single)),
                opacity=_MONTHLY_OPACITY,
                legend_group=ticker,
                show_legend=False,
                hover=HoverSpec(label=f"{ticker} Monthly Return", value_format="percent2", date_header=False),
            )
            for ticker in tickers
        ),
        title="Monthly Returns",
        yaxis=Axis(title="Monthly Return", tick_format="percent0"),
        height_ratio=_DATA_PANEL_HEIGHTS[2],
    )

    return FigureSpec(
        title=f"{' vs '.join(tickers)} Performance Dashboard",
        panels=(cumulative, drawdowns, monthly_panel),
        height=_DASHBOARD_HEIGHT_PX,
        arrangement="stacked",
        shared_x=True,
        vertical_spacing=_PANEL_GAP,
    )


def _drawdown_values(prices: pl.DataFrame, price_col: str) -> list[float]:
    """Decline from the running peak, as a fraction.

    Args:
        prices: Frame holding the cumulative price series.
        price_col: Column to measure.

    Returns:
        list[float]: One value per observation, at most zero.

    """
    series = prices[price_col]
    cummax = prices.select(pl.col(price_col).cum_max().alias("cummax"))["cummax"]
    return ((series - cummax) / cummax).to_list()


def _monthly_colors(values: list[float], color: str, light: str, *, single: bool) -> list[str]:
    """Colour each monthly bar by the sign of its return.

    Args:
        values: The monthly returns.
        color: The asset's base colour.
        light: A faded version of it, for negative months.
        single: Whether the dataset holds exactly one asset, which allows
            plain green and red.

    Returns:
        list[str]: One colour per value.

    """
    if single:
        return [_SINGLE_POSITIVE if value > 0 else _SINGLE_NEGATIVE for value in values]
    return [color if value > 0 else light for value in values]


def portfolio_snapshot_spec(portfolio: PortfolioLike, log_scale: bool) -> FigureSpec:
    """Describe the two-panel portfolio dashboard.

    Accumulated NAV — with its tilt and timing components, and the net-of-cost
    path when a cost model is active — over the drawdown it produced.

    Args:
        portfolio: The portfolio to plot.
        log_scale: Use a logarithmic scale for NAV.

    Returns:
        FigureSpec: Two stacked panels sharing an x-axis.

    """
    components = [
        ("NAV", portfolio.nav_accumulated),
        ("Tilt", portfolio.tilt.nav_accumulated),
        ("Timing", portfolio.timing.nav_accumulated),
    ]
    lines = [
        LineSeries(name=name, x=frame["date"], y=frame["NAV_accumulated"], show_legend=False)
        for name, frame in components
    ]

    if portfolio.cost_model.cost_per_unit > 0:
        net = portfolio.net_cost_nav
        lines.append(
            LineSeries(
                name="Net-of-Cost NAV",
                # The frame may not carry a date column; the renderer then
                # numbers the points, matching what this chart already did.
                x=net["date"] if "date" in net.columns else None,
                y=net["NAV_accumulated_net"],
                dash="dash",
                show_legend=True,
            )
        )

    nav = Panel(
        lines=tuple(lines),
        title="Accumulated Profit",
        yaxis=Axis(title="NAV (accumulated)", tick_format="si2", log=log_scale),
        height_ratio=_PORTFOLIO_PANEL_HEIGHTS[0],
    )

    drawdown = portfolio.drawdown
    drawdown_panel = Panel(
        lines=(
            LineSeries(
                name="Drawdown",
                x=drawdown["date"],
                y=drawdown["drawdown_pct"],
                fill=True,
                show_legend=False,
            ),
        ),
        ref_lines=(RefLine(value=0),),
        title="Drawdown",
        yaxis=Axis(title="Drawdown", tick_format="percent0"),
        height_ratio=_PORTFOLIO_PANEL_HEIGHTS[1],
    )

    return FigureSpec(
        title="Performance Dashboard",
        panels=(nav, drawdown_panel),
        height=_DASHBOARD_HEIGHT_PX,
        arrangement="stacked",
        shared_x=True,
        vertical_spacing=_PANEL_GAP,
    )


def _nav_comparison_spec(
    curves: list[tuple[str, pl.DataFrame]],
    title: str,
    log_scale: bool,
) -> FigureSpec:
    """Describe a set of NAV curves drawn on one pair of axes.

    Args:
        curves: ``(label, frame)`` pairs, each frame holding a NAV series.
        title: Chart title.
        log_scale: Use a logarithmic y-axis.

    Returns:
        FigureSpec: One line per curve.

    """
    panel = Panel(
        lines=tuple(
            LineSeries(name=label, x=frame["date"], y=frame["NAV_accumulated"], width=_NAV_WIDTH)
            for label, frame in curves
        ),
        yaxis=Axis(title="NAV (accumulated)", log=log_scale),
    )
    return FigureSpec(title=title, panels=(panel,))


def lagged_performance_spec(portfolio: PortfolioLike, lags: list[int] | None, log_scale: bool) -> FigureSpec:
    """Describe the NAV curves of several execution-delayed portfolios.

    Args:
        portfolio: The portfolio to plot.
        lags: Integer lags to apply, or None for 0 through 4.
        log_scale: Use a logarithmic y-axis.

    Returns:
        FigureSpec: One line per lag.

    Raises:
        TypeError: If *lags* is not a list of integers.

    """
    lags = _DEFAULT_LAGS if lags is None else lags
    if not isinstance(lags, list) or not all(isinstance(x, int) for x in lags):
        raise TypeError

    curves = [(f"lag {lag}", (portfolio if lag == 0 else portfolio.lag(lag)).nav_accumulated) for lag in lags]
    return _nav_comparison_spec(curves, "NAV accumulated by lag", log_scale)


def smoothed_holdings_performance_spec(
    portfolio: PortfolioLike,
    windows: list[int] | None,
    log_scale: bool,
) -> FigureSpec:
    """Describe the NAV curves of several smoothed-holding portfolios.

    Args:
        portfolio: The portfolio to plot.
        windows: Smoothing step counts, or None for 0 through 4.
        log_scale: Use a logarithmic y-axis.

    Returns:
        FigureSpec: One line per smoothing level.

    Raises:
        TypeError: If *windows* is not a list of non-negative integers.

    """
    windows = _DEFAULT_LAGS if windows is None else windows
    if not isinstance(windows, list) or not all(isinstance(x, int) and x >= 0 for x in windows):
        raise TypeError

    curves = [
        (f"smooth {n}", (portfolio if n == 0 else portfolio.smoothed_holding(n)).nav_accumulated) for n in windows
    ]
    return _nav_comparison_spec(curves, "NAV accumulated by smoothed holdings", log_scale)
