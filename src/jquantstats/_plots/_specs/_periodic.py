"""Spec builders for the periodic bar charts and the monthly calendar.

The bar charts colour each bar by the sign of its value rather than by series,
which is why `~jquantstats._plots._spec.BarSeries` carries a colour per bar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from .._spec import Axis, BarSeries, FigureSpec, HeatmapGrid, HoverSpec, Panel
from .._style import bar_colors, ticker_colors, yearly_bar_colors

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike

__all__ = [
    "daily_returns_spec",
    "monthly_heatmap_spec",
    "monthly_returns_spec",
    "yearly_returns_spec",
]

# Bars are drawn slightly translucent so overlapping series stay readable.
_BAR_OPACITY = 0.85

_MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# A calendar row per year, plus room for the title and colour bar.
_HEATMAP_ROW_PX = 40
_HEATMAP_CHROME_PX = 100
_HEATMAP_MIN_PX = 300


def _split_columns(frame: pl.DataFrame) -> tuple[str, list[str]]:
    """Separate the date column from the value columns.

    Args:
        frame: A frame whose first column is the date axis.

    Returns:
        tuple[str, list[str]]: The date column name and every other column.

    """
    date_col = frame.columns[0]
    return date_col, [c for c in frame.columns if c != date_col]


def period_agg_exprs(tickers: list[str], compounded: bool) -> list[pl.Expr]:
    """Per-ticker aggregation expressions for a period bucket.

    Args:
        tickers: Asset column names to aggregate.
        compounded: Compound returns within the bucket when True, sum them
            when False.

    Returns:
        list[pl.Expr]: One aliased expression per ticker.

    """
    if compounded:
        return [((1.0 + pl.col(t)).product() - 1.0).alias(t) for t in tickers]
    return [pl.col(t).sum().alias(t) for t in tickers]


def _sign_coloured_bars(
    frame: pl.DataFrame,
    x_col: str,
    tickers: list[str],
    colors: dict[str, str],
    *,
    single: bool,
) -> tuple[BarSeries, ...]:
    """Build one bar series per ticker, each bar coloured by its sign.

    Args:
        frame: The prepared frame holding the plotted values.
        x_col: Column giving the bar positions.
        tickers: Columns to draw, in order.
        colors: Ticker to hex colour mapping.
        single: Whether the dataset holds exactly one asset, which selects
            the plain green/red palette over the faded per-asset one.

    Returns:
        tuple[BarSeries, ...]: One series per ticker, in the given order.

    """
    return tuple(
        BarSeries(
            name=ticker,
            x=frame[x_col],
            y=frame[ticker],
            colors=tuple(bar_colors(frame[ticker].to_list(), colors[ticker], single_asset=single)),
            opacity=_BAR_OPACITY,
            hover=HoverSpec(label=ticker, value_format="percent2", date_header=False),
        )
        for ticker in tickers
    )


def daily_returns_spec(data: DataLike, title: str) -> FigureSpec:
    """Describe the daily-returns bar chart.

    Args:
        data: The dataset to plot.
        title: Chart title.

    Returns:
        FigureSpec: One bar series per asset, coloured by sign.

    """
    df = data.all
    date_col, tickers = _split_columns(df)

    panel = Panel(
        bars=_sign_coloured_bars(df, date_col, tickers, ticker_colors(tickers), single=len(tickers) == 1),
        yaxis=Axis(title="Return", tick_format="percent1"),
    )
    return FigureSpec(title=title, panels=(panel,))


def yearly_returns_spec(data: DataLike, title: str, compounded: bool) -> FigureSpec:
    """Describe the annual-returns grouped bar chart.

    Args:
        data: The dataset to plot.
        title: Chart title.
        compounded: Compound returns within each year.

    Returns:
        FigureSpec: One grouped bar series per asset, over a year axis.

    """
    df = data.all
    date_col, tickers = _split_columns(df)
    colors = ticker_colors(tickers)

    yearly = (
        df.with_columns(pl.col(date_col).dt.year().alias("_year"))
        .group_by("_year")
        .agg(period_agg_exprs(tickers, compounded))
        .sort("_year")
    )

    panel = Panel(
        bars=tuple(
            BarSeries(
                name=ticker,
                x=yearly["_year"],
                y=yearly[ticker],
                # A flat zero year counts as positive here, unlike the other
                # bar charts — see `yearly_bar_colors`.
                colors=tuple(yearly_bar_colors(yearly[ticker].to_list(), colors[ticker])),
                opacity=_BAR_OPACITY,
                hover=HoverSpec(label=ticker, value_format="percent2", date_header=False),
            )
            for ticker in tickers
        ),
        xaxis=Axis(title="Year"),
        yaxis=Axis(title="Annual Return", tick_format="percent1"),
    )
    # No range selector: the axis is calendar years, not a continuous date line.
    return FigureSpec(title=title, panels=(panel,), date_range_selector=False, bar_mode="group")


def monthly_returns_spec(data: DataLike, title: str, compounded: bool) -> FigureSpec:
    """Describe the monthly-returns bar chart.

    Args:
        data: The dataset to plot.
        title: Chart title.
        compounded: Compound returns within each month.

    Returns:
        FigureSpec: One bar series per asset, coloured by sign.

    """
    df = data.all
    date_col, tickers = _split_columns(df)

    monthly = df.group_by_dynamic(index_column=date_col, every="1mo", period="1mo", closed="right", label="right").agg(
        period_agg_exprs(tickers, compounded)
    )

    panel = Panel(
        bars=_sign_coloured_bars(monthly, date_col, tickers, ticker_colors(tickers), single=len(tickers) == 1),
        yaxis=Axis(title="Monthly Return", tick_format="percent1"),
    )
    return FigureSpec(title=title, panels=(panel,))


def _heatmap_grids(
    monthly: pl.DataFrame, years: list[int]
) -> tuple[tuple[tuple[float | None, ...], ...], tuple[tuple[str, ...], ...]]:
    """Build the year-by-month value and label grids.

    Args:
        monthly: Aggregated frame with ``_year``, ``_month`` and ``ret`` columns.
        years: Sorted unique years, defining the row order of the output grids.

    Returns:
        A ``(z, text)`` pair: ``z`` holds returns scaled to percent, with None
        for months the data does not cover, and ``text`` the formatted labels.

    """
    year_idx = {y: i for i, y in enumerate(years)}
    z: list[list[float | None]] = [[None] * 12 for _ in years]
    text: list[list[str]] = [[""] * 12 for _ in years]
    for row in monthly.iter_rows(named=True):
        yi = year_idx[row["_year"]]
        mi = row["_month"] - 1
        val = row["ret"]
        z[yi][mi] = val * 100 if val is not None else None
        text[yi][mi] = f"{val:.1%}" if val is not None else ""
    return tuple(tuple(r) for r in z), tuple(tuple(r) for r in text)


def monthly_heatmap_spec(data: DataLike, title: str, compounded: bool, asset: str | None) -> FigureSpec:
    """Describe the monthly-returns calendar heatmap.

    One asset per chart: the grid is already two-dimensional, so a second
    asset would need a second grid.

    Args:
        data: The dataset to plot.
        title: Chart title, which gains the asset name.
        compounded: Compound returns within each month.
        asset: Asset column to display, or None for the first.

    Returns:
        FigureSpec: A single year-by-month grid.

    """
    df = data.all
    date_col, tickers = _split_columns(df)
    col = asset if asset in tickers else tickers[0]

    agg = ((1.0 + pl.col(col)).product() - 1.0) if compounded else pl.col(col).sum()
    monthly = (
        df.with_columns(
            pl.col(date_col).dt.year().alias("_year"),
            pl.col(date_col).dt.month().alias("_month"),
        )
        .group_by(["_year", "_month"])
        .agg(agg.alias("ret"))
        .sort(["_year", "_month"])
    )

    years = sorted(monthly["_year"].unique().to_list())
    z, text = _heatmap_grids(monthly, years)

    panel = Panel(
        heatmap=HeatmapGrid(
            x_labels=_MONTH_NAMES,
            y_labels=tuple(str(y) for y in years),
            z=z,
            text=text,
            colorscale="red_white_green",
            colorbar_title="Return (%)",
            hover_label="Return",
        ),
        # Months read as column headings, so they belong along the top.
        xaxis=Axis(opposite_side=True),
    )
    return FigureSpec(
        title=f"{title} — {col}",
        panels=(panel,),
        height=max(_HEATMAP_MIN_PX, _HEATMAP_ROW_PX * len(years) + _HEATMAP_CHROME_PX),
        chrome="bare",
    )
