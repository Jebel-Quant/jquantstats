"""Spec builders for the return-distribution charts.

`histogram_spec` overlays the whole return distribution of each series on one
pair of axes. `distribution_spec` instead asks how the distribution *widens* as
the holding period lengthens, which needs a panel per asset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from .._spec import Axis, BoxSeries, FigureSpec, HistogramSeries, HoverSpec, Panel
from .._style import ticker_colors

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike

__all__ = ["distribution_spec", "histogram_spec"]

# Overlaid distributions have to be seen through one another.
_OVERLAY_OPACITY = 0.6

# Holding periods, shortest first, so the widening reads left to right.
# None means "no aggregation": the raw per-observation returns.
_PERIODS: tuple[tuple[str, str | None], ...] = (
    ("Daily", None),
    ("Weekly", "1w"),
    ("Monthly", "1mo"),
    ("Quarterly", "3mo"),
    ("Yearly", "1y"),
)

_PANEL_HEIGHT_PX = 500


def _split_columns(frame: pl.DataFrame) -> tuple[str, list[str]]:
    """Separate the date column from the value columns.

    Args:
        frame: A frame whose first column is the date axis.

    Returns:
        tuple[str, list[str]]: The date column name and every other column.

    """
    date_col = frame.columns[0]
    return date_col, [c for c in frame.columns if c != date_col]


def histogram_spec(data: DataLike, title: str, bins: int) -> FigureSpec:
    """Describe the overlaid return-distribution histogram.

    Args:
        data: The dataset to plot.
        title: Chart title.
        bins: Number of histogram bins.

    Returns:
        FigureSpec: One translucent histogram per column, on shared axes.

    """
    df = data.all
    _, tickers = _split_columns(df)
    colors = ticker_colors(tickers)

    panel = Panel(
        histograms=tuple(
            HistogramSeries(
                name=ticker,
                values=df[ticker].drop_nulls().to_list(),
                color=colors[ticker],
                bins=bins,
                opacity=_OVERLAY_OPACITY,
                hover=HoverSpec(
                    label=ticker,
                    value_format="percent2",
                    date_header=False,
                    axis="x",
                    hide_extra=True,
                ),
            )
            for ticker in tickers
        ),
        xaxis=Axis(title="Return", tick_format="percent1"),
        yaxis=Axis(title="Count"),
    )
    # No range selector: the x-axis is return magnitude, not time.
    return FigureSpec(title=title, panels=(panel,), date_range_selector=False, bar_mode="overlay")


def _period_values(df: pl.DataFrame, date_col: str, ticker: str, trunc: str | None, compounded: bool) -> list[float]:
    """Aggregate one ticker's returns into buckets of a given length.

    Args:
        df: The combined index/returns frame.
        date_col: Name of the date column.
        ticker: Column to aggregate.
        trunc: Polars duration to bucket by, or None for no aggregation.
        compounded: Compound returns within each bucket.

    Returns:
        list[float]: One value per bucket, nulls dropped.

    """
    if trunc is None:
        return df[ticker].drop_nulls().to_list()

    agg = ((1.0 + pl.col(ticker)).product() - 1.0) if compounded else pl.col(ticker).sum()
    bucketed = (
        df.with_columns(pl.col(date_col).dt.truncate(trunc).alias("_period")).group_by("_period").agg(agg.alias("ret"))
    )
    return bucketed["ret"].drop_nulls().to_list()


def distribution_spec(data: DataLike, title: str, compounded: bool) -> FigureSpec:
    """Describe the by-holding-period distribution chart.

    One panel per asset, each holding a box per period, so the widening of the
    distribution with holding length can be compared across assets.

    Args:
        data: The dataset to plot.
        title: Chart title.
        compounded: Compound returns within each period.

    Returns:
        FigureSpec: One side-by-side panel per asset, sharing a vertical scale.

    """
    df = data.all
    date_col, tickers = _split_columns(df)
    colors = ticker_colors(tickers)

    panels = tuple(
        Panel(
            boxes=tuple(
                BoxSeries(
                    name=period_name,
                    values=_period_values(df, date_col, ticker, trunc, compounded),
                    color=colors[ticker],
                    # The same periods repeat in every panel, so only the
                    # first names them; the groups tie them together.
                    show_legend=(index == 0),
                    legend_group=period_name,
                    hover=HoverSpec(
                        label=period_name,
                        value_format="percent2",
                        date_header=False,
                        hide_extra=True,
                    ),
                )
                for period_name, trunc in _PERIODS
            ),
            title=ticker,
            yaxis=Axis(tick_format="percent1"),
        )
        for index, ticker in enumerate(tickers)
    )

    return FigureSpec(
        title=title,
        panels=panels,
        height=_PANEL_HEIGHT_PX,
        chrome="panels",
        arrangement="side_by_side",
        shared_y=True,
    )
