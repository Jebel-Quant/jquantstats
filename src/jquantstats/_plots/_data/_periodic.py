"""Periodic-return bar charts and the monthly-return heatmap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl

from ._styling import _apply_base_layout, _bar_colors, _hex_to_rgba, _ticker_colors

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike


def _monthly_heatmap_matrix(
    monthly: pl.DataFrame, years: list[int]
) -> tuple[list[list[float | None]], list[list[str]]]:
    """Build the year-by-month value and label grids for the monthly heatmap.

    Args:
        monthly: Aggregated frame with ``_year``, ``_month`` and ``ret`` columns.
        years: Sorted unique years, defining the row order of the output grids.

    Returns:
        A ``(z, text)`` tuple: ``z`` holds returns scaled to percent (``None``
        for missing cells) and ``text`` the formatted per-cell labels.

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
    return z, text


class _PeriodicPlotsMixin:
    """Daily/monthly/yearly bar charts and the monthly heatmap for :class:`DataPlots`."""

    __slots__ = ()

    _data: DataLike

    def daily_returns(self, title: str = "Daily Returns") -> go.Figure:
        """Daily returns as a bar chart.

        Each bar is coloured green for positive returns and red for negative
        returns.  When multiple assets are present each asset gets its own
        trace in the palette colour with opacity used for positive/negative
        differentiation.

        Args:
            title: Chart title. Defaults to ``"Daily Returns"``.

        Returns:
            go.Figure: Interactive Plotly bar chart.

        """
        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)
        single = len(tickers) == 1

        fig = go.Figure()
        for ticker in tickers:
            values = df[ticker].to_list()
            bar_colors = _bar_colors(values, colors[ticker], single_asset=single)

            fig.add_trace(
                go.Bar(
                    x=df[date_col],
                    y=df[ticker],
                    name=ticker,
                    marker={"color": bar_colors, "line": {"width": 0}},
                    opacity=0.85,
                    hovertemplate=f"{ticker}: %{{y:.2%}}",
                )
            )

        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text="Return", tickformat=".1%")
        return fig

    def yearly_returns(self, title: str = "Yearly Returns", compounded: bool = True) -> go.Figure:
        """Annual compounded (or summed) returns as a grouped bar chart.

        Args:
            title: Chart title. Defaults to ``"Yearly Returns"``.
            compounded: Compound returns within each year. Defaults to True.

        Returns:
            go.Figure: Interactive Plotly grouped bar chart.

        """
        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        agg_exprs = (
            [((1.0 + pl.col(t)).product() - 1.0).alias(t) for t in tickers]
            if compounded
            else [pl.col(t).sum().alias(t) for t in tickers]
        )
        yearly = (
            df.with_columns(pl.col(date_col).dt.year().alias("_year")).group_by("_year").agg(agg_exprs).sort("_year")
        )

        fig = go.Figure()
        for ticker in tickers:
            values = yearly[ticker].to_list()
            bar_colors = [
                colors[ticker] if v is not None and v >= 0 else _hex_to_rgba(colors[ticker], 0.5) for v in values
            ]
            fig.add_trace(
                go.Bar(
                    x=yearly["_year"],
                    y=yearly[ticker],
                    name=ticker,
                    marker={"color": bar_colors, "line": {"width": 0}},
                    opacity=0.85,
                    hovertemplate=f"{ticker}: %{{y:.2%}}",
                )
            )

        _apply_base_layout(fig, title, with_range_selector=False)
        fig.update_layout(barmode="group", xaxis_title="Year")
        fig.update_yaxes(title_text="Annual Return", tickformat=".1%")
        return fig

    def monthly_returns(self, title: str = "Monthly Returns", compounded: bool = True) -> go.Figure:
        """Monthly compounded (or summed) returns as a bar chart.

        Args:
            title: Chart title. Defaults to ``"Monthly Returns"``.
            compounded: Compound returns within each month. Defaults to True.

        Returns:
            go.Figure: Interactive Plotly bar chart.

        """
        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)
        single = len(tickers) == 1

        monthly = df.group_by_dynamic(
            index_column=date_col, every="1mo", period="1mo", closed="right", label="right"
        ).agg(
            [((1.0 + pl.col(t)).product() - 1.0).alias(t) if compounded else pl.col(t).sum().alias(t) for t in tickers]
        )

        fig = go.Figure()
        for ticker in tickers:
            values = monthly[ticker].to_list()
            bar_colors = _bar_colors(values, colors[ticker], single_asset=single)

            fig.add_trace(
                go.Bar(
                    x=monthly[date_col],
                    y=monthly[ticker],
                    name=ticker,
                    marker={"color": bar_colors, "line": {"width": 0}},
                    opacity=0.85,
                    hovertemplate=f"{ticker}: %{{y:.2%}}",
                )
            )

        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text="Monthly Return", tickformat=".1%")
        return fig

    def monthly_heatmap(
        self,
        title: str = "Monthly Returns Heatmap",
        compounded: bool = True,
        asset: str | None = None,
    ) -> go.Figure:
        """Monthly returns calendar heatmap (year x month).

        One heatmap is produced per call for a single asset.  Green cells
        indicate positive months; red cells indicate negative months.

        Args:
            title: Chart title. Defaults to ``"Monthly Returns Heatmap"``.
            compounded: Compound intra-month returns. Defaults to True.
            asset: Asset column name to display.  Defaults to the first
                non-date column in the dataset.

        Returns:
            go.Figure: Interactive Plotly heatmap.

        """
        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        col = asset if asset in tickers else tickers[0]

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        agg_expr = ((1.0 + pl.col(col)).product() - 1.0).alias("ret") if compounded else pl.col(col).sum().alias("ret")
        monthly = (
            df.with_columns(
                [
                    pl.col(date_col).dt.year().alias("_year"),
                    pl.col(date_col).dt.month().alias("_month"),
                ]
            )
            .group_by(["_year", "_month"])
            .agg(agg_expr.alias("ret"))
            .sort(["_year", "_month"])
        )

        years = sorted(monthly["_year"].unique().to_list())
        z, text = _monthly_heatmap_matrix(monthly, years)

        fig = go.Figure(
            go.Heatmap(
                x=month_names,
                y=[str(y) for y in years],
                z=z,
                text=text,
                texttemplate="%{text}",
                colorscale=[[0, "#d62728"], [0.5, "#ffffff"], [1, "#2ca02c"]],
                zmid=0,
                showscale=True,
                colorbar={"title": "Return (%)"},
                hovertemplate="<b>%{y} %{x}</b><br>Return: %{text}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"{title} — {col}",
            height=max(300, 40 * len(years) + 100),
            plot_bgcolor="white",
            xaxis={"side": "top"},
        )
        return fig
