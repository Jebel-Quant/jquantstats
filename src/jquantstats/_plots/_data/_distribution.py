"""Return-distribution charts (histogram and per-period box plots)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ._styling import _apply_base_layout, _ticker_colors

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike


class _DistributionPlotsMixin:
    """Return-distribution plots for :class:`DataPlots`."""

    __slots__ = ()

    _data: DataLike

    def histogram(self, title: str = "Returns Distribution", bins: int = 50) -> go.Figure:
        """Return histogram with a kernel density overlay.

        Each asset is shown as a semi-transparent histogram overlaid on the
        same axes so distributions can be compared visually.

        Args:
            title: Chart title. Defaults to ``"Returns Distribution"``.
            bins: Number of histogram bins. Defaults to 50.

        Returns:
            go.Figure: Interactive Plotly histogram figure.

        """
        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        fig = go.Figure()
        for ticker in tickers:
            values = df[ticker].drop_nulls().to_list()
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=ticker,
                    nbinsx=bins,
                    marker_color=colors[ticker],
                    opacity=0.6,
                    hovertemplate=f"{ticker}: %{{x:.2%}}<extra></extra>",
                )
            )

        _apply_base_layout(fig, title, with_range_selector=False)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="Return", tickformat=".1%")
        fig.update_yaxes(title_text="Count")
        return fig

    def distribution(
        self,
        title: str = "Return Distribution by Period",
        compounded: bool = True,
    ) -> go.Figure:
        """Return distributions across daily, weekly, monthly, quarterly and yearly periods.

        Renders a box plot for each aggregation period so the user can compare
        how the distribution widens as the holding period lengthens.  One
        subplot column is produced per asset.

        Args:
            title: Chart title. Defaults to ``"Return Distribution by Period"``.
            compounded: Compound returns within each period. Defaults to True.

        Returns:
            go.Figure: Interactive Plotly figure with one subplot per asset.

        """
        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        periods = [
            ("Daily", None),
            ("Weekly", "1w"),
            ("Monthly", "1mo"),
            ("Quarterly", "3mo"),
            ("Yearly", "1y"),
        ]

        n_assets = len(tickers)
        fig = make_subplots(
            rows=1,
            cols=n_assets,
            subplot_titles=tickers,
            shared_yaxes=True,
        )

        for col_idx, ticker in enumerate(tickers, start=1):
            for period_name, trunc in periods:
                if trunc is None:
                    values = df[ticker].drop_nulls().to_list()
                else:
                    agg_expr = (
                        ((1.0 + pl.col(ticker)).product() - 1.0).alias("ret")
                        if compounded
                        else pl.col(ticker).sum().alias("ret")
                    )
                    agg_df = (
                        df.with_columns(pl.col(date_col).dt.truncate(trunc).alias("_period"))
                        .group_by("_period")
                        .agg(agg_expr)
                    )
                    values = agg_df["ret"].drop_nulls().to_list()

                fig.add_trace(
                    go.Box(
                        y=values,
                        name=period_name,
                        marker_color=colors[ticker],
                        showlegend=(col_idx == 1),
                        legendgroup=period_name,
                        boxpoints="outliers",
                        hovertemplate=f"{period_name}: %{{y:.2%}}<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )

        fig.update_layout(
            title=title,
            height=500,
            plot_bgcolor="white",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "xanchor": "right", "x": 1},
        )
        fig.update_yaxes(tickformat=".1%", showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
        fig.update_xaxes(showgrid=False)
        return fig
