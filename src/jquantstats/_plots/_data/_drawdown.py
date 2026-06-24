"""Drawdown charts (underwater curve and worst-period shading)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from ._styling import _apply_base_layout, _compute_drawdown_periods, _hex_to_rgba, _ticker_colors

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike


class _DrawdownPlotsMixin:
    """Drawdown plots for :class:`DataPlots`."""

    __slots__ = ()

    _data: DataLike

    def drawdown(self, title: str = "Drawdowns") -> go.Figure:
        """Underwater equity curve (drawdown) chart.

        Shows the percentage decline from the running peak for every column
        in the dataset (assets and benchmark where present).

        Args:
            title: Chart title. Defaults to ``"Drawdowns"``.

        Returns:
            go.Figure: Interactive Plotly filled-area chart.

        """
        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        prices = df.with_columns([(1.0 + pl.col(t)).cum_prod().alias(t) for t in tickers])

        fig = go.Figure()
        for ticker in tickers:
            price_s = prices[ticker]
            hwm = price_s.cum_max()
            dd = ((price_s - hwm) / hwm).to_list()

            fig.add_trace(
                go.Scatter(
                    x=prices[date_col],
                    y=dd,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor=_hex_to_rgba(colors[ticker], 0.3),
                    line={"color": colors[ticker], "width": 1.5},
                    name=ticker,
                    hovertemplate=f"{ticker}: %{{y:.2%}}",
                )
            )

        fig.add_hline(y=0, line_width=1, line_color="gray")
        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text="Drawdown", tickformat=".0%")
        return fig

    def drawdowns_periods(
        self,
        n: int = 5,
        title: str = "Top Drawdown Periods",
        asset: str | None = None,
    ) -> go.Figure:
        """Cumulative returns chart with the worst *n* drawdown periods shaded.

        Identifies the *n* deepest drawdown periods and overlays coloured
        rectangular shading on the cumulative returns line.  One asset is
        shown per call.

        Args:
            n: Number of worst drawdown periods to highlight. Defaults to 5.
            title: Chart title. Defaults to ``"Top Drawdown Periods"``.
            asset: Asset column name.  Defaults to the first non-date column.

        Returns:
            go.Figure: Interactive Plotly figure.

        """
        df = self._data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        col = asset if asset in tickers else tickers[0]

        price_series = (1.0 + df[col].cast(pl.Float64)).cum_prod()
        price_list = price_series.to_list()
        dates = df[date_col].to_list()

        drawdown_periods = _compute_drawdown_periods(price_list, n)

        dd_colors = px.colors.qualitative.Plotly

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=price_list,
                mode="lines",
                name=col,
                line={"color": "#1f77b4", "width": 2},
                hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{col}: %{{y:.2f}}x",
            )
        )

        for i, period in enumerate(drawdown_periods):
            start_date = dates[period["start_idx"]]
            end_date = dates[min(period["end_idx"] + 1, len(dates) - 1)]
            max_dd = period["max_drawdown"]
            shade_color = _hex_to_rgba(dd_colors[i % len(dd_colors)], alpha=0.2)

            fig.add_vrect(
                x0=start_date,
                x1=end_date,
                fillcolor=shade_color,
                line_width=0,
                annotation_text=f"#{i + 1} {max_dd:.1%}",
                annotation_position="top left",
                annotation_font_size=10,
            )

        _apply_base_layout(fig, f"{title} — {col}")
        fig.update_yaxes(title_text="Cumulative Return", tickformat=".2f")
        return fig
