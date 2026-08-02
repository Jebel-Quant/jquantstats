"""Diagnostic charts: lead/lag IR, correlation, monthly calendar, cost impact.

Split out of the former single-module `_plots/_portfolio.py`; composed into
:class:`PortfolioPlots` by `_core.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

if TYPE_CHECKING:
    from .._protocol import PortfolioLike


class _DiagnosticPlotsMixin:
    """Diagnostic charts for :class:`PortfolioPlots`."""

    __slots__ = ()

    _portfolio: PortfolioLike

    def lead_lag_ir_plot(self, start: int = -10, end: int = 19) -> go.Figure:
        """Plot Sharpe ratio (IR) across lead/lag variants of the portfolio.

        Builds portfolios with cash positions lagged from ``start`` to ``end``
        (inclusive) and plots a bar chart of the Sharpe ratio for each lag.
        Positive lags delay weights; negative lags lead them.

        Args:
            start: First lag to include (default: -10).
            end: Last lag to include (default: +19).

        Returns:
            A Plotly Figure with one bar per lag labeled by the lag value.
        """
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError
        if start > end:
            start, end = end, start

        lags = list(range(start, end + 1))

        x_vals: list[int] = []
        y_vals: list[float] = []

        for n in lags:
            pf = self._portfolio if n == 0 else self._portfolio.lag(n)
            # Compute Sharpe on the portfolio's returns series
            sharpe_val = pf.stats.sharpe().get("returns", float("nan"))
            # Ensure a float (Stats returns mapping asset->value)
            y_vals.append(float(sharpe_val) if sharpe_val is not None else float("nan"))
            x_vals.append(n)

        colors = ["red" if x == 0 else "#1f77b4" for x in x_vals]
        fig = go.Figure(
            data=[
                go.Bar(x=x_vals, y=y_vals, name="Sharpe by lag", marker_color=colors),
            ]
        )
        fig.update_layout(
            title="Lead/Lag Information Ratio (Sharpe) by Lag",
            xaxis_title="Lag (steps)",
            yaxis_title="Sharpe ratio",
            plot_bgcolor="white",
            hovermode="x",
        )
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
        return fig

    def correlation_heatmap(
        self,
        frame: pl.DataFrame | None = None,
        name: str = "portfolio",
        title: str = "Correlation heatmap",
    ) -> go.Figure:
        """Plot a correlation heatmap for assets and the portfolio.

        If ``frame`` is None, uses the portfolio's prices. The portfolio's
        profit series is appended under ``name`` before computing the
        correlation matrix.

        Args:
            frame: Optional Polars DataFrame with at least the asset price
                columns. If omitted, uses ``self._portfolio.prices``.
            name: Column name under which to include the portfolio profit.
            title: Plot title.

        Returns:
            A Plotly Figure rendering the correlation matrix as a heatmap.
        """
        if frame is None:
            frame = self._portfolio.prices

        corr = self._portfolio.correlation(frame, name=name)

        # Create an interactive heatmap
        fig = px.imshow(
            corr,
            x=corr.columns,
            y=corr.columns,
            text_auto=".2f",  # show correlation values
            color_continuous_scale="RdBu_r",  # red-blue diverging colormap
            zmin=-1,
            zmax=1,  # correlation range
            title=title,
        )

        # Adjust layout
        fig.update_layout(
            xaxis_title="", yaxis_title="", width=700, height=600, coloraxis_colorbar={"title": "Correlation"}
        )

        return fig

    def monthly_returns_heatmap(self) -> go.Figure:
        """Plot a monthly returns calendar heatmap.

        Groups portfolio returns by calendar year and month, then renders a
        Plotly heatmap with months on the x-axis and years on the y-axis.
        Green cells indicate positive months; red cells indicate negative
        months.  Cell text shows the percentage return for that month.

        Returns:
            A Plotly Figure with a calendar heatmap of monthly returns.

        Raises:
            ValueError: If the portfolio has no ``date`` column.
        """
        monthly = self._portfolio.monthly

        years = monthly["year"].unique().sort().to_list()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        z: list[list[float | None]] = []
        text: list[list[str]] = []
        for year in years:
            year_data = monthly.filter(pl.col("year") == year)
            year_row: list[float | None] = []
            year_text: list[str] = []
            for m in range(1, 13):
                month_data = year_data.filter(pl.col("month") == m)
                if month_data.is_empty():
                    year_row.append(None)
                    year_text.append("")
                else:
                    ret = float(month_data["returns"][0])
                    year_row.append(ret * 100.0)
                    year_text.append(f"{ret * 100.0:.1f}%")
            z.append(year_row)
            text.append(year_text)

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=month_names,
                y=[str(y) for y in years],
                text=text,
                texttemplate="%{text}",
                colorscale="RdYlGn",
                zmid=0,
                colorbar={"title": "Return (%)"},
                hovertemplate="<b>%{y} %{x}</b><br>Return: %{text}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            plot_bgcolor="white",
            yaxis={"type": "category"},
        )

        return fig

    def trading_cost_impact_plot(self, max_bps: int = 20) -> go.Figure:
        """Plot the Sharpe ratio as a function of one-way trading costs.

        Evaluates the portfolio's annualised Sharpe ratio at each integer
        cost level from 0 up to ``max_bps`` basis points and renders the
        result as a line chart.  The zero-cost Sharpe is shown as a
        reference horizontal line so that the reader can quickly gauge
        at what cost level the strategy's edge is eroded.

        Args:
            max_bps: Maximum one-way trading cost to evaluate, in basis
                points.  Defaults to 20.

        Returns:
            A Plotly Figure with one line trace showing Sharpe vs. cost.

        Raises:
            ValueError: If ``max_bps`` is not a positive integer.
        """
        impact = self._portfolio.trading_cost_impact(max_bps=max_bps)

        cost_vals = impact["cost_bps"].to_list()
        sharpe_vals = impact["sharpe"].to_list()

        # Baseline Sharpe at zero cost
        baseline = float(sharpe_vals[0]) if sharpe_vals and sharpe_vals[0] is not None else float("nan")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cost_vals,
                y=sharpe_vals,
                mode="lines+markers",
                name="Sharpe (cost-adjusted)",
                marker={"size": 6},
                line={"width": 2, "color": "#1f77b4"},
            )
        )
        if baseline == baseline:  # only add when baseline is finite (NaN != NaN)
            fig.add_hline(
                y=baseline,
                line_width=1,
                line_dash="dash",
                line_color="gray",
                annotation_text="0 bps baseline",
                annotation_position="top right",
            )

        fig.update_layout(
            title=f"Trading Cost Impact on Sharpe Ratio (0\u2013{max_bps} bps)",
            hovermode="x unified",
            plot_bgcolor="white",
        )
        fig.update_xaxes(
            title_text="One-way cost (basis points)",
            showgrid=True,
            gridwidth=0.5,
            gridcolor="lightgrey",
            dtick=1,
        )
        fig.update_yaxes(
            title_text="Annualised Sharpe ratio",
            showgrid=True,
            gridwidth=0.5,
            gridcolor="lightgrey",
        )
        return fig
