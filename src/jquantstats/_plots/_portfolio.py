"""Plotting utilities for portfolio analytics using Plotly.

This module defines the PortfolioPlots facade which renders common portfolio visuals
such as snapshots, lagged performance curves, smoothed-holdings curves, and
lead/lag information ratio bar charts. Designed for notebook use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from plotly.subplots import make_subplots

from ._data import _apply_base_layout

if TYPE_CHECKING:
    from ._protocol import PortfolioLike

# Ensure Plotly works with Marimo (set after imports to satisfy linters)
pio.renderers.default = "plotly_mimetype"


class PortfolioPlots:
    """Facade for portfolio plots built with Plotly.

    Provides convenience methods to visualize portfolio performance and
    diagnostics directly from a Portfolio instance (e.g., snapshot charts,
    lagged performance, smoothed holdings, and lead/lag IR).
    """

    __slots__ = ("_portfolio",)

    def __init__(self, portfolio: PortfolioLike) -> None:
        self._portfolio = portfolio

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

    def snapshot(self, log_scale: bool = False) -> go.Figure:
        """Return a snapshot dashboard of NAV and drawdown.

        When the portfolio has a non-zero ``cost_model.cost_per_unit``, an additional
        ``"Net-of-Cost NAV"`` trace is overlaid on the NAV panel showing the
        realised NAV path after deducting position-delta trading costs.

        Args:
            log_scale (bool, optional): If True, display NAV on a log scale. Defaults to False.

        Returns:
            plotly.graph_objects.Figure: A Figure with accumulated NAV (including tilt/timing)
                and drawdown shaded area, equipped with a range selector.
        """
        # Create subplot grid with domain for stats table
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.66, 0.33],
            subplot_titles=["Accumulated Profit", "Drawdown"],
            vertical_spacing=0.05,
        )

        # --- Row 1: Cumulative Returns
        fig.add_trace(
            go.Scatter(
                x=self._portfolio.nav_accumulated["date"],
                y=self._portfolio.nav_accumulated["NAV_accumulated"],
                mode="lines",
                name="NAV",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self._portfolio.tilt.nav_accumulated["date"],
                y=self._portfolio.tilt.nav_accumulated["NAV_accumulated"],
                mode="lines",
                name="Tilt",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self._portfolio.timing.nav_accumulated["date"],
                y=self._portfolio.timing.nav_accumulated["NAV_accumulated"],
                mode="lines",
                name="Timing",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Net-of-cost NAV overlay (only when a cost model is active)
        if self._portfolio.cost_model.cost_per_unit > 0:
            net_nav_df = self._portfolio.net_cost_nav
            x_dates = net_nav_df["date"] if "date" in net_nav_df.columns else None
            fig.add_trace(
                go.Scatter(
                    x=x_dates,
                    y=net_nav_df["NAV_accumulated_net"],
                    mode="lines",
                    name="Net-of-Cost NAV",
                    line={"dash": "dash"},
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=self._portfolio.drawdown["date"],
                y=self._portfolio.drawdown["drawdown_pct"],
                mode="lines",
                fill="tozeroy",
                name="Drawdown",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_hline(y=0, line_width=1, line_color="gray", row=2, col=1)

        _apply_base_layout(fig, "Performance Dashboard", height=1200)

        fig.update_yaxes(title_text="NAV (accumulated)", row=1, col=1, tickformat=".2s")
        fig.update_yaxes(title_text="Drawdown", row=2, col=1, tickformat=".0%")

        if log_scale:
            fig.update_yaxes(type="log", row=1, col=1)
            # Ensure the first y-axis is explicitly set for environments
            # where subplot updates may not propagate to layout alias.
            if hasattr(fig.layout, "yaxis"):  # pragma: no branch — plotly figures always have .yaxis
                fig.layout.yaxis.type = "log"

        return fig

    @staticmethod
    def _apply_nav_layout(fig: go.Figure, title: str, log_scale: bool = False) -> None:
        """Apply common NAV-accumulated layout to *fig* in-place.

        Configures the plot background, legend, hover mode, x-axis date range
        selector, y-axis label, grid lines, and optional logarithmic y-scale.
        Shared by `lagged_performance_plot` and
        `smoothed_holdings_performance_plot`.

        Args:
            fig: The Plotly Figure to configure.
            title: Chart title text.
            log_scale: If True, set the primary y-axis to logarithmic scale.
        """
        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text="NAV (accumulated)")

        if log_scale:
            fig.update_yaxes(type="log")
            if hasattr(fig.layout, "yaxis"):  # pragma: no branch — plotly figures always have .yaxis
                fig.layout.yaxis.type = "log"

    def lagged_performance_plot(self, lags: list[int] | None = None, log_scale: bool = False) -> go.Figure:
        """Plot NAV_accumulated for multiple lagged portfolios.

        Creates a Plotly figure with one line per lag value showing the
        accumulated NAV series for the portfolio with cash positions
        shifted by that lag. By default, lags [0, 1, 2, 3, 4] are used.

        Args:
            lags: A list of integer lags to apply; defaults to [0, 1, 2, 3, 4].
            log_scale: If True, set the primary y-axis to logarithmic scale.

        Returns:
            A Plotly Figure containing one trace per requested lag.
        """
        if lags is None:
            lags = [0, 1, 2, 3, 4]
        if not isinstance(lags, list) or not all(isinstance(x, int) for x in lags):
            raise TypeError

        fig = go.Figure()
        for lag in lags:
            pf = self._portfolio if lag == 0 else self._portfolio.lag(lag)
            nav = pf.nav_accumulated
            fig.add_trace(
                go.Scatter(
                    x=nav["date"],
                    y=nav["NAV_accumulated"],
                    mode="lines",
                    name=f"lag {lag}",
                    line={"width": 1},
                )
            )

        self._apply_nav_layout(fig, title="NAV accumulated by lag", log_scale=log_scale)
        return fig

    def rolling_sharpe_plot(self, window: int = 63) -> go.Figure:
        """Plot rolling annualised Sharpe ratio over time.

        Computes the rolling Sharpe for each asset column using the given
        window and renders one line per asset.

        Args:
            window: Rolling-window size in periods. Defaults to 63.

        Returns:
            A Plotly Figure with one trace per asset.

        Raises:
            ValueError: If ``window`` is not a positive integer.
        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError(f"window must be a positive integer, got {window!r}")  # noqa: TRY003

        rolling = self._portfolio.stats.rolling_sharpe(rolling_period=window)

        fig = go.Figure()
        date_col = rolling["date"] if "date" in rolling.columns else None
        for col in rolling.columns:
            if col == "date":
                continue
            fig.add_trace(
                go.Scatter(
                    x=date_col,
                    y=rolling[col],
                    mode="lines",
                    name=col,
                    line={"width": 1},
                )
            )

        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")

        _apply_base_layout(fig, f"Rolling Sharpe Ratio ({window}-period window)")
        fig.update_yaxes(title_text="Sharpe ratio")
        return fig

    def rolling_volatility_plot(self, window: int = 63) -> go.Figure:
        """Plot rolling annualised volatility over time.

        Computes the rolling volatility for each asset column using the given
        window and renders one line per asset.

        Args:
            window: Rolling-window size in periods. Defaults to 63.

        Returns:
            A Plotly Figure with one trace per asset.

        Raises:
            ValueError: If ``window`` is not a positive integer.
        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError(f"window must be a positive integer, got {window!r}")  # noqa: TRY003

        rolling = self._portfolio.stats.rolling_volatility(rolling_period=window)

        fig = go.Figure()
        date_col = rolling["date"] if "date" in rolling.columns else None
        for col in rolling.columns:
            if col == "date":
                continue
            fig.add_trace(
                go.Scatter(
                    x=date_col,
                    y=rolling[col],
                    mode="lines",
                    name=col,
                    line={"width": 1},
                )
            )

        _apply_base_layout(fig, f"Rolling Volatility ({window}-period window)")
        fig.update_yaxes(title_text="Annualised volatility")
        return fig

    def annual_sharpe_plot(self) -> go.Figure:
        """Plot annualised Sharpe ratio broken down by calendar year.

        Computes the Sharpe ratio for each calendar year from the portfolio
        returns and renders a grouped bar chart with one bar per year per
        asset.

        Returns:
            A Plotly Figure with one bar group per asset.
        """
        breakdown = self._portfolio.stats.annual_breakdown()

        # Extract the sharpe row for each year
        sharpe_rows = breakdown.filter(pl.col("metric") == "sharpe")
        asset_cols = [c for c in sharpe_rows.columns if c not in ("year", "metric")]

        fig = go.Figure()
        for asset in asset_cols:
            fig.add_trace(
                go.Bar(
                    x=sharpe_rows["year"],
                    y=sharpe_rows[asset],
                    name=asset,
                )
            )

        fig.add_hline(y=0, line_width=1, line_color="gray")

        fig.update_layout(
            title="Annual Sharpe Ratio by Year",
            barmode="group",
            hovermode="x unified",
            plot_bgcolor="white",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        fig.update_yaxes(title_text="Sharpe ratio")
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey", title_text="Year")
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

    def smoothed_holdings_performance_plot(
        self,
        windows: list[int] | None = None,
        log_scale: bool = False,
    ) -> go.Figure:
        """Plot NAV_accumulated for smoothed-holding portfolios.

        Builds portfolios with cash positions smoothed by a trailing rolling
        mean over the previous ``n`` steps (window size n+1) for n in
        ``windows`` (defaults to [0, 1, 2, 3, 4]) and plots their
        accumulated NAV curves.

        Args:
            windows: List of non-negative integers specifying smoothing steps
                to include; defaults to [0, 1, 2, 3, 4].
            log_scale: If True, set the primary y-axis to logarithmic scale.

        Returns:
            A Plotly Figure containing one line per requested smoothing level.
        """
        if windows is None:
            windows = [0, 1, 2, 3, 4]
        if not isinstance(windows, list) or not all(isinstance(x, int) and x >= 0 for x in windows):
            raise TypeError

        fig = go.Figure()
        for n in windows:
            pf = self._portfolio if n == 0 else self._portfolio.smoothed_holding(n)
            nav = pf.nav_accumulated
            fig.add_trace(
                go.Scatter(
                    x=nav["date"],
                    y=nav["NAV_accumulated"],
                    mode="lines",
                    name=f"smooth {n}",
                    line={"width": 1},
                )
            )

        self._apply_nav_layout(fig, title="NAV accumulated by smoothed holdings", log_scale=log_scale)
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
