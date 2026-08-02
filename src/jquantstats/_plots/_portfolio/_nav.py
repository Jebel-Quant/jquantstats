"""NAV-accumulated performance charts: snapshot, lag sweep, holdings smoothing.

Split out of the former single-module `_plots/_portfolio.py`; composed into
:class:`PortfolioPlots` by `_core.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .._data._styling import _apply_base_layout

if TYPE_CHECKING:
    from .._protocol import PortfolioLike


class _NavPlotsMixin:
    """Accumulated-NAV charts for :class:`PortfolioPlots`."""

    __slots__ = ()

    _portfolio: PortfolioLike

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
