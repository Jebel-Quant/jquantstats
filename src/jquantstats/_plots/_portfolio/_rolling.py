"""Rolling-window and per-year risk charts for a portfolio.

Split out of the former single-module `_plots/_portfolio.py`; composed into
:class:`PortfolioPlots` by `_core.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl

from .._data._styling import _apply_base_layout

if TYPE_CHECKING:
    from .._protocol import PortfolioLike


class _RollingPortfolioPlotsMixin:
    """Rolling-window and annual risk charts for :class:`PortfolioPlots`."""

    __slots__ = ()

    _portfolio: PortfolioLike

    @staticmethod
    def _validate_window(window: int) -> None:
        """Reject a non-positive or non-integer rolling window.

        Args:
            window: The candidate rolling-window size.

        Raises:
            ValueError: If ``window`` is not a positive integer.
        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError(f"window must be a positive integer, got {window!r}")  # noqa: TRY003

    @staticmethod
    def _line_per_column(rolling: pl.DataFrame) -> go.Figure:
        """Render one line trace per non-date column of *rolling*.

        Shared by `rolling_sharpe_plot` and `rolling_volatility_plot`, which
        differ only in the metric they fetch and the labels they apply.

        Args:
            rolling: A frame with an optional ``date`` column and one column
                per asset.

        Returns:
            A Figure carrying the traces, with no layout applied yet.
        """
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
        self._validate_window(window)

        fig = self._line_per_column(self._portfolio.stats.rolling_sharpe(rolling_period=window))
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
        self._validate_window(window)

        fig = self._line_per_column(self._portfolio.stats.rolling_volatility(rolling_period=window))

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
