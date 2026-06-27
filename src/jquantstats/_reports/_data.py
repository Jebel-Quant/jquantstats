"""Financial report generation from returns data."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from jquantstats._protocol import DataLike

from ._html import (
    _build_full_html,
    _drawdowns_section_html,
    _metrics_table_html,
    _try_plotly_div,
)
from ._metrics import (
    _add_drawdown_rows,
    _add_full_mode_rows,
    _add_overview_rows,
    _add_recent_returns_rows,
    _add_risk_adjusted_rows,
    _add_trading_rows,
    _build_metrics_df,
)


class Reports:
    """A class for generating financial reports from Data objects.

    This class provides methods for calculating and formatting various financial metrics
    into report-ready formats such as DataFrames.
    """

    __slots__ = ("_data",)

    def __init__(self, data: DataLike) -> None:
        self._data = data

    def metrics(
        self,
        mode: str = "basic",
        periods_per_year: int | float = 252,
        rf: float = 0.0,
    ) -> pl.DataFrame:
        """Comprehensive performance metrics table matching ``qs.reports.metrics``.

        Computes an ordered set of performance, risk, and trading metrics for
        every asset in the dataset and returns them as a tidy DataFrame.

        Args:
            mode: ``"basic"`` (default) for core metrics, ``"full"`` for the
                extended set including smart ratios, expected returns, streaks,
                best/worst periods, win rates, and benchmark greeks.
            periods_per_year: Annualisation factor. Defaults to 252.
            rf: Annualised risk-free rate used in ratio calculations.
                Defaults to 0.0.

        Returns:
            pl.DataFrame: One row per metric, one column per asset, plus a
            leading ``"Metric"`` column with the metric label.

        """
        s = self._data.stats
        ppy = float(periods_per_year)
        is_full = mode.lower() == "full"

        rows: list[tuple[str, dict[str, Any]]] = []

        all_df: pl.DataFrame | None = getattr(self._data, "all", None)
        asset_cols: list[str] = []
        date_col: str | None = None
        has_dates = False

        if all_df is not None:  # pragma: no branch — Data always exposes .all; getattr default is defensive
            date_col = all_df.columns[0]
            asset_cols = [c for c in all_df.columns if c != date_col]
            has_dates = all_df[date_col].dtype.is_temporal()

        _add_overview_rows(rows, s, ppy)
        _add_risk_adjusted_rows(rows, s, ppy)
        _add_drawdown_rows(rows, s)
        _add_trading_rows(rows, s)

        if has_dates and date_col is not None and all_df is not None:  # pragma: no branch
            _add_recent_returns_rows(rows, all_df, date_col, asset_cols, ppy, s)

        if is_full:
            _add_full_mode_rows(rows, s, ppy, self._data, all_df, date_col, asset_cols)

        return _build_metrics_df(rows)

    def full(
        self,
        title: str = "Performance Report",
        periods_per_year: int | float = 252,
        rf: float = 0.0,
    ) -> str:
        """Generate a self-contained HTML performance report.

        Combines a comprehensive metrics table (full mode), worst-5 drawdown
        periods per asset, and interactive Plotly charts into a single
        dark-themed HTML document.

        Args:
            title: Page ``<h1>`` title. Defaults to ``"Performance Report"``.
            periods_per_year: Annualisation factor passed to
                `metrics`. Defaults to 252.
            rf: Annualised risk-free rate. Defaults to 0.0.

        Returns:
            str: A complete, self-contained HTML document.

        """
        # ── Metrics ───────────────────────────────────────────────────────────
        metrics_df = self.metrics(mode="full", periods_per_year=periods_per_year, rf=rf)
        assets = [c for c in metrics_df.columns if c != "Metric"]
        metrics_html = _metrics_table_html(metrics_df)

        # ── Period info for header ────────────────────────────────────────────
        all_df: pl.DataFrame | None = getattr(self._data, "all", None)
        period_info = ""
        temporal_index = False
        if all_df is not None:  # pragma: no branch — Data always exposes .all; getattr default is defensive
            date_col = all_df.columns[0]
            temporal_index = all_df[date_col].dtype.is_temporal()
            if temporal_index:
                start_dt = all_df[date_col].min()
                end_dt = all_df[date_col].max()
                n = len(all_df)
                period_info = f"{start_dt!s} → {end_dt!s} | {n:,} observations"

        # ── Drawdowns ─────────────────────────────────────────────────────────
        drawdowns_html = _drawdowns_section_html(self._data, assets)

        # ── Charts ────────────────────────────────────────────────────────────
        plots = getattr(self._data, "plots", None)
        chart_parts: list[str] = []
        if plots is not None:  # pragma: no branch — Data always exposes .plots; getattr default is defensive
            _chart_methods: list[tuple[str, dict[str, Any]]] = [
                ("snapshot", {}),
                ("returns", {}),
                ("drawdown", {}),
                ("rolling_sharpe", {}),
                ("rolling_volatility", {}),
                ("monthly_heatmap", {}),
                ("yearly_returns", {}),
                ("histogram", {}),
            ]
            # These charts aggregate by calendar period (resample, dt.year/
            # dt.month) and cannot be computed for an integer index.
            _calendar_charts = {"snapshot", "monthly_heatmap", "yearly_returns"}
            if not temporal_index:
                skipped = ", ".join(m for m, _ in _chart_methods if m in _calendar_charts)
                warnings.warn(
                    f"Index is not temporal; skipping calendar-based charts: {skipped}.",
                    stacklevel=2,
                )
            for method, kwargs in _chart_methods:
                if not temporal_index and method in _calendar_charts:
                    continue
                fn = getattr(plots, method, None)
                if fn is None:
                    continue
                div = _try_plotly_div(fn(**kwargs), include_cdn=not chart_parts)
                if div:  # pragma: no branch — _try_plotly_div only returns falsy on render failure
                    chart_parts.append(f'<div style="margin-bottom:24px">{div}</div>')

        charts_html = "\n".join(chart_parts) if chart_parts else "<p>No charts available.</p>"

        return _build_full_html(
            title=title,
            period_info=period_info,
            assets_str=", ".join(assets),
            metrics_html=metrics_html,
            drawdowns_html=drawdowns_html,
            charts_html=charts_html,
        )
