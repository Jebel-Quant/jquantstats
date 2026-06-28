"""Metric-row builders for the performance metrics table.

These helpers assemble the ordered ``(label, values)`` rows that
``Reports.metrics`` turns into a tidy DataFrame, and the period-window
calculations (CAGR/compounding since a cutoff) they rely on.
"""

from __future__ import annotations

import datetime
import math
from typing import Any, cast

import polars as pl

from ._formatting import _is_finite

# ── Private helpers ───────────────────────────────────────────────────────────


def _safe(fn: Any, *args: Any, **kwargs: Any) -> dict[str, float]:
    """Call ``fn(*args, **kwargs)`` and return ``{}`` on any exception."""
    try:
        result: dict[str, float] = fn(*args, **kwargs)
    except Exception:
        return {}
    return result


def _pct(d: dict[str, float]) -> dict[str, float]:
    """Multiply every finite value in *d* by 100."""
    return {k: v * 100.0 if _is_finite(v) else float("nan") for k, v in d.items()}


# ── Period-return helpers ─────────────────────────────────────────────────────


def _comp_since(all_df: pl.DataFrame, date_col: str, asset_cols: list[str], cutoff: Any) -> dict[str, float]:
    """Compounded return for each asset from *cutoff* to the last date."""
    filtered = all_df.filter(pl.col(date_col) >= cutoff)
    result: dict[str, float] = {}
    for col in asset_cols:
        s = filtered[col].drop_nulls().cast(pl.Float64)
        result[col] = float((1.0 + s).product()) - 1.0 if len(s) > 0 else float("nan")
    return result


def _cagr_since(
    all_df: pl.DataFrame,
    date_col: str,
    asset_cols: list[str],
    cutoff: Any,
    periods_per_year: float,
) -> dict[str, float]:
    """Annualised CAGR for each asset from *cutoff* to the last date."""
    filtered = all_df.filter(pl.col(date_col) >= cutoff)
    result: dict[str, float] = {}
    for col in asset_cols:
        s = filtered[col].drop_nulls().cast(pl.Float64)
        n = len(s)
        if n < 2:
            result[col] = float("nan")
            continue
        total = float((1.0 + s).product()) - 1.0
        years = n / periods_per_year
        result[col] = float(abs(1.0 + total) ** (1.0 / years) - 1.0)
    return result


# ── Metrics-row helpers ───────────────────────────────────────────────────────


def _cutoff_months(today: Any, n: int) -> Any:
    """Return the date *n* calendar months before *today*.

    Args:
        today: Reference date (must support ``.year``, ``.month``, ``.day``).
        n: Number of calendar months to subtract.

    Returns:
        A `datetime.date` exactly *n* months before *today*.

    """
    import calendar
    from datetime import date as _date

    y = today.year
    m = today.month
    for _ in range(n):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    d = min(today.day, calendar.monthrange(y, m)[1])
    return _date(y, m, d)


def _add_overview_rows(rows: list[tuple[str, dict[str, Any]]], s: Any, ppy: float) -> None:
    """Append overview metric rows to *rows*.

    Args:
        rows: Accumulator list of ``(label, values)`` tuples.
        s: Stats object providing the metric methods.
        ppy: Periods per year for annualisation.

    """
    rows.append(("Time in Market", _pct(_safe(s.exposure))))
    rows.append(("Cumulative Return", _pct(_safe(s.comp))))
    rows.append(("CAGR", _pct(_safe(s.cagr, periods=ppy))))


def _add_risk_adjusted_rows(rows: list[tuple[str, dict[str, Any]]], s: Any, ppy: float) -> None:
    """Append risk-adjusted ratio rows to *rows*.

    Args:
        rows: Accumulator list of ``(label, values)`` tuples.
        s: Stats object providing the metric methods.
        ppy: Periods per year for annualisation.

    """
    rows.append(("Sharpe", _safe(s.sharpe, periods=ppy)))
    rows.append(("Prob. Sharpe Ratio", _pct(_safe(s.probabilistic_sharpe_ratio))))
    rows.append(("Sortino", _safe(s.sortino, periods=ppy)))
    rows.append(("Sortino / √2", _safe(s.adjusted_sortino, periods=ppy)))
    rows.append(("Omega", _safe(s.omega, periods=ppy)))


def _add_drawdown_rows(rows: list[tuple[str, dict[str, Any]]], s: Any) -> None:
    """Append drawdown metric rows to *rows*.

    Args:
        rows: Accumulator list of ``(label, values)`` tuples.
        s: Stats object providing the metric methods.

    """
    rows.append(("Max Drawdown", _pct(_safe(s.max_drawdown))))
    rows.append(("Max DD Duration", _safe(s.max_drawdown_duration)))
    rows.append(("Avg Drawdown", _pct(_safe(s.avg_drawdown))))
    rows.append(("Recovery Factor", _safe(s.recovery_factor)))
    rows.append(("Ulcer Index", _safe(s.ulcer_index)))
    rows.append(("Serenity Index", _safe(s.serenity_index)))


def _add_trading_rows(rows: list[tuple[str, dict[str, Any]]], s: Any) -> None:
    """Append trading metric rows to *rows*.

    Args:
        rows: Accumulator list of ``(label, values)`` tuples.
        s: Stats object providing the metric methods.

    """
    rows.append(("Gain/Pain Ratio", _safe(s.gain_to_pain_ratio)))
    rows.append(("Gain/Pain (1M)", _safe(s.gain_to_pain_ratio, aggregate="ME")))
    rows.append(("Payoff Ratio", _safe(s.payoff_ratio)))
    rows.append(("Profit Factor", _safe(s.profit_factor)))
    rows.append(("Common Sense Ratio", _safe(s.common_sense_ratio)))
    rows.append(("CPC Index", _safe(s.cpc_index)))
    rows.append(("Tail Ratio", _safe(s.tail_ratio)))
    rows.append(("Outlier Win Ratio", _safe(s.outlier_win_ratio)))
    rows.append(("Outlier Loss Ratio", _safe(s.outlier_loss_ratio)))


def _add_recent_returns_rows(
    rows: list[tuple[str, dict[str, Any]]],
    all_df: pl.DataFrame,
    date_col: str,
    asset_cols: list[str],
    ppy: float,
    s: Any,
) -> None:
    """Append date-filtered recent return rows to *rows*.

    Args:
        rows: Accumulator list of ``(label, values)`` tuples.
        all_df: Combined DataFrame containing date and return columns.
        date_col: Name of the date column in *all_df*.
        asset_cols: Names of asset return columns in *all_df*.
        ppy: Periods per year for annualisation.
        s: Stats object used for the all-time CAGR.

    """
    today = cast(datetime.date, all_df[date_col].max())
    mtd_start = today.replace(day=1)
    ytd_start = today.replace(month=1, day=1)

    rows.append(("MTD", _pct(_comp_since(all_df, date_col, asset_cols, mtd_start))))
    rows.append(("3M", _pct(_comp_since(all_df, date_col, asset_cols, _cutoff_months(today, 3)))))
    rows.append(("6M", _pct(_comp_since(all_df, date_col, asset_cols, _cutoff_months(today, 6)))))
    rows.append(("YTD", _pct(_comp_since(all_df, date_col, asset_cols, ytd_start))))
    rows.append(("1Y", _pct(_comp_since(all_df, date_col, asset_cols, _cutoff_months(today, 12)))))
    rows.append(("3Y (ann.)", _pct(_cagr_since(all_df, date_col, asset_cols, _cutoff_months(today, 36), ppy))))
    rows.append(("5Y (ann.)", _pct(_cagr_since(all_df, date_col, asset_cols, _cutoff_months(today, 60), ppy))))
    rows.append(("All-time (ann.)", _pct(_safe(s.cagr, periods=ppy))))


def _add_full_mode_rows(
    rows: list[tuple[str, dict[str, Any]]],
    s: Any,
    ppy: float,
    data: Any,
    all_df: pl.DataFrame | None,
    date_col: str | None,
    asset_cols: list[str],
) -> None:
    """Append all full-mode extension rows to *rows*.

    Covers smart ratios, extended risk, averages, expected returns, tail risk,
    streaks, best/worst periods, and benchmark metrics.

    Args:
        rows: Accumulator list of ``(label, values)`` tuples.
        s: Stats object providing the metric methods.
        ppy: Periods per year for annualisation.
        data: The DataLike object (used for benchmark access).
        all_df: Combined DataFrame or ``None`` if unavailable.
        date_col: Name of the date column or ``None`` if unavailable.
        asset_cols: Asset column names.

    """
    # Smart ratios
    rows.append(("Smart Sharpe", _safe(s.smart_sharpe, periods=ppy)))
    ss = _safe(s.smart_sortino, periods=ppy)
    rows.append(("Smart Sortino", ss))
    rows.append(("Smart Sortino / √2", {k: v / math.sqrt(2) for k, v in ss.items() if _is_finite(v)}))

    # Risk
    rows.append(("Volatility (ann.)", _pct(_safe(s.volatility, periods=ppy))))
    rows.append(("Calmar", _safe(s.calmar, periods=ppy)))
    rows.append(("Risk-Adjusted Return", _pct(_safe(s.rar, periods=ppy))))
    rows.append(("Risk-Return Ratio", _safe(s.risk_return_ratio)))
    rows.append(("Ulcer Performance Index", _safe(s.ulcer_performance_index)))
    rows.append(("Skew", _safe(s.skew)))
    rows.append(("Kurtosis", _safe(s.kurtosis)))

    # Averages
    rows.append(("Avg. Return", _pct(_safe(s.avg_return))))
    rows.append(("Avg. Win", _pct(_safe(s.avg_win))))
    rows.append(("Avg. Loss", _pct(_safe(s.avg_loss))))
    rows.append(("Win/Loss Ratio", _safe(s.payoff_ratio)))
    rows.append(("Profit Ratio", _safe(s.profit_ratio)))
    rows.append(("Win Rate", _pct(_safe(s.win_rate))))
    rows.append(("Monthly Win Rate", _pct(_safe(s.monthly_win_rate))))

    # Expected returns
    rows.append(("Expected Daily", _pct(_safe(s.expected_return))))
    rows.append(("Expected Monthly", _pct(_safe(s.expected_return, aggregate="monthly"))))
    rows.append(("Expected Yearly", _pct(_safe(s.expected_return, aggregate="yearly"))))

    # Tail risk
    rows.append(("Kelly Criterion", _pct(_safe(s.kelly_criterion))))
    rows.append(("Risk of Ruin", _pct(_safe(s.risk_of_ruin))))
    rows.append(("Daily VaR", _pct(_safe(s.value_at_risk))))
    rows.append(("Expected Shortfall (cVaR)", _pct(_safe(s.conditional_value_at_risk))))

    # Streaks & best / worst
    rows.append(("Max Consecutive Wins", _safe(s.consecutive_wins)))
    rows.append(("Max Consecutive Losses", _safe(s.consecutive_losses)))
    rows.append(("Best Day", _pct(_safe(s.best))))
    rows.append(("Worst Day", _pct(_safe(s.worst))))

    # Benchmark greeks (only if benchmark is present). Without a benchmark,
    # ``s.greeks()`` reaches ``benchmark_data.columns`` on a ``None`` benchmark
    # and raises ``AttributeError``; we tolerate only that case and omit the
    # rows. Any other error (malformed column, polars schema/arithmetic bug)
    # propagates so genuine bugs are not silently swallowed.
    try:
        greeks = s.greeks()
        if greeks:  # pragma: no branch — greeks() raises without a benchmark, so falsy is unreachable
            beta = {k: v["beta"] for k, v in greeks.items()}
            alpha = {k: v["alpha"] * 100.0 for k, v in greeks.items()}
            rows.append(("Beta", beta))
            rows.append(("Alpha", alpha))
    except AttributeError:
        pass

    # Correlation against the benchmark. When the benchmark column is absent
    # (no benchmark configured, or it is missing from the joined frame) polars
    # raises ``ColumnNotFoundError`` and ``None.columns`` raises
    # ``AttributeError`` — both mean "no benchmark to correlate against", so the
    # row is omitted. Any other error propagates.
    try:
        bench_obj = getattr(data, "benchmark", None)
        if bench_obj is not None and all_df is not None and date_col is not None:  # pragma: no branch
            bench_col = bench_obj.columns[0]
            corr_dict: dict[str, float] = {}
            for ac in asset_cols:
                if ac == bench_col:
                    continue
                sub = all_df.select([date_col, ac, bench_col]).drop_nulls()
                corr_val = float(sub.select(pl.corr(ac, bench_col))[0, 0])
                corr_dict[ac] = corr_val * 100.0
            rows.append(("Correlation", corr_dict))
    except (AttributeError, pl.exceptions.ColumnNotFoundError):
        pass

    rows.append(("R²", _safe(s.r_squared)))
    rows.append(("Treynor Ratio", _safe(s.treynor_ratio, periods=ppy)))


def _build_metrics_df(rows: list[tuple[str, dict[str, Any]]]) -> pl.DataFrame:
    """Build a metrics `pl.DataFrame` from accumulated row data.

    Args:
        rows: List of ``(label, values)`` tuples where *values* maps asset
            names to numeric results.

    Returns:
        A DataFrame with a leading ``"Metric"`` column and one column per
        asset, preserving the insertion order of both metrics and assets.

    """
    all_assets: list[str] = []
    seen: set[str] = set()
    for _, vals in rows:
        for k in vals:
            if k not in seen:
                all_assets.append(k)
                seen.add(k)
    return pl.DataFrame([{"Metric": label, **{a: vals.get(a) for a in all_assets}} for label, vals in rows])
