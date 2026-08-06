# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.20.4",
#     "jquantstats",
#     "numpy>=2.0.0",
#     "polars>=1.0.0",
#     "plotly>=6.0.0",
#     "jinja2>=3.1.0",
# ]
# [tool.uv.sources]
# jquantstats = { path = "../../..", editable = true }
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import os
    from datetime import date, timedelta
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl

    from jquantstats import Data, Portfolio

    NOTEBOOK_DIR = Path(__file__).parent


@app.cell
def cell_intro() -> None:
    """Render the introduction."""
    mo.md(
        r"""
        # 📊 jquantstats — Plots & Reports Gallery

        This notebook illustrates **every plot and report** available in `jquantstats`.

        It is organised into two sections:

        | Section | Entry point | What you get |
        |---------|-------------|--------------|
        | **Part 1 — Data** | `Data.from_returns(...)` | 15 `DataPlots` methods · `metrics()` · `full()` |
        | **Part 2 — Portfolio** | `Portfolio.from_cash_position(...)` | 10 `PortfolioPlots` · `to_html()` |

        **Data used in Part 1:** real AAPL + META daily returns (≈ 11 k rows) with SPY as benchmark,
        loaded from CSV files bundled with the repository.

        **Data used in Part 2:** synthetic 3-asset price + position series (500 trading days).
        """
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — DATA (RETURNS-SERIES)
# ─────────────────────────────────────────────────────────────────────────────


@app.cell
def cell_part1_header() -> None:
    """Part 1 section header."""
    mo.md("---\n## Part 1 — `Data` entry point (returns series)")
    return


@app.cell
def cell_load_data():
    """Load portfolio returns and SPY benchmark from CSV."""
    data_dir = NOTEBOOK_DIR / "data"

    returns_df = pl.read_csv(data_dir / "portfolio.csv", try_parse_dates=True).with_columns(
        [
            pl.col("AAPL").cast(pl.Float64, strict=False),
            pl.col("META").cast(pl.Float64, strict=False),
            pl.col("Date").cast(pl.Date, strict=False),
        ]
    )

    benchmark_df = pl.read_csv(data_dir / "benchmark.csv", try_parse_dates=True).select(["Date", "SPY -- Benchmark"])

    data = Data.from_returns(returns=returns_df, benchmark=benchmark_df)

    benchmark_cols = data.benchmark.columns if data.benchmark is not None else []
    mo.md(
        f"""
        **Data loaded** ✅

        - Assets: `{data.assets}`
        - Benchmark columns: `{benchmark_cols}`
        - Rows: `{returns_df.height:,}`
        - Date range: `{returns_df["Date"].min()}` → `{returns_df["Date"].max()}`
        """
    )
    return (data,)


# ── DataPlots ─────────────────────────────────────────────────────────────────


@app.cell
def cell_dp_snapshot_header() -> None:
    """DataPlots — snapshot header."""
    mo.md("### 1 · `snapshot` — three-panel dashboard")
    return


@app.cell
def cell_dp_snapshot(data):
    """DataPlots.snapshot — cumulative returns, drawdown, daily returns."""
    fig_snapshot = data.plots.snapshot(title="AAPL + META vs SPY")
    return (fig_snapshot,)


@app.cell
def cell_dp_snapshot_show(fig_snapshot) -> None:
    """Display snapshot figure."""
    fig_snapshot
    return


@app.cell
def cell_dp_returns_header() -> None:
    """DataPlots — cumulative returns header."""
    mo.md("### 2 · `returns` — cumulative compounded returns")
    return


@app.cell
def cell_dp_returns(data):
    """DataPlots.returns."""
    fig_returns = data.plots.returns(title="Cumulative Returns")
    return (fig_returns,)


@app.cell
def cell_dp_returns_show(fig_returns) -> None:
    """Display cumulative returns figure."""
    fig_returns
    return


@app.cell
def cell_dp_log_returns_header() -> None:
    """DataPlots — log returns header."""
    mo.md("### 3 · `log_returns` — cumulative log returns")
    return


@app.cell
def cell_dp_log_returns(data):
    """DataPlots.log_returns."""
    fig_log_returns = data.plots.log_returns(title="Log Returns")
    return (fig_log_returns,)


@app.cell
def cell_dp_log_returns_show(fig_log_returns) -> None:
    """Display log returns figure."""
    fig_log_returns
    return


@app.cell
def cell_dp_daily_header() -> None:
    """DataPlots — daily returns header."""
    mo.md("### 4 · `daily_returns` — daily returns bar chart")
    return


@app.cell
def cell_dp_daily(data):
    """DataPlots.daily_returns."""
    fig_daily = data.plots.daily_returns(title="Daily Returns")
    return (fig_daily,)


@app.cell
def cell_dp_daily_show(fig_daily) -> None:
    """Display daily returns figure."""
    fig_daily
    return


@app.cell
def cell_dp_yearly_header() -> None:
    """DataPlots — yearly returns header."""
    mo.md("### 5 · `yearly_returns` — annual returns bar chart")
    return


@app.cell
def cell_dp_yearly(data):
    """DataPlots.yearly_returns."""
    fig_yearly = data.plots.yearly_returns(title="Yearly Returns")
    return (fig_yearly,)


@app.cell
def cell_dp_yearly_show(fig_yearly) -> None:
    """Display yearly returns figure."""
    fig_yearly
    return


@app.cell
def cell_dp_monthly_header() -> None:
    """DataPlots — monthly returns header."""
    mo.md("### 6 · `monthly_returns` — monthly returns bar chart")
    return


@app.cell
def cell_dp_monthly(data):
    """DataPlots.monthly_returns."""
    fig_monthly = data.plots.monthly_returns(title="Monthly Returns")
    return (fig_monthly,)


@app.cell
def cell_dp_monthly_show(fig_monthly) -> None:
    """Display monthly returns figure."""
    fig_monthly
    return


@app.cell
def cell_dp_heatmap_header() -> None:
    """DataPlots — monthly heatmap header."""
    mo.md("### 7 · `monthly_heatmap` — year × month heatmap")
    return


@app.cell
def cell_dp_heatmap(data):
    """DataPlots.monthly_heatmap (AAPL)."""
    fig_heatmap = data.plots.monthly_heatmap(title="Monthly Heatmap — AAPL", asset="AAPL")
    return (fig_heatmap,)


@app.cell
def cell_dp_heatmap_show(fig_heatmap) -> None:
    """Display monthly heatmap figure."""
    fig_heatmap
    return


@app.cell
def cell_dp_histogram_header() -> None:
    """DataPlots — histogram header."""
    mo.md("### 8 · `histogram` — overlaid return histograms, one per series")
    return


@app.cell
def cell_dp_histogram(data):
    """DataPlots.histogram."""
    fig_histogram = data.plots.histogram(title="Returns Distribution")
    return (fig_histogram,)


@app.cell
def cell_dp_histogram_show(fig_histogram) -> None:
    """Display histogram figure."""
    fig_histogram
    return


@app.cell
def cell_dp_distribution_header() -> None:
    """DataPlots — distribution header."""
    mo.md("### 9 · `distribution` — returns across aggregation periods (box plot)")
    return


@app.cell
def cell_dp_distribution(data):
    """DataPlots.distribution."""
    fig_distribution = data.plots.distribution(title="Return Distribution by Period")
    return (fig_distribution,)


@app.cell
def cell_dp_distribution_show(fig_distribution) -> None:
    """Display distribution figure."""
    fig_distribution
    return


@app.cell
def cell_dp_drawdown_header() -> None:
    """DataPlots — drawdown header."""
    mo.md("### 10 · `drawdown` — underwater equity curve")
    return


@app.cell
def cell_dp_drawdown(data):
    """DataPlots.drawdown."""
    fig_drawdown = data.plots.drawdown(title="Drawdowns")
    return (fig_drawdown,)


@app.cell
def cell_dp_drawdown_show(fig_drawdown) -> None:
    """Display drawdown figure."""
    fig_drawdown
    return


@app.cell
def cell_dp_drawdown_periods_header() -> None:
    """DataPlots — drawdown periods header."""
    mo.md("### 11 · `drawdowns_periods` — top-5 drawdown shading (AAPL)")
    return


@app.cell
def cell_dp_drawdown_periods(data):
    """DataPlots.drawdowns_periods."""
    fig_dd_periods = data.plots.drawdowns_periods(n=5, title="Top 5 Drawdown Periods — AAPL", asset="AAPL")
    return (fig_dd_periods,)


@app.cell
def cell_dp_drawdown_periods_show(fig_dd_periods) -> None:
    """Display drawdown periods figure."""
    fig_dd_periods
    return


@app.cell
def cell_dp_earnings_header() -> None:
    """DataPlots — earnings header."""
    mo.md("### 12 · `earnings` — dollar equity curve")
    return


@app.cell
def cell_dp_earnings(data):
    """DataPlots.earnings."""
    fig_earnings = data.plots.earnings(start_balance=100_000, title="Portfolio Earnings ($100k start)")
    return (fig_earnings,)


@app.cell
def cell_dp_earnings_show(fig_earnings) -> None:
    """Display earnings figure."""
    fig_earnings
    return


@app.cell
def cell_dp_rolling_sharpe_header() -> None:
    """DataPlots — rolling Sharpe header."""
    mo.md("### 13 · `rolling_sharpe` — 6-month rolling Sharpe ratio")
    return


@app.cell
def cell_dp_rolling_sharpe(data):
    """DataPlots.rolling_sharpe."""
    fig_rolling_sharpe = data.plots.rolling_sharpe(rolling_period=126, title="Rolling Sharpe (126-day)")
    return (fig_rolling_sharpe,)


@app.cell
def cell_dp_rolling_sharpe_show(fig_rolling_sharpe) -> None:
    """Display rolling Sharpe figure."""
    fig_rolling_sharpe
    return


@app.cell
def cell_dp_rolling_sortino_header() -> None:
    """DataPlots — rolling Sortino header."""
    mo.md("### 14 · `rolling_sortino` — 6-month rolling Sortino ratio")
    return


@app.cell
def cell_dp_rolling_sortino(data):
    """DataPlots.rolling_sortino."""
    fig_rolling_sortino = data.plots.rolling_sortino(rolling_period=126, title="Rolling Sortino (126-day)")
    return (fig_rolling_sortino,)


@app.cell
def cell_dp_rolling_sortino_show(fig_rolling_sortino) -> None:
    """Display rolling Sortino figure."""
    fig_rolling_sortino
    return


@app.cell
def cell_dp_rolling_vol_header() -> None:
    """DataPlots — rolling volatility header."""
    mo.md("### 15 · `rolling_volatility` — 6-month rolling annualised volatility")
    return


@app.cell
def cell_dp_rolling_vol(data):
    """DataPlots.rolling_volatility."""
    fig_rolling_vol = data.plots.rolling_volatility(rolling_period=126, title="Rolling Volatility (126-day)")
    return (fig_rolling_vol,)


@app.cell
def cell_dp_rolling_vol_show(fig_rolling_vol) -> None:
    """Display rolling volatility figure."""
    fig_rolling_vol
    return


@app.cell
def cell_dp_rolling_beta_header() -> None:
    """DataPlots — rolling beta header."""
    mo.md("### 16 · `rolling_beta` — rolling beta versus benchmark")
    return


@app.cell
def cell_dp_rolling_beta(data):
    """DataPlots.rolling_beta."""
    fig_rolling_beta = data.plots.rolling_beta(rolling_period=126, title="Rolling Beta vs SPY (126-day)")
    return (fig_rolling_beta,)


@app.cell
def cell_dp_rolling_beta_show(fig_rolling_beta) -> None:
    """Display rolling beta figure."""
    fig_rolling_beta
    return


# ── Reports ───────────────────────────────────────────────────────────────────


@app.cell
def cell_reports_header() -> None:
    """Reports section header."""
    mo.md("### Reports — `Reports.metrics()` and `Reports.full()`")
    return


@app.cell
def cell_reports_metrics(data):
    """Reports.metrics — comprehensive performance table."""
    mo.md("#### `Reports.metrics(mode='full')` — performance metrics table")
    metrics_df = data.reports.metrics(mode="full")
    return (metrics_df,)


@app.cell
def cell_reports_metrics_show(metrics_df) -> None:
    """Display metrics dataframe."""
    mo.plain_text(str(metrics_df))
    return


@app.cell
def cell_reports_full(data):
    """Reports.full — self-contained HTML report."""
    mo.md("#### `Reports.full()` — self-contained HTML report")
    html_report = data.reports.full(title="AAPL + META vs SPY — Performance Report")
    mo.md(f"HTML report generated: **{len(html_report):,}** characters.")
    return (html_report,)


@app.cell
def cell_reports_full_export(html_report) -> None:
    """Write the Data HTML report to NOTEBOOK_OUTPUT_FOLDER if set."""
    _output_folder = os.environ.get("NOTEBOOK_OUTPUT_FOLDER")
    if _output_folder:
        _artefact_path = Path(_output_folder) / "data_report.html"
        _artefact_path.write_text(html_report)
        _msg = mo.md(f"✅ Data report saved to `{_artefact_path}`")
    else:
        _msg = mo.md(
            "ℹ️ `NOTEBOOK_OUTPUT_FOLDER` is not set — artefact saving is skipped "
            "(this variable is set automatically by `rhiza_marimo`)."
        )
    return


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — PORTFOLIO (PRICES + POSITIONS)
# ─────────────────────────────────────────────────────────────────────────────


@app.cell
def cell_part2_header() -> None:
    """Part 2 section header."""
    mo.md("---\n## Part 2 — `Portfolio` entry point (prices + positions)")
    return


@app.cell
def cell_portfolio_data():
    """Generate synthetic 3-asset price and cash-position data (500 days)."""
    n = 500
    start_date = date(2021, 1, 4)
    end_date = start_date + timedelta(days=n - 1)
    dates = pl.date_range(start=start_date, end=end_date, interval="1d", eager=True).cast(pl.Date)

    rng = np.random.default_rng(0)

    # Asset X: steady uptrend
    price_x = pl.Series(
        [100.0 * np.exp(0.0003 * i + 0.012 * rng.standard_normal()) for i in range(n)],
        dtype=pl.Float64,
    )
    # Asset Y: mean-reverting around 50
    price_y = pl.Series(
        [50.0 + 5.0 * np.sin(0.05 * i) + 1.5 * rng.standard_normal() for i in range(n)],
        dtype=pl.Float64,
    )
    # Asset Z: higher volatility
    price_z = pl.Series(
        [200.0 * np.exp(0.0001 * i + 0.025 * rng.standard_normal()) for i in range(n)],
        dtype=pl.Float64,
    )

    prices = pl.DataFrame({"date": dates, "X": price_x, "Y": price_y, "Z": price_z})

    # Cash positions ($ value held in each asset)
    pos_x = pl.Series([5000.0 + 10.0 * i for i in range(n)], dtype=pl.Float64)
    pos_y = pl.Series([3000.0 + float(i % 10) * 200.0 for i in range(n)], dtype=pl.Float64)
    pos_z = pl.Series([2000.0 - 2.0 * (i % 20) for i in range(n)], dtype=pl.Float64)

    positions = pl.DataFrame({"date": dates, "X": pos_x, "Y": pos_y, "Z": pos_z})

    mo.md(
        f"""
        **Synthetic portfolio data generated** ✅

        - Assets: X, Y, Z
        - Trading days: {n}
        - Date range: `{start_date}` → `{end_date}`
        """
    )
    return (dates, n, positions, price_x, price_y, price_z, prices, pos_x, pos_y, pos_z, rng, start_date, end_date)


@app.cell
def cell_portfolio_build(prices, positions):
    """Build Portfolio from synthetic prices and cash positions."""
    portfolio = Portfolio.from_cash_position(
        prices=prices,
        cash_position=positions,
        aum=1_000_000.0,
        cost_bps=2.0,
    )
    mo.md(
        f"""
        **Portfolio created** ✅

        - Assets: `{portfolio.assets}`
        - AUM: `{portfolio.aum:,.0f}`
        - Cost model: 2 bps one-way
        """
    )
    return (portfolio,)


# ── PortfolioPlots ────────────────────────────────────────────────────────────


@app.cell
def cell_pp_snapshot_header() -> None:
    """PortfolioPlots — snapshot header."""
    mo.md("### 1 · `snapshot` — NAV + drawdown dashboard")
    return


@app.cell
def cell_pp_snapshot(portfolio):
    """PortfolioPlots.snapshot."""
    fig_pf_snapshot = portfolio.plots.snapshot()
    return (fig_pf_snapshot,)


@app.cell
def cell_pp_snapshot_show(fig_pf_snapshot) -> None:
    """Display portfolio snapshot figure."""
    fig_pf_snapshot
    return


@app.cell
def cell_pp_lead_lag_header() -> None:
    """PortfolioPlots — lead/lag IR header."""
    mo.md("### 2 · `lead_lag_ir_plot` — Sharpe ratio across execution lags")
    return


@app.cell
def cell_pp_lead_lag(portfolio):
    """PortfolioPlots.lead_lag_ir_plot."""
    fig_lead_lag = portfolio.plots.lead_lag_ir_plot(start=-5, end=10)
    return (fig_lead_lag,)


@app.cell
def cell_pp_lead_lag_show(fig_lead_lag) -> None:
    """Display lead/lag IR figure."""
    fig_lead_lag
    return


@app.cell
def cell_pp_lagged_perf_header() -> None:
    """PortfolioPlots — lagged performance header."""
    mo.md("### 3 · `lagged_performance_plot` — NAV curves for multiple execution lags")
    return


@app.cell
def cell_pp_lagged_perf(portfolio):
    """PortfolioPlots.lagged_performance_plot."""
    fig_lagged_perf = portfolio.plots.lagged_performance_plot(lags=[0, 1, 2, 3, 5])
    return (fig_lagged_perf,)


@app.cell
def cell_pp_lagged_perf_show(fig_lagged_perf) -> None:
    """Display lagged performance figure."""
    fig_lagged_perf
    return


@app.cell
def cell_pp_rolling_sharpe_header() -> None:
    """PortfolioPlots — rolling Sharpe header."""
    mo.md("### 4 · `rolling_sharpe_plot` — rolling annualised Sharpe ratio")
    return


@app.cell
def cell_pp_rolling_sharpe(portfolio):
    """PortfolioPlots.rolling_sharpe_plot."""
    fig_pf_rolling_sharpe = portfolio.plots.rolling_sharpe_plot(window=63)
    return (fig_pf_rolling_sharpe,)


@app.cell
def cell_pp_rolling_sharpe_show(fig_pf_rolling_sharpe) -> None:
    """Display portfolio rolling Sharpe figure."""
    fig_pf_rolling_sharpe
    return


@app.cell
def cell_pp_rolling_vol_header() -> None:
    """PortfolioPlots — rolling volatility header."""
    mo.md("### 5 · `rolling_volatility_plot` — rolling annualised volatility")
    return


@app.cell
def cell_pp_rolling_vol(portfolio):
    """PortfolioPlots.rolling_volatility_plot."""
    fig_pf_rolling_vol = portfolio.plots.rolling_volatility_plot(window=63)
    return (fig_pf_rolling_vol,)


@app.cell
def cell_pp_rolling_vol_show(fig_pf_rolling_vol) -> None:
    """Display portfolio rolling volatility figure."""
    fig_pf_rolling_vol
    return


@app.cell
def cell_pp_annual_sharpe_header() -> None:
    """PortfolioPlots — annual Sharpe header."""
    mo.md("### 6 · `annual_sharpe_plot` — calendar-year Sharpe breakdown")
    return


@app.cell
def cell_pp_annual_sharpe(portfolio):
    """PortfolioPlots.annual_sharpe_plot."""
    fig_annual_sharpe = portfolio.plots.annual_sharpe_plot()
    return (fig_annual_sharpe,)


@app.cell
def cell_pp_annual_sharpe_show(fig_annual_sharpe) -> None:
    """Display annual Sharpe figure."""
    fig_annual_sharpe
    return


@app.cell
def cell_pp_corr_heatmap_header() -> None:
    """PortfolioPlots — correlation heatmap header."""
    mo.md("### 7 · `correlation_heatmap` — asset + portfolio correlation matrix")
    return


@app.cell
def cell_pp_corr_heatmap(portfolio):
    """PortfolioPlots.correlation_heatmap."""
    fig_corr = portfolio.plots.correlation_heatmap()
    return (fig_corr,)


@app.cell
def cell_pp_corr_heatmap_show(fig_corr) -> None:
    """Display correlation heatmap figure."""
    fig_corr
    return


@app.cell
def cell_pp_monthly_heatmap_header() -> None:
    """PortfolioPlots — monthly heatmap header."""
    mo.md("### 8 · `monthly_returns_heatmap` — year × month returns heatmap")
    return


@app.cell
def cell_pp_monthly_heatmap(portfolio):
    """PortfolioPlots.monthly_returns_heatmap."""
    fig_pf_monthly_heatmap = portfolio.plots.monthly_returns_heatmap()
    return (fig_pf_monthly_heatmap,)


@app.cell
def cell_pp_monthly_heatmap_show(fig_pf_monthly_heatmap) -> None:
    """Display portfolio monthly heatmap figure."""
    fig_pf_monthly_heatmap
    return


@app.cell
def cell_pp_smoothed_header() -> None:
    """PortfolioPlots — smoothed holdings header."""
    mo.md("### 9 · `smoothed_holdings_performance_plot` — NAV with smoothed position entry")
    return


@app.cell
def cell_pp_smoothed(portfolio):
    """PortfolioPlots.smoothed_holdings_performance_plot."""
    fig_smoothed = portfolio.plots.smoothed_holdings_performance_plot(windows=[0, 1, 3, 5])
    return (fig_smoothed,)


@app.cell
def cell_pp_smoothed_show(fig_smoothed) -> None:
    """Display smoothed holdings performance figure."""
    fig_smoothed
    return


@app.cell
def cell_pp_cost_impact_header() -> None:
    """PortfolioPlots — trading cost impact header."""
    mo.md("### 10 · `trading_cost_impact_plot` — Sharpe vs. one-way trading costs")
    return


@app.cell
def cell_pp_cost_impact(portfolio):
    """PortfolioPlots.trading_cost_impact_plot."""
    fig_cost_impact = portfolio.plots.trading_cost_impact_plot(max_bps=20)
    return (fig_cost_impact,)


@app.cell
def cell_pp_cost_impact_show(fig_cost_impact) -> None:
    """Display trading cost impact figure."""
    fig_cost_impact
    return


# ── Portfolio Report ──────────────────────────────────────────────────────────


@app.cell
def cell_pf_report_header() -> None:
    """Portfolio Report section header."""
    mo.md("### `Report.to_html()` — self-contained portfolio HTML report")
    return


@app.cell
def cell_pf_report(portfolio):
    """Report.to_html — self-contained portfolio HTML report."""
    pf_html = portfolio.report.to_html(title="Synthetic 3-Asset Portfolio Report")
    mo.md(f"Portfolio HTML report generated: **{len(pf_html):,}** characters.")
    return (pf_html,)


@app.cell
def cell_pf_report_export(pf_html) -> None:
    """Write the Portfolio HTML report to NOTEBOOK_OUTPUT_FOLDER if set."""
    _output_folder = os.environ.get("NOTEBOOK_OUTPUT_FOLDER")
    if _output_folder:
        _artefact_path = Path(_output_folder) / "portfolio_report.html"
        _artefact_path.write_text(pf_html)
        _msg = mo.md(f"✅ Portfolio report saved to `{_artefact_path}`")
    else:
        _msg = mo.md(
            "ℹ️ `NOTEBOOK_OUTPUT_FOLDER` is not set — artefact saving is skipped "
            "(this variable is set automatically by `rhiza_marimo`)."
        )
    return


if __name__ == "__main__":
    app.run()
