r"""Regenerate the chart gallery under ``docs/shots/``.

Every image on the documentation gallery page comes out of this script, so the
figures can be refreshed whenever the plotting code changes:

.. code-block:: bash

    uv run --with pillow python book/shots/generate.py

The subject is a 20/100-day moving-average crossover on the AAPL/META price
fixtures in ``tests/test_jquantstats/resources`` — long enough (1980-2025) that
the long-history charts have something to show, and position-based so the
``Portfolio``-only diagnostics (lag sweeps, turnover cost, attribution) are
available too.

Three images on the gallery page are screenshots of the ``reports.full()``
HTML document rather than Plotly exports.  This script writes that document to
``output/report.html`` (gitignored — it is a couple of megabytes); capture it
with headless Chrome and crop:

.. code-block:: bash

    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \\
        --headless=new --disable-gpu --hide-scrollbars \\
        --force-device-scale-factor=2 --virtual-time-budget=60000 \\
        --window-size=1400,5000 --screenshot=full.png \\
        "file://$PWD/output/report.html"

The crop offsets used for ``report-metrics``, ``report-drawdowns`` and
``report-charts`` are listed in ``REPORT_CROPS`` below (CSS pixels, so double
them against a 2x capture).
"""

from __future__ import annotations

import math
import pathlib

import plotly.graph_objects as go
import polars as pl
from PIL import Image

from jquantstats._cost_model import CostModel
from jquantstats.data import Data
from jquantstats.portfolio import Portfolio

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESOURCES = ROOT / "tests" / "test_jquantstats" / "resources"
DEST = ROOT / "docs" / "shots"

AUM = 1_000_000.0
TARGET_VOL = 0.15
"""Annualised volatility the risk positions are scaled to."""

WIDTH = 1900
"""Delivered image width in pixels; figures are exported at 2x and downscaled."""

QUALITY = 82
"""WebP quality. Verified against the densest chart (the 45-row monthly
heatmap, which carries several hundred small percentage labels) — its text is
still artefact-free here, and the whole gallery fits in ~1.5 MB."""

REPORT_CROPS = {
    "report-metrics": (0, 1790),
    "report-drawdowns": (2530, 3280),
    "report-charts": (3280, 4640),
}
"""Vertical crop bounds (CSS px) into a 1400px-wide render of ``report.html``.

The charts only appear if the render has network access — ``reports.full()``
pulls plotly.js from the CDN rather than inlining it.
"""


def _read(path: pathlib.Path, date_col: str) -> pl.DataFrame:
    """Load a fixture CSV, normalising the date column name and numeric dtypes.

    Args:
        path: CSV file to read.
        date_col: Name of the date column in the file, renamed to ``date``.

    Returns:
        A frame with a ``date`` column and Float64 asset columns.
    """
    frame = pl.read_csv(path, try_parse_dates=True, null_values=["", "NA", "nan"])
    frame = frame.rename({date_col: "date"})
    return frame.with_columns(pl.col(c).cast(pl.Float64) for c in frame.columns if c != "date")


def _crossover(prices: pl.DataFrame, fast: int, slow: int) -> pl.DataFrame:
    """Build risk positions from a moving-average crossover on *prices*.

    The sign of the fast-minus-slow spread sets the direction; the magnitude is
    the per-asset share of the daily cash-vol budget, which is the unit
    ``Portfolio.from_risk_position`` expects (it divides by an EWMA vol
    estimate to reach a cash position).

    Args:
        prices: Price levels with a ``date`` column.
        fast: Fast moving-average window in days.
        slow: Slow moving-average window in days.

    Returns:
        A risk-position frame aligned with *prices*.
    """
    assets = [c for c in prices.columns if c != "date"]
    budget = AUM * TARGET_VOL / math.sqrt(252) / len(assets)
    return prices.with_columns(
        ((pl.col(a).rolling_mean(fast) - pl.col(a).rolling_mean(slow)).sign() * budget).fill_null(0.0).alias(a)
        for a in assets
    )


def _save(fig: go.Figure, name: str, height: int | None = None) -> None:
    """Export a Plotly figure to ``docs/shots/<name>.webp``.

    Rendered at 2x the delivered width and downscaled, which keeps text crisp
    without shipping multi-megabyte PNGs in the repository.

    Args:
        fig: The figure to export.
        name: Output stem, without extension.
        height: Optional height override for figures whose default is too
            short for the amount of data (the 45-row monthly heatmap).
    """
    if height is not None:
        fig.update_layout(height=height)
    px_h = fig.layout.height or 600
    px_w = fig.layout.width or 1600
    png = DEST / f"{name}.png"
    fig.write_image(str(png), width=px_w, height=px_h, scale=2)

    image = Image.open(png).convert("RGB")
    width = min(WIDTH, image.size[0])
    scaled = image.resize((width, round(image.size[1] * width / image.size[0])), Image.LANCZOS)
    scaled.save(DEST / f"{name}.webp", "WEBP", quality=QUALITY, method=6)
    png.unlink()
    print(f"{name:32s} {scaled.size[0]}x{scaled.size[1]}")


def main() -> None:
    """Render every gallery image and the HTML report."""
    DEST.mkdir(parents=True, exist_ok=True)

    prices = _read(RESOURCES / "prices.csv", "date")
    stocks = _read(RESOURCES / "stock_prices.csv", "date")
    benchmark = _read(RESOURCES / "benchmark.csv", "Date")
    returns = _read(RESOURCES / "portfolio.csv", "Date").fill_null(0.0)

    # ── Portfolio route: the trend follower, with a turnover cost model ──────
    pf = Portfolio.from_risk_position(
        prices=prices,
        risk_position=_crossover(prices, 20, 100),
        aum=AUM,
        vola=32,
        cost_model=CostModel.turnover_bps(5.0),
    )
    _save(pf.plots.snapshot(), "portfolio-snapshot")
    _save(pf.plots.lead_lag_ir_plot(), "portfolio-lead-lag-ir")
    _save(pf.plots.lagged_performance_plot(), "portfolio-lagged-performance")
    _save(pf.plots.trading_cost_impact_plot(max_bps=20), "portfolio-trading-cost-impact")
    _save(pf.plots.rolling_sharpe_plot(window=252), "portfolio-rolling-sharpe")
    _save(pf.plots.annual_sharpe_plot(), "portfolio-annual-sharpe")
    _save(pf.plots.smoothed_holdings_performance_plot(), "portfolio-smoothed-holdings")

    # 45 calendar years of rows do not fit the 600px default.
    heatmap = pf.plots.monthly_returns_heatmap()
    heatmap.update_layout(title="Monthly Returns Heatmap — trend portfolio, 1981-2025")
    _save(heatmap, "portfolio-monthly-heatmap", height=1500)

    # A second portfolio over 20 large caps, so the correlation matrix is worth
    # plotting — the two-asset book above would render a 3x3 grid.
    pf20 = Portfolio.from_risk_position(
        prices=stocks,
        risk_position=_crossover(stocks, 10, 50),
        aum=AUM,
        vola=20,
    )
    _save(pf20.plots.correlation_heatmap(name="trend portfolio"), "portfolio-correlation-heatmap")

    # ── Data route ──────────────────────────────────────────────────────────
    # META is padded flat before its 2012 listing, so the long-history charts
    # use AAPL alone and the two-asset charts start at the listing date.
    single = Data.from_returns(returns=returns.select(["date", "AAPL"]), benchmark=benchmark, date_col="date")
    listing = returns.filter(pl.col("META") != 0.0).select(pl.col("date").min()).item()
    duo = Data.from_returns(returns=returns.filter(pl.col("date") >= listing), benchmark=benchmark, date_col="date")

    _save(single.plots.snapshot(title="AAPL vs SPY — 1993 to 2025", log_scale=True), "data-snapshot")
    _save(
        single.plots.drawdowns_periods(n=5, asset="AAPL", title="AAPL — Five Deepest Drawdowns"),
        "data-drawdown-periods",
    )
    _save(
        single.plots.monthly_heatmap(asset="AAPL", title="AAPL — Monthly Returns, 33 Years"),
        "data-monthly-heatmap",
    )
    _save(single.plots.yearly_returns(), "data-yearly-returns")
    _save(single.plots.histogram(bins=120), "data-histogram")
    _save(single.plots.montecarlo_distribution(n=2000, period=252), "data-montecarlo-distribution")

    _save(duo.plots.compare(title="AAPL & META vs SPY (since META's IPO)"), "data-compare")
    _save(duo.plots.distribution(), "data-distribution")
    _save(duo.plots.montecarlo(n=300, period=252), "data-montecarlo")
    _save(duo.plots.rolling_sharpe(), "data-rolling-sharpe")
    _save(duo.plots.rolling_volatility(), "data-rolling-volatility")
    _save(duo.plots.rolling_beta(rolling_period=126, rolling_period2=504), "data-rolling-beta")

    # ── The self-contained HTML report (screenshot separately, see module docs)
    report = ROOT / "output" / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(duo.reports.full(title="jquantstats — Performance Report"))
    print(f"{'report.html':32s} {report.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
