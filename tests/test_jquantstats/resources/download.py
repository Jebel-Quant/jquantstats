# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars-yahoo==0.2.0",
# ]
# ///
"""Download some data needed for the tests.

Self-contained (PEP 723): run with ``uv run download.py`` from this directory.
Fetches Yahoo's split- and dividend-adjusted closes with polars-yahoo and writes
daily percentage returns to ``meta.csv``, ``benchmark.csv`` and ``portfolio.csv``.
"""

import polars as pl
import polars_yahoo as yh


def download_returns(columns: dict[str, str]) -> pl.DataFrame:
    """Download the full history of daily returns for some tickers.

    Args:
        columns: Maps each Yahoo ticker to the name of its column in the result.

    Returns:
        A frame with a ``Date`` column and one return column per ticker, on the
        union of the tickers' trading days. The first return of each ticker is null.

    """
    # An explicit early start rather than period="max": for "max" Yahoo downgrades
    # daily bars to monthly ones.
    prices = yh.download(list(columns), start="1900-01-01", raise_errors=True)
    wide = prices.pivot(on="ticker", index="date", values="adj_close").sort("date")
    return wide.select(
        pl.col("date").alias("Date"),
        *(pl.col(ticker).pct_change().alias(name) for ticker, name in columns.items()),
    )


if __name__ == "__main__":
    download_returns({"META": "Meta"}).write_csv("meta.csv")
    download_returns({"SPY": "SPY -- Benchmark"}).write_csv("benchmark.csv")
    download_returns({"AAPL": "AAPL", "META": "META"}).write_csv("portfolio.csv")
