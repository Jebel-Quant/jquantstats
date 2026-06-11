"""Tests for the Data.from_prices classmethod."""

import polars as pl
import pytest

from jquantstats import Data

from .tolerances import TOL_COMPOUNDING


@pytest.fixture
def prices() -> pl.DataFrame:
    """Simple price DataFrame with a date column and two assets."""
    return pl.DataFrame(
        {
            "Date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 5), interval="1d", eager=True),
            "Asset1": [100.0, 102.0, 101.0, 103.0, 105.0],
            "Asset2": [50.0, 51.0, 50.5, 52.0, 53.0],
        }
    )


def test_from_prices_returns_data(prices: pl.DataFrame) -> None:
    """Data.from_prices returns a Data object."""
    data = Data.from_prices(prices=prices)
    assert isinstance(data, Data)


def test_from_prices_drops_first_row(prices: pl.DataFrame) -> None:
    """Prices has n rows; the derived Data object has n-1 return rows."""
    data = Data.from_prices(prices=prices)
    assert data.returns.shape[0] == prices.shape[0] - 1


def test_from_prices_correct_returns(prices: pl.DataFrame) -> None:
    """Returns computed from prices match manual pct_change calculation."""
    data = Data.from_prices(prices=prices)
    expected_asset1 = prices["Asset1"].pct_change().drop_nulls()
    assert data.returns["Asset1"].to_list() == pytest.approx(expected_asset1.to_list(), rel=TOL_COMPOUNDING)


def test_from_prices_max_drawdown(prices: pl.DataFrame) -> None:
    """Data.from_prices works end-to-end with stats.max_drawdown."""
    data = Data.from_prices(prices=prices)
    max_dd = data.stats.max_drawdown()
    assert isinstance(max_dd, dict)
    assert "Asset1" in max_dd
    assert float(data.stats.max_drawdown()["Asset1"]) <= 0.0


def test_from_prices_issue_example() -> None:
    """Reproduces the exact usage pattern from the issue."""
    prices = pl.DataFrame(
        {
            "Date": pl.date_range(pl.date(2022, 1, 1), pl.date(2022, 12, 31), interval="1d", eager=True),
            "Asset": [100.0 + i * 0.1 for i in range(365)],
        }
    )
    data = Data.from_prices(prices=prices)
    result = float(data.stats.max_drawdown()["Asset"])
    assert result >= 0.0


def test_from_prices_with_benchmark(prices: pl.DataFrame) -> None:
    """Data.from_prices accepts a benchmark price DataFrame."""
    benchmark_prices = pl.DataFrame(
        {
            "Date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 5), interval="1d", eager=True),
            "Benchmark": [200.0, 202.0, 201.0, 203.0, 205.0],
        }
    )
    data = Data.from_prices(prices=prices, benchmark=benchmark_prices)
    assert data.benchmark is not None


def test_from_prices_missing_date_col_raises(prices):
    """Data.from_prices raises MissingDateColumnError when date_col is absent from prices."""
    from jquantstats.exceptions import MissingDateColumnError

    with pytest.raises(MissingDateColumnError, match="'prices' has no column 'date'") as exc_info:
        Data.from_prices(prices=prices, date_col="date")
    assert exc_info.value.frame_name == "prices"
    assert "Date" in (exc_info.value.available or [])


def test_from_prices_missing_date_col_in_benchmark_raises(prices):
    """Data.from_prices raises MissingDateColumnError when date_col is absent from benchmark."""
    from jquantstats.exceptions import MissingDateColumnError

    benchmark = prices.select(pl.col("Date").alias("timestamp"), pl.col("Asset1").alias("Market"))
    with pytest.raises(MissingDateColumnError, match="'benchmark' has no column 'Date'"):
        Data.from_prices(prices=prices, benchmark=benchmark, date_col="Date")
