"""#929: Data and Portfolio agree on the name of the date column.

``Data.from_returns`` used to default to ``date_col="Date"`` while every
``Portfolio`` internal canonicalised on lowercase ``'date'``, so two objects of
the same class carried a different index column name depending on how they were
built. The date column is now auto-detected and a temporal axis always ends up
as ``'date'``.
"""

import polars as pl
import pytest

from jquantstats import Data, Portfolio
from jquantstats.exceptions import MissingDateColumnError

N = 6


@pytest.fixture
def dates() -> pl.Series:
    """Six consecutive days."""
    return pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 6), interval="1d", eager=True)


@pytest.fixture
def returns(dates: pl.Series) -> pl.DataFrame:
    """A returns frame whose date column is labelled ``'Date'``."""
    return pl.DataFrame({"Date": dates, "Fund": [0.01, -0.02, 0.03, 0.0, 0.01, -0.01]})


@pytest.fixture
def prices(dates: pl.Series) -> pl.DataFrame:
    """A price frame whose date column is labelled ``'Date'``."""
    return pl.DataFrame({"Date": dates, "A": [100.0, 101.0, 99.0, 99.5, 102.0, 101.0]})


def test_from_returns_auto_detects_and_canonicalises(returns: pl.DataFrame) -> None:
    """A 'Date' column is found without being named and reported as 'date'."""
    data = Data.from_returns(returns)
    assert data.index.columns == ["date"]
    assert data.date_col == ["date"]
    assert data.index["date"].to_list() == returns["Date"].to_list()


def test_from_returns_explicit_date_col_is_canonicalised(returns: pl.DataFrame) -> None:
    """Naming the column explicitly picks the same column and the same label."""
    assert Data.from_returns(returns, date_col="Date").index.columns == ["date"]


def test_from_prices_auto_detects_and_canonicalises(prices: pl.DataFrame) -> None:
    """from_prices resolves its date column the same way from_returns does."""
    assert Data.from_prices(prices).index.columns == ["date"]


def test_lowercase_input_is_accepted(returns: pl.DataFrame) -> None:
    """A frame already using 'date' is passed through untouched."""
    lowercase = returns.rename({"Date": "date"})
    assert Data.from_returns(lowercase).index.columns == ["date"]


def test_data_and_portfolio_data_agree(returns: pl.DataFrame, prices: pl.DataFrame) -> None:
    """The issue's reproduction: one reader works against both objects."""
    data = Data.from_returns(returns)
    portfolio = Portfolio.from_cash_position(
        prices=prices,
        cash_position=prices.with_columns(pl.lit(1e5).alias("A")),
        aum=1e6,
    )

    def first_date(obj: Data) -> object:
        """Read the earliest date through the canonical 'date' column."""
        return obj.index["date"].min()

    assert data.date_col == portfolio.data.date_col == ["date"]
    assert first_date(data) == first_date(portfolio.data)


def test_frames_may_label_their_date_columns_differently(returns: pl.DataFrame, dates: pl.Series) -> None:
    """Auto-detection runs per frame, so returns/benchmark/rf need not agree."""
    benchmark = pl.DataFrame({"timestamp": dates, "Market": [0.005] * N})
    rf = pl.DataFrame({"when": dates, "rf": [0.0001] * N})

    data = Data.from_returns(returns, benchmark=benchmark, rf=rf)

    assert data.index.columns == ["date"]
    assert data.benchmark is not None
    assert data.benchmark.columns == ["Market"]


def test_from_prices_rf_frame_is_canonicalised(prices: pl.DataFrame, dates: pl.Series) -> None:
    """A frame-valued rf reaching from_prices resolves like the price frame."""
    rf = pl.DataFrame({"Date": dates[1:], "rf": [0.0001] * (N - 1)})
    assert Data.from_prices(prices, rf=rf).index.columns == ["date"]


def test_direct_construction_normalises_the_index(returns: pl.DataFrame) -> None:
    """Building a Data by hand carries the same guarantee as the constructors."""
    data = Data(returns=returns.drop("Date"), index=returns.select("Date"))
    assert data.index.columns == ["date"]


def test_no_temporal_column_raises_and_names_the_frame(returns: pl.DataFrame) -> None:
    """Auto-detection failure says what it looked for and how to override it."""
    integer_indexed = returns.with_columns(pl.int_range(0, N).alias("Date"))
    with pytest.raises(MissingDateColumnError, match="'returns' has no temporal column") as exc_info:
        Data.from_returns(integer_indexed)
    assert exc_info.value.frame_name == "returns"
    assert exc_info.value.column is None
    assert "Date" in exc_info.value.available


def test_non_temporal_date_col_keeps_its_name(returns: pl.DataFrame) -> None:
    """An integer axis nominated by name is not relabelled 'date' — it is not one."""
    integer_indexed = returns.with_columns(pl.int_range(0, N).alias("i"))
    data = Data.from_returns(integer_indexed.drop("Date"), date_col="i")
    assert data.date_col == ["i"]
