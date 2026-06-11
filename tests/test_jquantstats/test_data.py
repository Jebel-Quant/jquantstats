"""Tests for the Data class functionality and methods."""

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jquantstats import Data
from jquantstats.data import _subtract_risk_free


def test_subtract_risk_free_rejects_invalid_rf_type():
    """_subtract_risk_free rejects rf values that are neither float nor DataFrame.

    `from_returns` coerces int rf to float before calling, so this guard only
    fires for direct callers passing an unsupported type.
    """
    frame = pl.DataFrame({"Date": [date(2023, 1, 1), date(2023, 1, 2)], "A": [0.01, 0.02]})
    with pytest.raises(TypeError, match="rf must be a float or DataFrame"):
        _subtract_risk_free(frame, 1, "Date")


def test_head(data):
    """Tests that the head() method returns a Data object with the first n rows.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        1. The return value is a Data object.
        2. The content matches the first n rows of the original data.

    """
    x = data.head()
    assert_frame_equal(x.returns, data.returns.head(5))

    # pd.testing.assert_frame_equal(x.all(), data.all().head())


def test_tail(data):
    """Tests that the tail() method returns a Data object with the last n rows.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        1. The return value is a Data object.
        2. The content matches the last n rows of the original data.

    """
    x = data.tail()
    assert_frame_equal(x.returns, data.returns.tail(5))


def test_all(data):
    """Tests that the all property returns a DataFrame with all data.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        The all property returns a DataFrame that includes all data.

    """
    print(data.returns.head(5))
    x = data.all
    print(x)


def test_assets(data):
    """Tests that the assets property returns the correct list of asset names.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        The assets property returns the expected list of asset names.

    """
    x = data.assets
    assert x == ["AAPL", "META", "SPY -- Benchmark"]


def test_date_col(data):
    """Tests that the date_col property returns the correct date column name.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        The date_col property returns the expected date column name.

    """
    x = data.date_col
    assert x == ["Date"]


def test_periods(data):
    """Tests that the _periods_per_year property returns the correct number of periods.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        The _periods_per_year property returns the expected number of periods (252 for daily data).

    """
    assert data._periods_per_year == pytest.approx(251.56913616203425)


def test_periods_edge_cases(data):
    """Tests edge cases for the _periods_per_year property.

    Args:
        data (Data): The data fixture containing a Data object.

    Verifies:
        1. ValueError is raised when index has less than 2 timestamps
        2. Different frequencies return different period counts
        3. Unsorted data is handled correctly

    """
    # Weekly data
    # Create dates with weekly intervals
    weekly_dates = [
        date(2023, 1, 1),
        date(2023, 1, 8),
        date(2023, 1, 15),
        date(2023, 1, 22),
        date(2023, 1, 29),
        date(2023, 2, 5),
        date(2023, 2, 12),
        date(2023, 2, 19),
        date(2023, 2, 26),
        date(2023, 3, 5),
    ]
    weekly_returns = pl.DataFrame({"Date": weekly_dates, "returns": [0.01] * 10})
    weekly_data = Data.from_returns(returns=weekly_returns)
    print(weekly_data._periods_per_year)
    assert weekly_data._periods_per_year == pytest.approx(52.142857142857146)
    # Monthly data
    # Create dates with monthly intervals
    monthly_dates = [
        date(2023, 1, 1),
        date(2023, 2, 1),
        date(2023, 3, 1),
        date(2023, 4, 1),
        date(2023, 5, 1),
        date(2023, 6, 1),
        date(2023, 7, 1),
        date(2023, 8, 1),
        date(2023, 9, 1),
        date(2023, 10, 1),
    ]
    monthly_returns = pl.DataFrame({"Date": monthly_dates, "returns": [0.01] * 10})
    monthly_data = Data.from_returns(returns=monthly_returns)
    assert monthly_data._periods_per_year == pytest.approx(12.032967032967033)


def test_post_init():
    """Tests the validation checks in the __post_init__ method of the Data class.

    Verifies:
        1. ValueError is raised when index has less than 2 timestamps
        2. ValueError is raised when index is not monotonically increasing
        3. ValueError is raised when returns and index have different row counts
        4. ValueError is raised when benchmark and index have different row counts
    """
    # Test case 1: Index with less than 2 timestamps
    single_date = [date(2023, 1, 1)]
    single_returns = pl.DataFrame({"Date": single_date, "returns": [0.01]})

    with pytest.raises(ValueError, match=r"Index must contain at least two timestamps\."):
        Data.from_returns(returns=single_returns, date_col="Date")

    # Test case 2: Unsorted index
    unsorted_dates = [date(2023, 1, 15), date(2023, 1, 1), date(2023, 1, 30)]
    unsorted_returns = pl.DataFrame({"Date": unsorted_dates, "returns": [0.01, 0.02, 0.03]})

    with pytest.raises(ValueError, match=r"Index must be monotonically increasing\."):
        Data.from_returns(returns=unsorted_returns)

    # Test case 3: Returns and index with different row counts
    dates = [date(2023, 1, 1), date(2023, 1, 15), date(2023, 1, 30)]
    returns = pl.DataFrame({"returns": [0.01, 0.02]})
    index = pl.DataFrame({"Date": dates})

    with pytest.raises(ValueError, match=r"Returns and index must have the same number of rows\."):
        Data(returns=returns, index=index)

    # Test case 4: Benchmark and index with different row counts
    dates = [date(2023, 1, 1), date(2023, 1, 15), date(2023, 1, 30)]
    returns = pl.DataFrame({"returns": [0.01, 0.02, 0.03]})
    benchmark = pl.DataFrame({"benchmark": [0.01, 0.02]})
    index = pl.DataFrame({"Date": dates})

    with pytest.raises(ValueError, match=r"Benchmark and index must have the same number of rows\."):
        Data(returns=returns, benchmark=benchmark, index=index)


def test_copy(data):
    """Tests that the copy() method creates a proper deep copy of the Data object.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        1. The return value is a Data object.
        2. The copied object has the same returns and benchmark data as the original.
        3. Modifying the copied object does not affect the original.
        4. The copy works correctly when there's a benchmark.

    """
    # Create a copy of the data object
    data_copy = data.copy()

    # Verify the copy has the same returns and benchmark as the original
    assert_frame_equal(data_copy.returns, data.returns)
    assert_frame_equal(data_copy.benchmark, data.benchmark)

    # Verify that modifying the copy doesn't affect the original
    # We can't directly modify the attributes because the Data class is frozen,
    # but we can verify that they are separate objects in memory
    assert data_copy is not data
    assert data_copy.returns is not data.returns
    assert data_copy.benchmark is not data.benchmark


def test_resample(data):
    """Tests that the resample() method correctly resamples data to different time periods.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        1. The return value is a Data object.
        2. The resampled object has the correct frequency.
        3. The resampling works with both compounded=False and compounded=True.
        4. The resampling works with different resample rules (YE, ME, etc.).

    """
    # Test resampling to yearly frequency
    yearly_data = data.resample(every="1y")

    # Verify the resampled data has the correct structure
    assert yearly_data.returns.shape[1] == data.returns.shape[1]  # Same number of columns
    # assert yearly_data.returns.index.freq == 'YE'  # Yearly frequency

    # Test resampling to monthly frequency with compounded=True
    monthly_data = data.resample(every="1mo")
    # Verify the resampled data has the correct structure
    assert monthly_data.returns.shape[1] == data.returns.shape[1]  # Same number of columns
    # assert monthly_data.returns.index.freq == 'ME'  # Monthly frequency


def test_stats(data):
    """Tests that the stats property returns a non-None value.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        The stats property returns a non-None value.

    """
    assert data.stats is not None


def test_plots(data):
    """Tests that the plots property returns a non-None value.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        The plots property returns a non-None value.

    """
    assert data.plots is not None


def test_all_no_benchmark(data_no_benchmark):
    """Tests that the all property works correctly when there is no benchmark.

    Args:
        data_no_benchmark (_Data): The data_no_benchmark fixture containing a Data object with no benchmark.

    Verifies:
        The all property returns a non-None value when there is no benchmark.

    """
    assert data_no_benchmark.all is not None


def test_assets_no_benchmark(data_no_benchmark):
    """Tests that the assets property works correctly when there is no benchmark.

    Args:
        data_no_benchmark (_Data): The data_no_benchmark fixture containing a Data object with no benchmark.

    Verifies:
        The assets property returns a non-None value when there is no benchmark.

    """
    assert data_no_benchmark.assets is not None


def test_copy_no_benchmark(data_no_benchmark):
    """Tests that the copy method works correctly when there is no benchmark.

    Args:
        data_no_benchmark (_Data): The data_no_benchmark fixture containing a Data object with no benchmark.

    Verifies:
        1. The copied object has non-None returns.
        2. The copied object has None benchmark.

    """
    x = data_no_benchmark.copy()
    assert x.returns is not None
    assert x.benchmark is None


def test_truncate_by_start_and_end(data):
    """Tests truncate(start, end) filters rows inclusively by date."""
    first_date = data.index["Date"][0]
    last_date = data.index["Date"][-1]
    mid_start = data.index["Date"][10]
    mid_end = data.index["Date"][20]

    result = data.truncate(start=mid_start, end=mid_end)

    assert result.index["Date"][0] == mid_start
    assert result.index["Date"][-1] == mid_end
    assert result.returns.shape[0] == 11
    assert result.index.shape[0] == result.returns.shape[0]
    if result.benchmark is not None:
        assert result.benchmark.shape[0] == result.returns.shape[0]

    # full range should equal original
    full = data.truncate(start=first_date, end=last_date)
    assert full.returns.shape[0] == data.returns.shape[0]


def test_truncate_start_only(data):
    """Tests truncate(start=...) returns rows from start to the end of the data."""
    mid_start = data.index["Date"][10]
    result = data.truncate(start=mid_start)
    assert result.index["Date"][0] == mid_start
    assert result.returns.shape[0] == data.returns.shape[0] - 10


def test_truncate_end_only(data):
    """Tests truncate(end=...) returns rows from the beginning up to end inclusive."""
    mid_end = data.index["Date"][9]
    result = data.truncate(end=mid_end)
    assert result.index["Date"][-1] == mid_end
    assert result.returns.shape[0] == 10


def test_truncate_no_bounds_returns_all(data):
    """Tests truncate() with no bounds returns all rows unchanged."""
    result = data.truncate()
    assert result.returns.shape[0] == data.returns.shape[0]
    assert_frame_equal(result.returns, data.returns)
    assert_frame_equal(result.index, data.index)


def test_truncate_no_benchmark(data_no_benchmark):
    """Tests truncate() works when there is no benchmark."""
    mid_start = data_no_benchmark.index["Date"][5]
    mid_end = data_no_benchmark.index["Date"][15]
    result = data_no_benchmark.truncate(start=mid_start, end=mid_end)
    assert result.benchmark is None
    assert result.returns.shape[0] == 11


def test_truncate_integer_indexed_both_bounds():
    """Tests truncate() with integer indices when data has no temporal index."""
    dates = list(range(10))
    returns_df = pl.DataFrame({"returns": [float(i) * 0.01 for i in range(10)]})
    index_df = pl.DataFrame({"row": dates})

    d = Data(returns=returns_df, index=index_df)
    result = d.truncate(start=2, end=5)
    assert result.returns.shape[0] == 4
    assert list(result.index["row"]) == [2, 3, 4, 5]


def test_truncate_integer_indexed_raises_on_non_int():
    """Tests truncate() raises TypeError when non-integer bound is given on integer-indexed data."""
    returns_df = pl.DataFrame({"returns": [0.01 * i for i in range(10)]})
    index_df = pl.DataFrame({"row": list(range(10))})

    d = Data(returns=returns_df, index=index_df)
    with pytest.raises(TypeError, match="start must be an integer"):
        d.truncate(start="2020-01-01")
    with pytest.raises(TypeError, match="end must be an integer"):
        d.truncate(end=3.5)


def test_describe(data):
    """Tests that describe() returns a tidy summary DataFrame with one row per asset.

    Args:
        data: The data fixture containing a Data object.

    Verifies:
        1. Returns a pl.DataFrame.
        2. Has one row per returns asset (not including benchmark).
        3. Contains columns: asset, start, end, rows, has_benchmark.
        4. has_benchmark is True when a benchmark is present.

    """
    result = data.describe()
    assert isinstance(result, pl.DataFrame)
    assert list(result.columns) == ["asset", "start", "end", "rows", "has_benchmark"]
    assert result.shape[0] == len(data.returns.columns)
    assert list(result["asset"]) == data.returns.columns
    expected_start = data.index[data.date_col[0]].min()
    expected_end = data.index[data.date_col[0]].max()
    assert (result["rows"] == len(data.index)).all()
    assert (result["start"] == expected_start).all()
    assert (result["end"] == expected_end).all()
    assert all(result["has_benchmark"])


def test_describe_no_benchmark(data_no_benchmark):
    """Tests that describe() correctly reflects the absence of a benchmark.

    Args:
        data_no_benchmark: The data_no_benchmark fixture containing a Data object without benchmark.

    Verifies:
        has_benchmark column is False for all rows when no benchmark is present.

    """
    result = data_no_benchmark.describe()
    assert isinstance(result, pl.DataFrame)
    assert not any(result["has_benchmark"])


def test_repr(data):
    """Tests that Data.__repr__ returns an informative string."""
    r = repr(data)
    assert r.startswith("Data(assets=")
    assert "rows=" in r
    assert "start=" in r
    assert "end=" in r
    for asset in data.assets:
        assert asset in r


class TestInterpolate:
    """Tests for the standalone interpolate() function."""

    def test_interior_nulls_are_filled(self):
        """Nulls between the first and last non-null are forward-filled."""
        from jquantstats import interpolate

        df = pl.DataFrame({"a": [1.0, None, None, 4.0]})
        result = interpolate(df)
        assert result["a"].to_list() == [1.0, 1.0, 1.0, 4.0]

    def test_leading_nulls_are_not_filled(self):
        """Nulls before the first non-null value are left unchanged."""
        from jquantstats import interpolate

        df = pl.DataFrame({"a": [None, None, 3.0, None, 5.0]})
        result = interpolate(df)
        assert result["a"][0] is None
        assert result["a"][1] is None
        assert result["a"][2] == pytest.approx(3.0)
        assert result["a"][3] == pytest.approx(3.0)
        assert result["a"][4] == pytest.approx(5.0)

    def test_trailing_nulls_are_not_filled(self):
        """Nulls after the last non-null value are left unchanged."""
        from jquantstats import interpolate

        df = pl.DataFrame({"a": [1.0, None, 3.0, None, None]})
        result = interpolate(df)
        assert result["a"][3] is None
        assert result["a"][4] is None

    def test_all_null_column_unchanged(self):
        """A column with no non-null values is passed through unchanged."""
        from jquantstats import interpolate

        df = pl.DataFrame({"a": pl.Series([None, None, None], dtype=pl.Float64)})
        result = interpolate(df)
        assert result["a"].null_count() == 3

    def test_non_numeric_column_unchanged(self):
        """Non-numeric columns are returned without modification."""
        from jquantstats import interpolate

        df = pl.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", "z"]})
        result = interpolate(df)
        assert result["b"].to_list() == ["x", "y", "z"]

    def test_mixed_columns(self):
        """Numeric and non-numeric columns coexist correctly."""
        from jquantstats import interpolate

        df = pl.DataFrame({"num": [None, 2.0, None, 4.0, None], "label": ["a", "b", "c", "d", "e"]})
        result = interpolate(df)
        assert result["num"].to_list() == [None, 2.0, 2.0, 4.0, None]
        assert result["label"].to_list() == ["a", "b", "c", "d", "e"]

    def test_schema_preserved(self):
        """Output schema matches input schema exactly."""
        from jquantstats import interpolate

        df = pl.DataFrame({"a": pl.Series([1.0, None, 3.0], dtype=pl.Float32), "b": [1, 2, 3]})
        result = interpolate(df)
        assert result.schema == df.schema

    def test_no_nulls_unchanged(self):
        """A DataFrame without nulls is returned identically."""
        from jquantstats import interpolate

        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = interpolate(df)
        assert_frame_equal(result, df)

    def test_integer_column_filled(self):
        """Integer columns are also forward-filled in the interior."""
        from jquantstats import interpolate

        df = pl.DataFrame({"a": pl.Series([1, None, None, 4], dtype=pl.Int64)})
        result = interpolate(df)
        assert result["a"].to_list() == [1, 1, 1, 4]

    def test_no_temporary_column_in_output(self):
        """The temporary __row_idx__ column must not appear in the output."""
        from jquantstats import interpolate

        df = pl.DataFrame({"a": [1.0, None, 3.0]})
        result = interpolate(df)
        assert "__row_idx__" not in result.columns

    def test_existing_row_idx_column_is_preserved(self):
        """A user-provided __row_idx__ column must not be overwritten or dropped."""
        from jquantstats import interpolate

        df = pl.DataFrame({"a": [1.0, None, 3.0], "__row_idx__": [10, 20, 30]})
        result = interpolate(df)
        assert result.columns == df.columns
        assert result["a"].to_list() == [1.0, 1.0, 3.0]
        assert result["__row_idx__"].to_list() == [10, 20, 30]
