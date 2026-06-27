"""Fixtures for tests.

Security note: Test code uses pytest assertions (S101), which are intentional
and safe in the test context. No subprocess calls (S603/S607) are used here.
"""

import warnings
from pathlib import Path

import polars as pl
import pytest

from jquantstats import Data
from jquantstats.exceptions import BenchmarkAlignmentWarning


@pytest.fixture(scope="session")
def resource_dir():
    """Fixture that provides the path to the test resources directory."""
    return Path(__file__).parent / "resources"


@pytest.fixture
def project_root():
    """Fixture that provides the project root directory.

    Returns:
        Path: The path to the project root directory.

    """
    return Path(__file__).parent.parent.parent


@pytest.fixture
def env_content(project_root: Path):
    """Fixture that provides the content of the .env file as a dictionary.

    Returns:
        dict: A dictionary containing the key-value pairs from the .env file.

    """
    # Get the project root directory
    env_file_path = project_root / ".env"

    from dotenv import dotenv_values

    return dotenv_values(env_file_path)


@pytest.fixture
def returns(resource_dir) -> pl.DataFrame:
    """Fixture that returns a DataFrame with Meta returns.

    Args:
        resource_dir: The resource_dir fixture containing the path to test resources.

    Returns:
        pl.DataFrame: A DataFrame containing Date and Meta returns.

    """
    # Only feed in frames. No series.
    dframe = pl.read_csv(resource_dir / "meta.csv", try_parse_dates=True)
    return dframe.select(["Date", "Meta"])


@pytest.fixture
def benchmark_frame(resource_dir) -> pl.DataFrame:
    """Fixture that returns a DataFrame with benchmark returns.

    Args:
        resource_dir: The resource_dir fixture containing the path to test resources.

    Returns:
        pl.DataFrame: A DataFrame containing Date and SPY benchmark returns.

    """
    dframe = pl.read_csv(resource_dir / "benchmark.csv", try_parse_dates=True)
    return dframe.select(["Date", "SPY -- Benchmark"])


@pytest.fixture
def portfolio(resource_dir):
    """Fixture that returns a DataFrame with portfolio returns.

    Args:
        resource_dir: The resource_dir fixture containing the path to test resources.

    Returns:
        pl.DataFrame: A DataFrame containing Date, AAPL, and META returns.

    """
    # that's interesting, polars thinks META is a str type
    return pl.read_csv(resource_dir / "portfolio.csv", try_parse_dates=True).with_columns(
        [
            pl.col("AAPL").cast(pl.Float64, strict=False),
            pl.col("META").cast(pl.Float64, strict=False),
            pl.col("Date").cast(pl.Date, strict=False),
        ]
    )


@pytest.fixture
def data(portfolio, benchmark_frame):
    """Fixture that returns a Data object with portfolio and benchmark data.

    Args:
        portfolio: The portfolio fixture containing portfolio returns.
        benchmark_frame: The benchmark fixture containing benchmark returns.

    Returns:
        Data: A Data object containing portfolio returns and benchmark data.

    """
    # portfolio.csv and benchmark.csv cover overlapping but not identical date
    # ranges, so alignment legitimately drops rows. That is incidental to the
    # many tests that consume this fixture, so suppress the warning at the
    # source rather than letting it flood the suite output.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BenchmarkAlignmentWarning)
        return Data.from_returns(returns=portfolio, benchmark=benchmark_frame)


@pytest.fixture
def data_no_benchmark(portfolio):
    """Fixture that returns a Data object with portfolio data but no benchmark.

    Args:
        portfolio: The portfolio fixture containing portfolio returns.

    Returns:
        Data: A Data object containing portfolio returns without benchmark data.

    """
    return Data.from_returns(returns=portfolio)


@pytest.fixture
def edge(data):
    """Fixture that returns a Data object with edge case data (all zeros).

    This fixture creates a Data object with returns and benchmark data that are all zeros,
    which is useful for testing edge cases in statistical calculations.

    Args:
        data: The data fixture containing a Data object.

    Returns:
        Data: A Data object with returns and benchmark data that are all zeros.

    """
    index = data.index["Date"]
    returns = pl.DataFrame({"index": index, "returns": [0.0] * len(index)})
    benchmark = pl.DataFrame({"index": index, "benchmark": [0.0] * len(index)})

    return Data.from_returns(returns=returns, benchmark=benchmark, date_col="index")


@pytest.fixture
def readme_path() -> Path:
    """Provide the path to the project's README.md file.

    This fixture searches for the README.md file by starting in the current
    directory and moving up through parent directories until it finds the file.

    Returns:
    -------
    Path
        Path to the README.md file

    Raises:
    ------
    FileNotFoundError
        If the README.md file cannot be found in any parent directory

    """
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        candidate = current_dir / "README.md"
        if candidate.is_file():
            return candidate
        current_dir = current_dir.parent
    raise FileNotFoundError("README.md not found in any parent directory")  # noqa: TRY003
