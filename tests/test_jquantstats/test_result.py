"""Tests for the Result container and create_reports method."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from jquantstats import Portfolio
from jquantstats.result import Result


@pytest.fixture
def simple_portfolio() -> Portfolio:
    """20-day single-asset Portfolio for Result tests."""
    n = 20
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    return Portfolio.from_cash_position(
        prices=pl.DataFrame({"date": dates, "A": pl.Series([100.0 * (1.005**i) for i in range(n)])}),
        cash_position=pl.DataFrame({"date": dates, "A": pl.Series([1000.0] * n, dtype=pl.Float64)}),
        aum=1e5,
    )


def test_result_create_reports_creates_directories(tmp_path, simple_portfolio):
    """create_reports must create data/ and plots/ subdirectories."""
    result = Result(portfolio=simple_portfolio)
    result.create_reports(output_dir=tmp_path)
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "plots").is_dir()


def test_result_create_reports_csv_files_exist(tmp_path, simple_portfolio):
    """create_reports must write prices.csv, profit.csv, returns.csv, tilt_timing_decomp.csv, position.csv."""
    result = Result(portfolio=simple_portfolio)
    result.create_reports(output_dir=tmp_path)
    data_dir = tmp_path / "data"
    for name in ("prices.csv", "profit.csv", "returns.csv", "tilt_timing_decomp.csv", "position.csv"):
        assert (data_dir / name).exists(), f"Missing {name}"


def test_result_create_reports_html_files_exist(tmp_path, simple_portfolio):
    """create_reports must write snapshot.html, lag_ir.html, lagged_perf.html, smooth_perf.html."""
    result = Result(portfolio=simple_portfolio)
    result.create_reports(output_dir=tmp_path)
    plots_dir = tmp_path / "plots"
    for name in ("snapshot.html", "lag_ir.html", "lagged_perf.html", "smooth_perf.html"):
        assert (plots_dir / name).exists(), f"Missing {name}"


def test_result_create_reports_with_mu(tmp_path, simple_portfolio):
    """When mu is provided, create_reports must also write signal.csv."""
    n = 20
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    mu = pl.DataFrame({"date": dates, "A": pl.Series([0.001] * n, dtype=pl.Float64)})
    result = Result(portfolio=simple_portfolio, mu=mu)
    result.create_reports(output_dir=tmp_path)
    assert (tmp_path / "data" / "signal.csv").exists()


def test_result_create_reports_without_mu_no_signal_csv(tmp_path, simple_portfolio):
    """When mu is None, create_reports must NOT write signal.csv."""
    result = Result(portfolio=simple_portfolio, mu=None)
    result.create_reports(output_dir=tmp_path)
    assert not (tmp_path / "data" / "signal.csv").exists()


def test_result_rejects_non_dataframe_mu(simple_portfolio):
    """Result must raise TypeError when mu is not a polars DataFrame."""
    with pytest.raises(TypeError, match="mu must be a polars DataFrame"):
        Result(portfolio=simple_portfolio, mu={"A": [0.001]})


def test_result_rejects_mu_missing_asset_columns(simple_portfolio):
    """Result must raise MuSchemaError naming the asset columns mu lacks."""
    from jquantstats.exceptions import MuSchemaError

    mu = pl.DataFrame({"OTHER": [0.001] * 20})
    with pytest.raises(MuSchemaError, match=r"missing expected-return columns.*'A'"):
        Result(portfolio=simple_portfolio, mu=mu)
