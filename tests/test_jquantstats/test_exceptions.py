"""Focused tests for exception payload behavior."""

from jquantstats.exceptions import MissingDateColumnError


def test_missing_date_column_error_available_defaults_to_empty_list() -> None:
    """MissingDateColumnError stores ``available`` as an empty list by default."""
    err = MissingDateColumnError("prices", column="Date")
    assert err.available == []


def test_missing_date_column_error_copies_available_input() -> None:
    """MissingDateColumnError stores a copy of the provided available columns."""
    available = ["date", "asset"]
    err = MissingDateColumnError("prices", column="Date", available=available)
    available.append("new_col")
    assert err.available == ["date", "asset"]
