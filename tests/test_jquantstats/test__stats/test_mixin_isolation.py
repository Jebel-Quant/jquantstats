"""Isolation tests for stats mixins with cross-mixin dependencies."""

import pytest

from jquantstats._stats._reporting import _ReportingStatsMixin


class _IsolatedReportingStatsMixin(_ReportingStatsMixin):
    """Concrete isolated reporting mixin used to exercise dependency errors."""

    def cagr(self, periods=None):
        """Return a stub CAGR mapping for dependency testing."""
        return {"META": 0.1}


def test_reporting_mixin_rar_raises_attribute_error_for_missing_exposure_dependency():
    """The rar method should clearly fail when reporting mixin is used without basic mixin methods."""
    isolated = _IsolatedReportingStatsMixin()

    with pytest.raises(AttributeError, match="exposure"):
        isolated.rar()
