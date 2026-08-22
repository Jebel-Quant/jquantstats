"""Guard the public API surface against accidental removals."""

import jquantstats

PUBLIC_API = [
    "CostModel",
    "Data",
    "NativeFrame",
    "NativeFrameOrScalar",
    "Portfolio",
    "Result",
    "get_plot_backend",
    "interpolate",
    "plot_backend",
    "set_plot_backend",
]


def test_public_api_complete() -> None:
    """Assert every name in PUBLIC_API is present in the jquantstats namespace."""
    for name in PUBLIC_API:
        assert hasattr(jquantstats, name), f"{name!r} missing from public API"


def test_public_api_matches_dunder_all() -> None:
    """Assert PUBLIC_API and ``__all__`` describe the same surface.

    The two drifted before: ``Result`` and ``interpolate`` were exported from
    ``__all__`` but absent from PUBLIC_API, so removing either would not have
    failed `test_public_api_complete`. Comparing the sets makes the guard
    self-maintaining instead of hand-maintained.
    """
    assert sorted(PUBLIC_API) == sorted(jquantstats.__all__)


def test_version_exposed() -> None:
    """Assert that __version__ is exposed at the top-level package."""
    assert hasattr(jquantstats, "__version__")
