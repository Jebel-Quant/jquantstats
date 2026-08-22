"""Cover backend selection, its precedence rules and the availability guard.

The rendering backend is chosen at three levels — a per-call ``backend=``
argument, a `plot_backend` context manager, and the process-wide
`set_plot_backend` default. These tests pin the precedence between them, the
isolation of the scoped override, and both arms of the availability check.
"""

from __future__ import annotations

import threading

import pytest

import jquantstats
from jquantstats._plots import _backend
from jquantstats.exceptions import MissingBackendError, UnknownPlotBackendError


def test_default_backend_is_plotly() -> None:
    """Plotly stays the default so existing code keeps its behaviour."""
    assert jquantstats.get_plot_backend() == "plotly"


def test_set_plot_backend_roundtrip() -> None:
    """The process-wide default is readable back through the getter."""
    jquantstats.set_plot_backend("matplotlib")
    assert jquantstats.get_plot_backend() == "matplotlib"


@pytest.mark.parametrize("name", ["ggplot", "bokeh", "", "PLOTLY"])
def test_set_plot_backend_rejects_unknown_names(name: str) -> None:
    """An unsupported name raises rather than silently falling back."""
    with pytest.raises(UnknownPlotBackendError, match="unknown plot backend"):
        jquantstats.set_plot_backend(name)


def test_unknown_backend_error_lists_the_supported_names() -> None:
    """The message names the alternatives so the fix is obvious."""
    with pytest.raises(UnknownPlotBackendError) as excinfo:
        jquantstats.set_plot_backend("ggplot")
    assert "'matplotlib'" in str(excinfo.value)
    assert "'plotly'" in str(excinfo.value)
    assert excinfo.value.backend == "ggplot"


def test_context_manager_restores_the_previous_backend() -> None:
    """The scoped override is undone on exit."""
    jquantstats.set_plot_backend("plotly")
    with jquantstats.plot_backend("matplotlib"):
        assert jquantstats.get_plot_backend() == "matplotlib"
    assert jquantstats.get_plot_backend() == "plotly"


def test_context_manager_restores_even_when_the_body_raises() -> None:
    """An exception inside the block must not leak the override.

    This is the reason the context manager exists: hand-pairing
    ``set_plot_backend`` calls leaks whenever the code between them raises.
    """
    jquantstats.set_plot_backend("plotly")
    with pytest.raises(RuntimeError), jquantstats.plot_backend("matplotlib"):
        raise RuntimeError("boom")
    assert jquantstats.get_plot_backend() == "plotly"


def test_context_managers_nest() -> None:
    """Nested blocks unwind to the enclosing selection, not to the default."""
    jquantstats.set_plot_backend("plotly")
    with jquantstats.plot_backend("matplotlib"):
        with jquantstats.plot_backend("plotly"):
            assert jquantstats.get_plot_backend() == "plotly"
        assert jquantstats.get_plot_backend() == "matplotlib"
    assert jquantstats.get_plot_backend() == "plotly"


def test_context_manager_rejects_unknown_names() -> None:
    """Validation happens on entry, before anything is overridden."""
    jquantstats.set_plot_backend("plotly")
    with pytest.raises(UnknownPlotBackendError), jquantstats.plot_backend("ggplot"):
        pytest.fail("the body must not run")  # pragma: no cover - unreachable
    assert jquantstats.get_plot_backend() == "plotly"


def test_scoped_override_does_not_leak_into_other_threads() -> None:
    """A ContextVar override is invisible to a thread that did not set it.

    A `threading.local` would behave the same here, but a ContextVar also
    scopes correctly across asyncio tasks, which is why it is the primitive
    used.
    """
    jquantstats.set_plot_backend("plotly")
    seen: list[str] = []

    def _observe() -> None:
        """Record the backend this thread sees."""
        seen.append(jquantstats.get_plot_backend())

    with jquantstats.plot_backend("matplotlib"):
        worker = threading.Thread(target=_observe)
        worker.start()
        worker.join()
        assert jquantstats.get_plot_backend() == "matplotlib"

    assert seen == ["plotly"]


def test_global_default_is_visible_from_other_threads() -> None:
    """Unlike the scoped override, the process-wide default is shared."""
    jquantstats.set_plot_backend("matplotlib")
    seen: list[str] = []

    worker = threading.Thread(target=lambda: seen.append(jquantstats.get_plot_backend()))
    worker.start()
    worker.join()

    assert seen == ["matplotlib"]


@pytest.mark.parametrize("explicit", ["plotly", "matplotlib"])
def test_explicit_backend_outranks_the_ambient_selection(explicit: str) -> None:
    """A per-call argument wins over both the context manager and the default."""
    jquantstats.set_plot_backend("plotly")
    with jquantstats.plot_backend("plotly"):
        assert _backend.resolve(explicit) == explicit


def test_resolve_none_defers_to_the_ambient_selection() -> None:
    """``backend=None`` means "whatever is currently selected"."""
    jquantstats.set_plot_backend("matplotlib")
    assert _backend.resolve(None) == "matplotlib"
    with jquantstats.plot_backend("plotly"):
        assert _backend.resolve(None) == "plotly"


def test_resolve_rejects_unknown_names() -> None:
    """An explicit bad name is rejected at the same place as a bad default."""
    with pytest.raises(UnknownPlotBackendError):
        _backend.resolve("ggplot")


def test_require_backend_passes_when_the_library_is_installed() -> None:
    """The guard is a no-op in an environment that has matplotlib."""
    assert _backend.require_backend("matplotlib") is None


def test_require_backend_ignores_plotly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plotly is a core dependency, so it is never guarded as an extra.

    Stubbing the spec lookup proves the short-circuit: even with every lookup
    reporting "absent", plotly does not raise, because there is no extra to
    point anyone at.
    """
    monkeypatch.setattr(_backend, "_find_spec", lambda name: None)
    assert _backend.require_backend("plotly") is None


def test_require_backend_raises_with_the_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent matplotlib names the extra that provides it.

    Absence is simulated by rebinding the module-level ``_find_spec`` alias
    rather than poisoning ``sys.modules``: a ``None`` sentinel left in
    ``sys.modules`` survives a failed test and leaks across every later test in
    the same xdist worker, and it is a no-op once matplotlib has already been
    imported.
    """
    monkeypatch.setattr(_backend, "_find_spec", lambda name: None)
    with pytest.raises(MissingBackendError, match=r"jquantstats\[mpl\]") as excinfo:
        _backend.require_backend("matplotlib")
    assert excinfo.value.extra == "mpl"
    assert excinfo.value.backend == "matplotlib"


def test_missing_backend_error_is_an_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers already guarding with ``except ImportError`` keep working."""
    monkeypatch.setattr(_backend, "_find_spec", lambda name: None)
    with pytest.raises(ImportError):
        _backend.require_backend("matplotlib")
