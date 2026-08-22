"""Selection of the rendering backend used by the plotting facades.

jquantstats renders every chart with `Plotly <https://plotly.com/python/>`_ by
default.  A second, optional `matplotlib <https://matplotlib.org/>`_ backend
produces static figures instead, which is markedly cheaper when a script builds
many charts at once.

Three levels of selection compose, most specific first:

1. a per-call ``backend=`` argument on any plot method,
2. a `plot_backend` context manager, scoped to the current thread or task,
3. `set_plot_backend`, the process-wide default (``"plotly"`` when never set).

Examples:
    >>> from jquantstats import get_plot_backend, plot_backend, set_plot_backend
    >>> get_plot_backend()
    'plotly'
    >>> set_plot_backend("matplotlib")
    >>> get_plot_backend()
    'matplotlib'
    >>> with plot_backend("plotly"):
    ...     get_plot_backend()
    'plotly'
    >>> get_plot_backend()
    'matplotlib'
    >>> set_plot_backend("plotly")
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.util import find_spec
from typing import Literal

from jquantstats.exceptions import MissingBackendError, UnknownPlotBackendError

__all__ = [
    "Backend",
    "get_plot_backend",
    "plot_backend",
    "require_backend",
    "resolve",
    "set_plot_backend",
]

Backend = Literal["matplotlib", "plotly"]

#: Accepted backend names, in the order error messages list them.
SUPPORTED: tuple[Backend, ...] = ("matplotlib", "plotly")

# The packaging extra that installs each *optional* backend's rendering library.
# Plotly is deliberately absent: it is a core dependency, so it is always importable
# and there is no extra to point anyone at. (The `plot` extra is kaleido — static image
# export *for* the plotly figures — which is a different thing entirely.)
_OPTIONAL_EXTRAS: dict[Backend, str] = {"matplotlib": "mpl"}

# The process-wide default. A plain module global on purpose: "last writer wins,
# visible from every thread" is the semantic a global setter should have, and a
# single reference store is atomic under both the GIL and free-threaded CPython.
_default: Backend = "plotly"

# The scoped override. A ContextVar rather than threading.local because it is the
# only primitive that scopes correctly across both threads and asyncio tasks.
_override: contextvars.ContextVar[Backend | None] = contextvars.ContextVar(
    "jquantstats_plot_backend",
    default=None,
)

# Bound at module scope on purpose: the tests simulate an absent library by
# replacing *this* name, so they never poison ``sys.modules``. A ``None`` sentinel
# left in ``sys.modules`` survives a failed test and leaks across the many tests
# an xdist worker runs in one interpreter; rebinding an attribute cannot.
_find_spec = find_spec


def _validate(backend: str) -> Backend:
    """Narrow *backend* to `Backend`, rejecting anything unsupported.

    Args:
        backend: The candidate backend name.

    Returns:
        Backend: The same name, narrowed for the type checker.

    Raises:
        UnknownPlotBackendError: If *backend* is not a supported name.

    """
    if backend not in SUPPORTED:
        raise UnknownPlotBackendError(backend, list(SUPPORTED))
    return backend


def set_plot_backend(backend: Backend) -> None:
    """Set the process-wide default plotting backend.

    Args:
        backend: Either ``"plotly"`` (the default) or ``"matplotlib"``.

    Raises:
        UnknownPlotBackendError: If *backend* is not a supported name.

    Examples:
        >>> from jquantstats import get_plot_backend, set_plot_backend
        >>> set_plot_backend("matplotlib")
        >>> get_plot_backend()
        'matplotlib'
        >>> set_plot_backend("plotly")

    """
    global _default
    _default = _validate(backend)


def get_plot_backend() -> Backend:
    """Return the backend currently in effect.

    Returns:
        Backend: The scoped override if one is active, else the process-wide
        default.

    Examples:
        >>> from jquantstats import get_plot_backend
        >>> get_plot_backend()
        'plotly'

    """
    return _override.get() or _default


@contextmanager
def plot_backend(backend: Backend) -> Iterator[None]:
    """Select *backend* for the duration of the ``with`` block.

    Scoped to the current thread or asyncio task, and restored even if the body
    raises.  This is the only exception-safe way to use both backends in one
    process; pairing `set_plot_backend` calls by hand leaks the override when
    the code between them raises.

    Args:
        backend: Either ``"plotly"`` or ``"matplotlib"``.

    Yields:
        None: Control returns to the ``with`` body.

    Raises:
        UnknownPlotBackendError: If *backend* is not a supported name.

    Examples:
        >>> from jquantstats import get_plot_backend, plot_backend
        >>> with plot_backend("matplotlib"):
        ...     get_plot_backend()
        'matplotlib'
        >>> get_plot_backend()
        'plotly'

    """
    token = _override.set(_validate(backend))
    try:
        yield
    finally:
        _override.reset(token)


def resolve(backend: Backend | None) -> Backend:
    """Resolve an explicit per-call *backend* against the ambient selection.

    Args:
        backend: An explicit choice, or ``None`` to defer to the context
            manager and then the process-wide default.

    Returns:
        Backend: The backend to render with.

    Raises:
        UnknownPlotBackendError: If *backend* is not a supported name.

    Examples:
        >>> from jquantstats._plots._backend import resolve
        >>> resolve("matplotlib")
        'matplotlib'
        >>> resolve(None)
        'plotly'

    """
    return _validate(backend) if backend is not None else get_plot_backend()


def require_backend(backend: Backend) -> None:
    """Raise unless *backend*'s rendering library is importable.

    Only the optional backends are checked; plotly is a core dependency, so its
    absence means a broken install rather than a missing extra and there is no
    useful hint to give.

    Availability is probed with `importlib.util.find_spec` rather than a
    ``try``/``except ImportError`` around the import itself: a spec lookup is an
    ordinary condition whose arms are both reachable under the branch-coverage
    gate, whereas forcing a real ``ImportError`` would mean poisoning
    ``sys.modules`` — which leaks past a failed test and is a no-op once the
    module has already been imported.

    Args:
        backend: The backend about to be used.

    Raises:
        MissingBackendError: If *backend* is optional and not installed.

    Examples:
        >>> from jquantstats._plots._backend import require_backend
        >>> require_backend("plotly")
        >>> require_backend("matplotlib")

    """
    extra = _OPTIONAL_EXTRAS.get(backend)
    if extra is not None and _find_spec(backend) is None:
        raise MissingBackendError(backend, extra)
