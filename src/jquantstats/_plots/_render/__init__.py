"""Renderers turning a `~jquantstats._plots._spec.FigureSpec` into a figure.

One module per backend, and `render` is the only place that chooses between
them. Because every chart is described as a spec, that one function serves every
plot method — there is no per-method dispatch to write or to keep in step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from .._backend import Backend, require_backend, resolve
from ._plotly import render_plotly as render_plotly

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure as PlotlyFigure

    from .._spec import FigureSpec

    #: What a renderer returns once the backend is only known at runtime.
    #:
    #: Type-checking-only, like the two imports above it: matplotlib is an
    #: optional dependency, and every module here carries
    #: ``from __future__ import annotations``, so an annotation naming these is
    #: never evaluated.
    #:
    #: Public plot methods do not expose this union directly. They overload on
    #: the literal ``backend`` argument instead, so ``data.plots.returns()``
    #: still infers exactly ``go.Figure`` and no existing caller has to narrow
    #: a union that was not there before.
    Figure: TypeAlias = PlotlyFigure | MplFigure

__all__ = ["render", "render_plotly"]


def render(spec: FigureSpec, backend: Backend | None = None) -> Figure:
    """Render *spec* with the selected backend.

    Args:
        spec: The chart to draw.
        backend: An explicit choice, or None to use the ambient selection —
            a `~jquantstats.plot_backend` block, else the process-wide default.

    Returns:
        Figure: A `plotly.graph_objects.Figure` or a
        `matplotlib.figure.Figure`, according to the backend in effect.

    Raises:
        UnknownPlotBackendError: If *backend* names an unsupported backend.
        MissingBackendError: If the selected backend's library is not installed.

    """
    selected = resolve(backend)
    if selected == "plotly":
        return render_plotly(spec)

    # Imported here, not at module scope, so that `import jquantstats` never
    # imports matplotlib and the extra stays genuinely optional. The
    # availability check runs first so an absent library produces a message
    # naming the extra rather than a bare ModuleNotFoundError from deep inside
    # the import machinery.
    require_backend(selected)
    from ._mpl import render_mpl

    return render_mpl(spec)
