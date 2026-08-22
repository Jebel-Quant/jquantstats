"""Renderers turning a `~jquantstats._plots._spec.FigureSpec` into a figure.

One module per backend. Backend *selection* lands with the matplotlib renderer;
until there is a second backend to choose between, callers use `render_plotly`
directly.
"""

from ._plotly import render_plotly as render_plotly

__all__ = ["render_plotly"]
