"""Plotting utilities for portfolio analytics using Plotly.

Renders common portfolio visuals — snapshots, lagged performance curves,
smoothed-holdings curves, rolling risk metrics and lead/lag information ratio
bar charts. Designed for notebook use.
"""

from __future__ import annotations

from ._core import PortfolioPlots

__all__ = ["PortfolioPlots"]
