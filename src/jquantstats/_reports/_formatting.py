"""Shared formatting helpers for report modules."""

from __future__ import annotations

import math
from typing import Any, TypeGuard


def _is_finite(v: Any) -> TypeGuard[int | float]:
    """Return True when *v* is a real, finite number."""
    if not isinstance(v, (int, float)):
        return False
    return math.isfinite(float(v))


def _fmt(value: Any, fmt: str = ".4f", suffix: str = "") -> str:
    """Format *value* for display; return ``"N/A"`` for non-finite values."""
    if not _is_finite(value):
        return "N/A"
    return f"{float(value):{fmt}}{suffix}"


def _plotly_div(fig: Any, include_plotlyjs: bool | str = False) -> str:
    """Serialise a Plotly figure to a standalone HTML ``<div>``."""
    import plotly.io as pio

    return str(pio.to_html(fig, full_html=False, include_plotlyjs=include_plotlyjs))


def _table_html(header_cells: str, body_html: str) -> str:
    """Wrap pre-rendered table cells and body rows in the shared report table shell."""
    return (
        '<table class="stats-table">'
        "<thead><tr>"
        f'<th class="metric-header">Metric</th>{header_cells}'
        "</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
    )
