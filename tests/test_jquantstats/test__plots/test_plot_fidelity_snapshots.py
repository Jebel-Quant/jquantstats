"""Full-fidelity snapshot tests for Data.plots figures.

Unlike ``test_plot_snapshots.py`` (which fingerprints only trace counts and a
few layout keys), these tests pin the **complete** figure JSON — colors,
hovertemplates, axis configuration, legend placement, marker styles — via
``fig.to_json()``.  They exist to kill mutation-testing survivors in
``_plots/_data.py``: any change to a styling literal or layout constant
alters the serialized figure.

Long data arrays are digested to length + head/tail samples so the snapshot
file stays reviewable; computation mutants still shift the digests, while
styling mutants are pinned verbatim.

The shared ``data`` fixture is deterministic; charts that draw through
numpy's global RNG (Monte Carlo, distribution sampling) are seeded.

Run with ``--snapshot-update`` after an intentional styling change.
"""

from __future__ import annotations

import base64
import json
import math

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

_ARRAY_DIGEST_THRESHOLD = 8


@pytest.fixture
def plots(data):
    """Plots facade of the shared deterministic data fixture."""
    return data.plots


def _nan_safe(value):
    """Make non-finite floats snapshot-comparable (nan != nan breaks equality)."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def _digest(node, sort_arrays: bool = False):
    """Recursively replace long lists with a length + head/tail digest.

    Plotly serializes numeric arrays as base64 ``bdata`` blobs; those are
    decoded back to numbers first so the digest reflects actual values.
    """
    if isinstance(node, dict):
        if "bdata" in node and "dtype" in node:
            decoded = np.frombuffer(base64.b64decode(node["bdata"]), dtype=node["dtype"])
            if "shape" in node:
                decoded = decoded.reshape([int(d) for d in str(node["shape"]).split(",")])
            return _digest(decoded.tolist(), sort_arrays)
        return {k: _digest(v, sort_arrays) for k, v in node.items()}
    if isinstance(node, list):
        if len(node) > _ARRAY_DIGEST_THRESHOLD and not any(isinstance(x, (dict, list)) for x in node):
            values = sorted(node, key=repr) if sort_arrays else node
            return {
                "__len__": len(values),
                "head": [_nan_safe(x) for x in values[:3]],
                "tail": [_nan_safe(x) for x in values[-2:]],
            }
        return [_digest(x, sort_arrays) for x in node]
    return _nan_safe(node)


def _figure_json(fig, sort_arrays: bool = False) -> dict:
    """Canonical JSON serialization of a figure with long arrays digested.

    The plotly default template (a large built-in theme dict that no module
    code controls) is dropped.  ``sort_arrays`` normalizes order-insensitive
    traces (e.g. box plots fed from unordered group-bys).
    """
    payload = json.loads(fig.to_json())
    payload.get("layout", {}).pop("template", None)
    return _digest(payload, sort_arrays=sort_arrays)


_NO_ARG_METHODS = [
    "snapshot",
    "returns",
    "compare",
    "log_returns",
    "daily_returns",
    "yearly_returns",
    "monthly_returns",
    "monthly_heatmap",
    "histogram",
    "drawdown",
    "drawdowns_periods",
    "earnings",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_volatility",
    "rolling_beta",
]


@pytest.mark.parametrize("method", _NO_ARG_METHODS)
def test_plot_full_figure_json(plots, method: str, snapshot: SnapshotAssertion):
    """Each default-argument figure serializes to exactly the pinned JSON."""
    fig = getattr(plots, method)()
    assert _figure_json(fig) == snapshot


def test_distribution_full_figure_json(plots, snapshot: SnapshotAssertion):
    """The period-distribution chart serializes to the pinned JSON.

    Box-plot arrays come from unordered group-bys, so they are sorted before
    digesting.
    """
    fig = plots.distribution()
    assert _figure_json(fig, sort_arrays=True) == snapshot


def test_montecarlo_full_figure_json(plots, snapshot: SnapshotAssertion):
    """The Monte Carlo fan chart serializes to exactly the pinned JSON under a fixed seed."""
    np.random.seed(20260611)
    fig = plots.montecarlo(n=5, period=30)
    assert _figure_json(fig) == snapshot


def test_montecarlo_distribution_full_figure_json(plots, snapshot: SnapshotAssertion):
    """The Monte Carlo distribution chart serializes to exactly the pinned JSON under a fixed seed."""
    np.random.seed(20260611)
    fig = plots.montecarlo_distribution(n=20, period=30)
    assert _figure_json(fig) == snapshot
