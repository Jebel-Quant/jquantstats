"""Tests for kaleido static image export (fig.to_image / fig.write_image).

All tests are skipped automatically when the ``plot`` extra (kaleido) is not
installed so the base test suite remains dependency-free.
"""

from __future__ import annotations

import sys

import pytest

kaleido = pytest.importorskip("kaleido", reason="kaleido not installed (pip install jquantstats[plot])")

pytestmark = [
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="kaleido launches a Chrome subprocess that crashes the xdist worker on Windows CI",
    ),
    # The first render pays the Chrome cold-start, which can exceed the
    # global 60s timeout on a busy CI runner (flaked on the v0.9.4 tag build).
    pytest.mark.timeout(300),
]

# PNG magic bytes: \x89PNG
_PNG_MAGIC = b"\x89PNG"


# ── Data.plots ────────────────────────────────────────────────────────────────


@pytest.mark.kaleido
def test_data_plot_snapshot_to_image_returns_png_bytes(data):
    """to_image() on snapshot() returns non-empty PNG bytes."""
    fig = data.plots.snapshot()
    img = fig.to_image(format="png")
    assert isinstance(img, bytes)
    assert len(img) > 0
    assert img[:4] == _PNG_MAGIC


@pytest.mark.kaleido
def test_data_plot_snapshot_write_image(data, tmp_path):
    """write_image() writes a readable PNG file to disk."""
    out = tmp_path / "snapshot.png"
    fig = data.plots.snapshot()
    fig.write_image(str(out), format="png")
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:4] == _PNG_MAGIC


# ── Portfolio.plots ───────────────────────────────────────────────────────────


@pytest.mark.kaleido
def test_portfolio_snapshot_to_image_returns_png_bytes(pf):
    """to_image() on Portfolio.plots.snapshot returns non-empty PNG bytes."""
    fig = pf.plots.snapshot()
    img = fig.to_image(format="png")
    assert isinstance(img, bytes)
    assert len(img) > 0
    assert img[:4] == _PNG_MAGIC


@pytest.mark.kaleido
def test_portfolio_snapshot_write_image(pf, tmp_path):
    """write_image() writes a readable PNG file to disk for a Portfolio snapshot."""
    out = tmp_path / "portfolio_snapshot.png"
    fig = pf.plots.snapshot()
    fig.write_image(str(out), format="png")
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:4] == _PNG_MAGIC


@pytest.mark.kaleido
def test_portfolio_snapshot_to_image_svg(pf):
    """to_image() also works with SVG format."""
    fig = pf.plots.snapshot()
    img = fig.to_image(format="svg")
    assert isinstance(img, bytes)
    assert b"<svg" in img
