"""Tests for kaleido static image export (fig.to_image / fig.write_image).

All tests are skipped automatically when the ``plot`` extra (kaleido) is not
installed so the base test suite remains dependency-free.
"""

from __future__ import annotations

import os
import sys

import pytest

kaleido = pytest.importorskip("kaleido", reason="kaleido not installed (pip install jquantstats[plot])")


def _chrome_available() -> bool:
    """Report whether kaleido can locate a Chrome/Chromium to render with.

    kaleido v1 renders images by driving a headless Chrome subprocess. With no
    browser on the system (and none downloaded via ``plotly_get_chrome``),
    ``to_image`` / ``write_image`` block until the watchdog kills them — a 300s
    hang rather than a clean skip. Resolve the browser the same way kaleido does
    (choreographer's locator, which honours ``$BROWSER_PATH`` and the choreo
    download cache) so these tests skip cleanly in a browserless dev env while
    still running on CI runners, which ship Chrome. If the choreographer API
    ever moves, fall back to *not* skipping so CI coverage is never silently lost.
    """
    if os.environ.get("BROWSER_PATH"):
        return True
    try:
        from choreographer.browsers.chromium import chromium_based_browsers, get_browser_path

        return get_browser_path(chromium_based_browsers) is not None
    except Exception:
        return True


pytestmark = [
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="kaleido launches a Chrome subprocess that crashes the xdist worker on Windows CI",
    ),
    pytest.mark.skipif(
        not _chrome_available(),
        reason="no Chrome/Chromium found for kaleido (install a browser or run `plotly_get_chrome`)",
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
