"""Session-wide setup shared by every test package under ``tests/``.

Two concerns live here rather than in a narrower conftest, because both must
take effect before any test module is imported:

* the matplotlib backend selection and its font-cache warm-up, and
* restoration of the process-wide jquantstats plot backend between tests.
"""

import os

# Selected before matplotlib (or quantstats, which imports pyplot at module scope) is
# first imported. jquantstats itself never imports pyplot — test__plots/test_no_pyplot.py
# enforces that — but the comparison tests do, and on a headless runner an unset backend
# is a source of flakes. On macOS runners the default selection is the MacOSX backend,
# which is slow and occasionally flaky under xdist.
#
# Importing matplotlib here rather than inside a test also pays its cold font-cache
# build at collection time, outside pytest.ini's global `timeout = 60`.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl  # noqa: F401 - imported for its side effects; import order is the point
import pytest

import jquantstats


@pytest.fixture(autouse=True)
def _reset_plot_backend():
    """Restore the process-wide plot backend after every test.

    `jquantstats.set_plot_backend` is deliberately global, so a test that
    changes it would otherwise leak into every test scheduled after it in the
    same worker — including tests that never mention a backend at all.
    """
    saved = jquantstats.get_plot_backend()
    yield
    jquantstats.set_plot_backend(saved)


@pytest.fixture(autouse=True)
def _no_pyplot_leak():
    """Fail if library code registered a figure in pyplot's global manager.

    The matplotlib backend builds figures with `matplotlib.figure.Figure` so
    that nothing accumulates in pyplot's ``Gcf`` registry — the leak behind
    issue #628. A plain ``plt.close("all")`` teardown would quietly clean up a
    regression instead of reporting it, so the delta is asserted first and only
    then cleared.
    """
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    plt.close("all")
    assert not leaked, (
        f"pyplot-managed figures {sorted(leaked)} leaked — build figures with "
        "matplotlib.figure.Figure(), not plt.figure()/plt.subplots()"
    )
