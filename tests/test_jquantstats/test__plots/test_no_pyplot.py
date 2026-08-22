"""Guard the ban on ``matplotlib.pyplot`` inside the library.

``pyplot`` keeps every figure it creates alive in a global ``Gcf`` registry, so
a caller looping over hundreds of portfolios accumulates figures until
matplotlib warns and memory grows without bound. That resource cost is the
complaint behind issue #628, so the matplotlib backend builds figures with
`matplotlib.figure.Figure` directly: nothing is registered, nothing leaks, and
importing jquantstats never selects a GUI backend or mutates ``rcParams``.

This cannot be an import-linter contract — import-linter rejects
``matplotlib.pyplot`` with *"subpackages of external packages are not valid"*,
and forbidding the whole ``matplotlib`` distribution would ban the renderer
itself. So the ban is enforced two ways instead:

* statically, by scanning the shipped source for a ``pyplot`` reference, which
  catches the mistake at authoring time even in a branch no test executes; and
* dynamically, by `test_importing_jquantstats_does_not_import_pyplot`, which
  catches an indirect import that the scan cannot see.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).absolute().parents[3] / "src" / "jquantstats"

# Matches `import matplotlib.pyplot`, `from matplotlib.pyplot import ...`,
# `from matplotlib import pyplot`, and any bare `pylab` reference.
_PYPLOT = re.compile(r"\bpyplot\b|\bpylab\b")


def _python_sources() -> list[Path]:
    """Every shipped Python module, sorted for a stable parameter order.

    Returns:
        list[Path]: Paths to the ``.py`` files under ``src/jquantstats``.

    """
    return sorted(_SRC.rglob("*.py"))


def test_source_tree_is_discoverable() -> None:
    """The scan below is worthless if it silently matches no files."""
    assert _python_sources(), f"no Python sources found under {_SRC}"


@pytest.mark.parametrize("path", _python_sources(), ids=lambda p: p.name)
def test_module_does_not_reference_pyplot(path: Path) -> None:
    """No shipped module may reference pyplot or pylab.

    Use ``matplotlib.figure.Figure()`` and ``fig.subplots()`` instead of
    ``plt.figure()`` / ``plt.subplots()``; a directly constructed Figure stays
    out of pyplot's global registry and still supports ``fig.savefig(...)``.
    """
    hit = _PYPLOT.search(path.read_text(encoding="utf-8"))
    assert hit is None, (
        f"{path.relative_to(_SRC.parents[1])} references {hit.group(0)!r} — "
        "build figures with matplotlib.figure.Figure() instead, so they are "
        "never registered in pyplot's global Gcf registry (see #628)"
    )


def test_importing_jquantstats_does_not_import_pyplot() -> None:
    """Importing the package must not drag pyplot in, directly or indirectly.

    Runs in a subprocess because the test session itself imports pyplot (the
    leak-detector fixture and the quantstats comparison tests both need it), so
    an in-process check would always find it already loaded.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import jquantstats, sys; print('matplotlib.pyplot' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", "importing jquantstats pulled in matplotlib.pyplot"
