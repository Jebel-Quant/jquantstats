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

* statically, by inspecting each module's import statements, which catches the
  mistake at authoring time even in a branch no test executes; and
* dynamically, by `test_importing_jquantstats_does_not_import_pyplot`, which
  catches an indirect import that no static check can see.

The static half reads the parsed syntax tree rather than the file's text. A
text search also matches prose — this module and the matplotlib renderer both
have to *name* pyplot to explain why they avoid it — and a guard that punishes
its own documentation would just be commented out.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).absolute().parents[3] / "src" / "jquantstats"
_BANNED = {"matplotlib.pyplot", "pylab"}


def _python_sources() -> list[Path]:
    """Every shipped Python module, sorted for a stable parameter order.

    Returns:
        list[Path]: Paths to the ``.py`` files under ``src/jquantstats``.

    """
    return sorted(_SRC.rglob("*.py"))


def _banned_imports(source: str) -> list[str]:
    """Find banned module imports in *source*.

    Covers ``import matplotlib.pyplot``, ``from matplotlib.pyplot import ...``
    and ``from matplotlib import pyplot``, in any nesting — a function-local
    import is exactly as leaky as a top-level one.

    Args:
        source: Python source text.

    Returns:
        list[str]: The banned module names imported, in source order.

    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names if alias.name in _BANNED)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in _BANNED:
                found.append(module)
            elif module == "matplotlib":
                found.extend(f"matplotlib.{a.name}" for a in node.names if a.name == "pyplot")
    return found


def test_source_tree_is_discoverable() -> None:
    """The scan below is worthless if it silently matches no files."""
    assert _python_sources(), f"no Python sources found under {_SRC}"


def test_detects_a_banned_import() -> None:
    """The detector itself works, on each spelling it must catch."""
    assert _banned_imports("import matplotlib.pyplot as plt") == ["matplotlib.pyplot"]
    assert _banned_imports("from matplotlib import pyplot") == ["matplotlib.pyplot"]
    assert _banned_imports("from matplotlib.pyplot import plot") == ["matplotlib.pyplot"]
    assert _banned_imports("def f():\n    import pylab") == ["pylab"]
    assert _banned_imports("from matplotlib.figure import Figure") == []
    assert _banned_imports('"""A docstring mentioning pyplot and plt."""') == []


@pytest.mark.parametrize("path", _python_sources(), ids=lambda p: p.name)
def test_module_does_not_import_pyplot(path: Path) -> None:
    """No shipped module may import pyplot or pylab.

    Use ``matplotlib.figure.Figure()`` and ``fig.subplots()`` instead of
    ``plt.figure()`` / ``plt.subplots()``; a directly constructed Figure stays
    out of pyplot's global registry and still supports ``fig.savefig(...)``.
    """
    found = _banned_imports(path.read_text(encoding="utf-8"))
    assert not found, (
        f"{path.relative_to(_SRC.parents[1])} imports {found} — build figures with "
        "matplotlib.figure.Figure() instead, so they are never registered in "
        "pyplot's global Gcf registry (see #628)"
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


def test_rendering_does_not_import_pyplot() -> None:
    """Nor may actually drawing a matplotlib figure pull pyplot in.

    The import-time check above would still pass if the renderer imported
    pyplot lazily inside its drawing code, which is precisely where a
    ``plt.subplots()`` would end up.

    One matplotlib-internal exception exists — see
    `test_box_charts_may_import_pyplot_but_never_register_a_figure`.
    """
    script = (
        "import sys, datetime as dt, polars as pl\n"
        "from jquantstats import Data\n"
        "d0 = dt.date(2023, 1, 1)\n"
        "r = pl.DataFrame({'date': [d0 + dt.timedelta(days=i) for i in range(9)],\n"
        "                  'A': [0.01, -0.02, 0.03] * 3})\n"
        "Data.from_returns(returns=r).plots.returns(backend='matplotlib')\n"
        "print('matplotlib.pyplot' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", "rendering a matplotlib figure pulled in pyplot"


def test_box_charts_may_import_pyplot_but_never_register_a_figure() -> None:
    """Box drawing pulls pyplot in from inside matplotlib, harmlessly.

    ``Axes.bxp`` reads ``rcParams.items()``, and ``RcParams.__getitem__``
    imports pyplot to resolve the lazily-chosen backend name
    (``matplotlib/__init__.py``). So rendering `distribution` on the matplotlib
    backend leaves ``matplotlib.pyplot`` in ``sys.modules`` even though this
    package never imports it.

    That is backend *name resolution*, not figure management, and it is the
    latter that issue #628 is about. What must still hold — and what this
    pins — is that no figure is registered in pyplot's global manager, so
    nothing accumulates. Recorded as a known exception so nobody later
    "fixes" the import away by reaching for ``plt``.
    """
    script = (
        "import sys, datetime as dt, polars as pl\n"
        "from jquantstats import Data\n"
        "d0 = dt.date(2021, 1, 1)\n"
        "r = pl.DataFrame({'date': [d0 + dt.timedelta(days=i) for i in range(40)],\n"
        "                  'A': [0.01, -0.02, 0.03, -0.004] * 10})\n"
        "Data.from_returns(returns=r).plots.distribution(backend='matplotlib')\n"
        "import matplotlib.pyplot as plt\n"
        "print(plt.get_fignums())\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]", "a box chart registered a figure with pyplot"
