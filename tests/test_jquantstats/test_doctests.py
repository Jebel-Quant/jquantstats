"""Execute the ``>>>`` examples in the library's docstrings.

``make docs-coverage`` (interrogate) asks whether a docstring *exists*; nothing
asked whether what it *claims* is still true.  The examples under
``src/jquantstats`` rendered perfectly while being free to drift, and the person
who would have found out is a newcomer copying one that no longer works.

``make docs-examples`` does not close this: it parses the fences in ``README.md``
and ``docs/``, and never opens a ``.py`` file.  The obvious alternative --
pytest's own ``--doctest-modules`` -- would have to be set in ``pytest.ini``,
which is template-owned (it appears in ``.rhiza/template.lock``), so setting it
here would be reverted by the next sync.  Collecting the examples from a test
module keeps the gate repo-owned and puts it inside ``make test``, which is what
CI already runs.

Each module is its own parameter rather than one test over the whole tree, so a
failure names the module that drifted instead of the first one that happened to
be checked.
"""

from __future__ import annotations

import doctest
import importlib
from pathlib import Path
from types import ModuleType

import pytest

_SRC = Path(__file__).absolute().parents[2] / "src"
_PACKAGE = _SRC / "jquantstats"

# The docstrings are written against ELLIPSIS -- `UnknownPlotBackendError` ends
# its expected traceback with `; ...` rather than repeating the full message, and
# it is not the only one. pytest enables this flag by default for
# `--doctest-modules`, bare `doctest` does not, so omitting it here would fail
# examples that are correct under the convention they were written in.
_OPTIONFLAGS = doctest.ELLIPSIS


def _module_names() -> list[str]:
    """Every shipped module, as a dotted name, sorted for a stable order.

    Returns:
        list[str]: Importable names for the ``.py`` files under ``src/jquantstats``.

    """
    names = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        parts = path.relative_to(_SRC).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


def _count(name: str) -> int:
    """Count the doctest examples *name* carries.

    Args:
        name: Dotted module name to import and scan.

    Returns:
        int: The number of ``>>>`` examples found in that module's docstrings.

    """
    module = importlib.import_module(name)
    return sum(len(test.examples) for test in doctest.DocTestFinder().find(module))


def _with_examples() -> list[str]:
    """The subset of modules that actually carry doctest examples.

    Parametrising over every module would report dozens of vacuous passes for the
    modules that document themselves in prose, which hides how many examples are
    really being exercised.

    Returns:
        list[str]: Dotted names of modules containing at least one example.

    """
    return [name for name in _module_names() if _count(name)]


def _run(module: ModuleType) -> tuple[int, int]:
    """Run every doctest in *module* and report the tally.

    Args:
        module: The already-imported module to examine.

    Returns:
        tuple[int, int]: The number of failed examples and attempted examples.

    """
    runner = doctest.DocTestRunner(optionflags=_OPTIONFLAGS)
    for test in doctest.DocTestFinder().find(module):
        runner.run(test)
    return runner.failures, runner.tries


def test_examples_are_discoverable() -> None:
    """Guard the guard: a gate that collects nothing passes forever.

    Pinned to floors rather than exact counts, so adding an example or a module
    does not fail the suite while deleting the lot -- or breaking the path
    arithmetic above -- still does. At the time of writing the tree carries 265
    examples across 18 modules.
    """
    modules = _with_examples()
    total = sum(_count(name) for name in modules)
    assert len(modules) >= 15, f"only {len(modules)} module(s) with examples found under {_PACKAGE}"
    assert total >= 200, f"only {total} doctest example(s) found under {_PACKAGE}"


@pytest.mark.parametrize("name", _with_examples())
def test_docstring_examples(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Every ``>>>`` example in *name* still evaluates to its documented output.

    Args:
        name: Dotted module name to check.
        capsys: Captures the report doctest writes to stdout, so a failure can
            quote it instead of leaving it in the test run's output.

    """
    failures, tries = _run(importlib.import_module(name))
    report = capsys.readouterr().out
    assert tries, f"{name} was selected as carrying examples but ran none"
    assert not failures, f"{failures} of {tries} doctest example(s) failed in {name}:\n\n{report}"
