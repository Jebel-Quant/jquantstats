"""Keep the ``# noqa: TRY003`` exemption inside the scope it was granted.

``src/jquantstats/exceptions.py`` records one standing exemption: argument
validation raises the builtin `TypeError`, `ValueError` or `AttributeError`
rather than a member of the `JQuantStatsError` taxonomy, because those signal a
broken call contract rather than a condition in the caller's data.  Each such
raise carries a ``# noqa: TRY003``.

A prose exemption decays.  The suppression is three tokens long and silences a
rule on whatever line it happens to sit on, so the cheap way to quiet an
unrelated TRY003 -- or to stop a new domain exception complaining -- is to paste
it somewhere it was never meant to go, and nothing would notice.  This module is
what notices: it holds the exemption to the three builtins it names, so the
paragraph in ``exceptions.py`` describes what the tree actually does rather than
what it did when the paragraph was written.

The check is deliberately about *scope*, not *count*: adding a validation raise
is ordinary work and must not fail the suite, while using the suppression
outside the exemption must.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import pytest

_SRC = Path(__file__).absolute().parents[2] / "src" / "jquantstats"

# The three the exemption names. `AttributeError` is here for the accessor that
# rejects a `compare()` call made without benchmark data -- also a call-contract
# failure, and raised as the builtin Python already uses for a missing attribute.
_EXEMPT = frozenset({"AttributeError", "TypeError", "ValueError"})

_SUPPRESSION = "noqa: TRY003"


def _python_sources() -> list[Path]:
    """Every shipped Python module, sorted for a stable parameter order.

    Returns:
        list[Path]: Paths to the ``.py`` files under ``src/jquantstats``.

    """
    return sorted(_SRC.rglob("*.py"))


def _raised_name(node: ast.Raise) -> str | None:
    """Name the exception class *node* raises, if it can be read statically.

    Args:
        node: The ``raise`` statement to inspect.

    Returns:
        str | None: The class name for ``raise Foo(...)`` or a bare ``raise
        Foo``, and None for anything indirect (a re-raise, or a class held in a
        variable) that this check cannot judge.

    """
    exc = node.exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    if isinstance(exc, ast.Name):
        return exc.id
    if isinstance(exc, ast.Attribute):
        return exc.attr
    return None


def _suppressions(path: Path) -> list[tuple[int, str | None]]:
    """Find the TRY003 suppressions in *path* and what each one sits on.

    Args:
        path: The module to scan.

    Returns:
        list[tuple[int, str | None]]: One ``(line number, raised class name)``
        pair per suppression, with None where no ``raise`` starts on that line.

    """
    source = path.read_text(encoding="utf-8")
    raises = {node.lineno: node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Raise)}
    found = []
    # Tokenised rather than searched line by line. A text search also matches
    # prose, and the paragraph in exceptions.py that grants this exemption has
    # to *quote* the suppression in order to explain it -- a guard that fails on
    # its own documentation would just be deleted. Only COMMENT tokens are real
    # suppressions; the same text inside a docstring is a STRING token.
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and _SUPPRESSION in token.string:
            lineno = token.start[0]
            node = raises.get(lineno)
            found.append((lineno, _raised_name(node) if node else None))
    return found


def test_the_scan_finds_something() -> None:
    """Guard the guard: a scanner that matches nothing passes forever.

    A floor rather than an exact count, so adding a validation raise does not
    fail the suite while deleting the lot -- or breaking the path arithmetic
    above -- still does. At the time of writing there are 36 suppressions across
    16 modules.
    """
    total = sum(len(_suppressions(path)) for path in _python_sources())
    assert total >= 20, f"only {total} TRY003 suppression(s) found under {_SRC}; is the scan still working?"


@pytest.mark.parametrize("path", _python_sources(), ids=lambda p: p.name)
def test_try003_is_only_suppressed_for_builtin_validation(path: Path) -> None:
    """Every TRY003 suppression sits on a raise of an exempt builtin.

    Args:
        path: The module to check.

    """
    for lineno, raised in _suppressions(path):
        assert raised is not None, (
            f"{path.relative_to(_SRC.parent.parent)}:{lineno} carries `# {_SUPPRESSION}` "
            f"but no raise statement starts on that line"
        )
        assert raised in _EXEMPT, (
            f"{path.relative_to(_SRC.parent.parent)}:{lineno} suppresses TRY003 on `raise {raised}`, "
            f"which the exemption in exceptions.py does not cover — it names only "
            f"{', '.join(sorted(_EXEMPT))}. Either raise one of those, or give {raised} "
            f"its own message so the suppression is unnecessary."
        )
