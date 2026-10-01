"""Execute the ``pycon`` examples in ``README.md``.

``test_doctests.py`` runs the ``>>>`` examples in the library's docstrings, but
the README is not a module, so nothing executed its examples: ``make
docs-examples`` and the rhiza README checks only *parse* fences, and report a
``pycon`` block as not checkable.  The README is the first thing a newcomer
copies from, so its examples are gated here, inside ``make test``.

The fences are one continuous interpreter session -- later blocks use names the
earlier ones defined -- so they run in document order in a single shared
namespace, from a single test.  Splitting them into parameters would break that
chain under ``pytest -n auto``, which may hand consecutive fences to different
workers.  A failure still names the README line of the fence that drifted.
"""

from __future__ import annotations

import doctest
import re
from pathlib import Path

import pytest

# Same convention as the docstring examples: long outputs elide with ``...``.
_OPTIONFLAGS = doctest.ELLIPSIS

_FENCE = re.compile(r"^```pycon\n(.*?)^```", re.DOTALL | re.MULTILINE)


def _fences(readme: Path) -> list[tuple[int, str]]:
    """Extract every ``pycon`` fence from *readme*, in document order.

    Args:
        readme: Path to ``README.md``.

    Returns:
        list[tuple[int, str]]: The 1-based line of each opening fence and its body.

    """
    text = readme.read_text(encoding="utf-8")
    return [(text.count("\n", 0, match.start()) + 1, match.group(1)) for match in _FENCE.finditer(text)]


def test_readme_examples_are_discoverable(readme_path: Path) -> None:
    """Guard the guard: a gate that collects nothing passes forever.

    Pinned to floors rather than exact counts, so adding an example does not fail
    the suite while losing the fences -- or breaking the fence pattern -- still
    does.  At the time of writing the README carries 67 examples in 12 fences.
    """
    parser = doctest.DocTestParser()
    fences = _fences(readme_path)
    total = sum(len(parser.get_examples(body)) for _, body in fences)
    assert len(fences) >= 10, f"only {len(fences)} pycon fence(s) found in {readme_path}"
    assert total >= 50, f"only {total} pycon example(s) found in {readme_path}"


def test_readme_examples(readme_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every ``>>>`` example in the README still evaluates to its documented output.

    Runs from *tmp_path*, because the report example writes ``output/report.html``
    relative to the working directory.
    """
    monkeypatch.chdir(tmp_path)
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=_OPTIONFLAGS)
    globs: dict[str, object] = {"__name__": "README"}
    report: list[str] = []
    failed_at: list[int] = []

    for line, body in _fences(readme_path):
        test = parser.get_doctest(body, globs, f"README.md:{line}", str(readme_path), line)
        result = runner.run(test, out=report.append, clear_globs=False)
        # DocTest copies the namespace it is given; carry its copy forward.
        globs = test.globs
        if result.failed:
            failed_at.append(line)

    assert not failed_at, f"README pycon fence(s) at line(s) {failed_at} drifted:\n" + "".join(report)
