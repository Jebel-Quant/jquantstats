"""Guard the prose documentation against API drift.

The Markdown under ``docs/`` and the Claude Code skill under ``plugin/`` are
hand-written, so nothing stops them naming a method that was renamed or never
existed. This module resolves every accessor reference they make against the
installed package, so a rename breaks the suite instead of silently shipping a
guide that tells the reader to call something imaginary.

It checks *names*, not behaviour: that ``data.stats.sharpe`` exists, not that
the surrounding snippet runs. That is deliberately cheap, and it is the failure
mode the docs actually had.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jquantstats import CostModel, Data, Portfolio
from jquantstats._plots import DataPlots, PortfolioPlots
from jquantstats._reports._data import Reports
from jquantstats._reports._portfolio import Report
from jquantstats._stats._stats import Stats

_ROOT = Path(__file__).absolute().parent.parent.parent

# Dotted prefix as it appears in the docs → the class the next segment must
# resolve against. Intermediate prefixes are listed alongside the chains that
# extend them (``pf`` as well as ``pf.data.stats``), so every segment of a
# chain gets checked rather than only the last one.
ACCESSORS: dict[str, type] = {
    "data.stats": Stats,
    "data.plots": DataPlots,
    "data.reports": Reports,
    "pf": Portfolio,
    "pf.stats": Stats,
    "pf.plots": PortfolioPlots,
    "pf.report": Report,
    "pf.data": Data,
    "pf.data.stats": Stats,
    "pf.data.plots": DataPlots,
    "pf.data.reports": Reports,
    "jqs.Data": Data,
    "jqs.Portfolio": Portfolio,
    "CostModel": CostModel,
}

# References the docs make on purpose to names that do not exist.
KNOWN_ABSENT = frozenset(
    {
        # Placeholders in "qs.stats.foo(r) → data.stats.foo()" translation rules.
        "data.stats.foo",
        "data.plots.foo",
        # Named so the guides can say these are *not* the method you want,
        # steering readers to data.reports.metrics()/full() and
        # data.stats.summary(). See docs/MIGRATION.md.
        "data.reports.summary",
        "data.reports.to_html",
    }
)

# A bare ``data`` root is deliberately absent from ACCESSORS: it would match the
# module name in prose like "data.py", and the accessors above already cover
# every reference the docs make through a Data instance.


def _documentation_files() -> list[Path]:
    """Collect the hand-written Markdown that references the public API.

    Returns:
        list[Path]: Markdown files under docs/, the README, and any bundled
        Claude Code skill, sorted for a stable parameter order.

    """
    return sorted(
        [
            *(_ROOT / "docs").glob("*.md"),
            _ROOT / "README.md",
            *(_ROOT / "plugin" / "skills").rglob("SKILL.md"),
        ]
    )


def _unresolved_names(text: str) -> list[str]:
    """Find accessor references in ``text`` that the package cannot resolve.

    Args:
        text: Markdown source to scan.

    Returns:
        list[str]: Sorted dotted names that do not resolve, excluding
        `KNOWN_ABSENT`.

    """
    found = {
        f"{prefix}.{match.group(1)}"
        for prefix, owner in ACCESSORS.items()
        # The lookbehind stops "pf.data.stats" from also matching as the
        # shorter "data.stats" glued to the tail of a longer chain.
        for match in re.finditer(rf"(?<![\w.]){re.escape(prefix)}\.([a-z_][a-z0-9_]*)", text)
        if not hasattr(owner, match.group(1))
    }
    return sorted(found - KNOWN_ABSENT)


@pytest.mark.parametrize("path", _documentation_files(), ids=lambda p: str(p.relative_to(_ROOT)))
def test_documented_api_names_resolve(path: Path) -> None:
    """Every accessor the documentation names exists on the installed package.

    Args:
        path: A Markdown file to scan.

    Verifies:
        No `data.stats.*`, `pf.plots.*`, `jqs.Portfolio.*` (etc.) reference
        points at a method the package does not provide.

    """
    unresolved = _unresolved_names(path.read_text())
    assert not unresolved, (
        f"{path.relative_to(_ROOT)} references names that do not exist: {unresolved}. "
        f"Rename them, or add them to KNOWN_ABSENT if the text names them precisely "
        f"because they are absent."
    )


def test_checker_detects_a_broken_name() -> None:
    """The scan reports an accessor that the package does not provide.

    Verifies:
        A reference to a non-existent method is flagged, so the guard above
        cannot pass merely because the regex stopped matching.

    """
    assert _unresolved_names("call `data.stats.not_a_metric()` now") == ["data.stats.not_a_metric"]


def test_checker_accepts_a_real_name() -> None:
    """The scan stays quiet on an accessor that does exist.

    Verifies:
        A valid reference is not reported, guarding against a checker that
        fails everything.

    """
    assert _unresolved_names("call `data.stats.sharpe()` now") == []


def test_known_absent_entries_are_still_absent() -> None:
    """Every KNOWN_ABSENT name is genuinely missing from the package.

    Verifies:
        The allowlist does not outlive its reason. Once such a name is
        implemented, this fails and the entry must be dropped so real
        references to it start being checked.

    """
    revived = [
        name
        for name in sorted(KNOWN_ABSENT)
        for prefix, owner in ACCESSORS.items()
        if name.startswith(f"{prefix}.") and hasattr(owner, name.removeprefix(f"{prefix}."))
    ]
    assert not revived, f"KNOWN_ABSENT names now exist and should be removed from the allowlist: {revived}"
