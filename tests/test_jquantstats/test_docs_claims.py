"""Guard the numeric claims the comparison docs make about this package.

`docs/engineering.md` compares jquantstats with QuantStats, and several of its
rows are facts about *this* repository rather than prose. Those drift silently:
the dependency count was a headline differentiator ("6, no pandas") at a point
when adding matplotlib to `[project] dependencies` rather than to an extra would
have quietly invalidated it, and the module-size row named a file that had not
existed for several releases.

Only the self-referential claims are checked. The QuantStats columns describe a
package this suite does not install and are left to human review.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).absolute().parent.parent.parent
_ENGINEERING = _ROOT / "docs" / "engineering.md"


def _pyproject() -> dict:
    """Parse the project manifest.

    Returns:
        dict: The decoded ``pyproject.toml``.

    """
    return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_runtime_dependency_count_matches_the_manifest() -> None:
    """The documented runtime-dependency count is the real one.

    "6 runtime deps, no pandas" is a claim the comparison leans on, so moving a
    package out of an extra and into `[project] dependencies` has to be a
    deliberate act that updates the prose too.
    """
    declared = len(_pyproject()["project"]["dependencies"])
    text = _ENGINEERING.read_text(encoding="utf-8")
    assert f"jquantstats runtime ({declared}):" in text, (
        f"docs/engineering.md does not say 'jquantstats runtime ({declared})' — "
        "the manifest declares that many runtime dependencies"
    )


def test_matplotlib_is_an_extra_not_a_runtime_dependency() -> None:
    """The second plotting backend must not reach the base install.

    Keeping it optional is the reason the count above stays at 6.
    """
    project = _pyproject()["project"]
    runtime = {d.split(">")[0].split("=")[0].split("[")[0] for d in project["dependencies"]}
    assert "matplotlib" not in runtime
    assert "matplotlib" in " ".join(project["optional-dependencies"]["mpl"])


def test_engineering_doc_names_a_module_that_exists() -> None:
    """The "largest module" row must name a real file.

    It named `_plots/_data.py` for several releases after that file was split
    into a package.
    """
    row = next(line for line in _ENGINEERING.read_text(encoding="utf-8").splitlines() if "| Largest module" in line)
    # Backtick-quoted filenames, wherever they sit in the cell — they are
    # parenthesised, so splitting on whitespace alone would miss them.
    named = re.findall(r"`([^`]+\.py)`", row)
    assert named, f"no module named in: {row}"
    for module in named:
        if module == "stats.py":  # QuantStats' file, not ours
            continue
        assert (_ROOT / "src" / "jquantstats" / module).exists(), f"{module} does not exist"
