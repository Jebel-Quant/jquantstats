"""Guard the repo's 100% coverage gate against rhiza template-sync regressions.

The gate used to live in ``.rhiza/make.d/custom-env.mk`` (``COVERAGE_FAIL_UNDER ?= 100``),
and this test read it back with ``make -s print-COVERAGE_FAIL_UNDER``. Rhiza v1.4.0 retired
the make layer entirely — that file, the ``print-%`` target and the whole
``.rhiza/make.d/`` folder are gone — so the value now lives in ``[tool.rhiza-task]`` in
``pyproject.toml``, which is where the ``rhiza-task`` CLI reads it from.

The regression this guards against is unchanged in substance: drop the setting and
``rhiza-task`` falls back to its own default of 90, so the suite could shed ten points of
coverage while CI stayed green. It is read straight out of the committed TOML rather than
through the CLI, so the test needs no network, no ``uvx`` and no subprocess.

``[tool.coverage.report] fail_under`` is asserted alongside it because the two are
genuinely independent: the ``test`` task passes ``--cov-fail-under`` on the command line,
which outranks the config file, so a bare-``pytest`` run and a ``rhiza-task test`` run
enforce different numbers if these ever drift apart.
"""

import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_EXPECTED = 100
_CLI_DEFAULT = 90  # rhiza-task's own fallback, i.e. what a dropped setting silently buys


def _pyproject() -> dict:
    """Parse the repo's ``pyproject.toml``.

    Returns:
        The parsed TOML document.
    """
    with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_coverage_gate_is_100():
    """``[tool.rhiza-task] coverage_fail_under`` must be 100, not rhiza-task's default of 90."""
    rhiza_task = _pyproject()["tool"]["rhiza-task"]
    assert "coverage_fail_under" in rhiza_task, (
        "[tool.rhiza-task] has no coverage_fail_under — the gate silently falls back to "
        f"rhiza-task's default of {_CLI_DEFAULT}%"
    )
    assert rhiza_task["coverage_fail_under"] == _EXPECTED


def test_coverage_report_fail_under_agrees():
    """``[tool.coverage.report] fail_under`` must match, so a bare pytest run enforces the same gate."""
    assert _pyproject()["tool"]["coverage"]["report"]["fail_under"] == _EXPECTED
