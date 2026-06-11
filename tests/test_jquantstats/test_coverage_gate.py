"""Guard the repo's 100% coverage gate against rhiza template-sync regressions.

The gate lives in ``.rhiza/make.d/custom-env.mk`` (``COVERAGE_FAIL_UNDER ?= 100``)
because that is the only committed channel that passes the template's own
validation tests. A rhiza sync could reset that file to its example content,
silently dropping the gate back to the template default of 90 while CI stays
green — this test makes such a regression fail CI loudly instead.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[2]


@pytest.mark.skipif(shutil.which("make") is None, reason="make not available")
def test_coverage_gate_is_100():
    """`make print-COVERAGE_FAIL_UNDER` must resolve to 100, not the template default of 90."""
    proc = subprocess.run(
        ["make", "-s", "print-COVERAGE_FAIL_UNDER"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    plain = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout)
    values = re.findall(r"^\s*(\d+)\s*$", plain, flags=re.MULTILINE)
    assert values == ["100"], f"coverage gate is not 100 — custom-env.mk override lost? output:\n{plain}"
