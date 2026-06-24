"""Fuzz jquantstats null-handling and returns ingestion against arbitrary frames.

``interpolate`` interior-forward-fills the numeric columns of an arbitrary
polars frame and is contracted to *never* crash, while ``Data.from_returns``
ingests an untrusted returns frame and must either build a ``Data`` object or
raise one of its documented errors (its guards raise ``ValueError``/``TypeError``
subclasses; polars raises its own errors on schema problems). This harness
exercises both contracts with coverage-guided input.

Run locally:
    pip install atheris polars numpy
    python tests/fuzz/fuzz_stats.py -atheris_runs=20000

Run in ClusterFuzzLite: this file is built by .clusterfuzzlite/build.sh.
"""

from __future__ import annotations

import contextlib
import sys

import atheris

# Pre-import the heavy native dependencies OUTSIDE the instrumentation block.
# Atheris's bytecode instrumentation miscompiles parts of polars' Python
# machinery (an instrumented build returns None from Series.is_not_null(),
# which crashes interpolate()), so we let these libraries load uninstrumented
# and instrument only the first-party package under test. Importing the
# top-level packages pulls in their submodules, so jquantstats's own imports
# below hit the module cache and are never re-instrumented.
import narwhals  # noqa: F401  # pre-imported uninstrumented (see note above)
import numpy as np  # noqa: F401  # pre-imported uninstrumented (see note above)
import polars as pl

with atheris.instrument_imports():
    from jquantstats import Data, interpolate
    from jquantstats.exceptions import JQuantStatsError

# Errors ``from_returns`` is allowed to raise on malformed input. Anything
# outside this set propagates and is recorded by Atheris as a crash.
_ALLOWED = (JQuantStatsError, ValueError, TypeError, pl.exceptions.PolarsError)


def _frame(fdp: atheris.FuzzedDataProvider) -> pl.DataFrame:
    """Build a small frame of (nullable) float columns from fuzzed bytes."""
    n_rows = fdp.ConsumeIntInRange(0, 16)
    n_cols = fdp.ConsumeIntInRange(1, 4)
    data: dict[str, list[float | None]] = {}
    for c in range(n_cols):
        column: list[float | None] = []
        for _ in range(n_rows):
            # Sprinkle nulls so the interior-forward-fill logic is exercised.
            column.append(None if fdp.ConsumeBool() else fdp.ConsumeFloat())
        data[f"R{c}"] = column
    return pl.DataFrame(data)


def test_one_input(data: bytes) -> None:
    """Exercise interpolate() and Data.from_returns() with a fuzzed frame."""
    fdp = atheris.FuzzedDataProvider(data)
    frame = _frame(fdp)

    # interpolate is contracted to never raise on any frame.
    interpolate(frame)

    # Prepend a monotonically increasing Date column and feed the ingester.
    returns = frame.with_columns(pl.int_range(0, frame.height).alias("Date")).select(["Date", *frame.columns])
    with contextlib.suppress(_ALLOWED):
        Data.from_returns(returns=returns)


def main() -> None:
    """Run the Atheris fuzz loop."""
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
