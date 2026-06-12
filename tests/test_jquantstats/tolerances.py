"""Shared float-comparison tolerances for the test suite.

One constant per comparison *kind*, so a tolerance says why it was chosen,
not just how big it is:

- ``TOL_FLOAT64`` (1e-12): identities that must hold up to float64 round-off
  only — same algorithm, same data, pure-arithmetic rearrangement.
- ``TOL_COMPOUNDING`` (1e-9): values built from long chains of products/sums
  (compounded NAV, resampled returns) where round-off accumulates beyond
  single-operation epsilon.
- ``TOL_PINNED`` (1e-10): pinned reference values and bound checks (e.g.
  drawdown stays in [0, 1]) where the assertion guards a regression rather
  than an identity.
- ``TOL_PARITY`` (1e-6): cross-implementation agreement — quantstats parity
  tests and alternative-formula cross-checks (e.g. autocorrelation), where
  different algorithms legitimately differ in the last few digits.
- ``TOL_ESTIMATE`` (1e-4): statistical estimators with platform-dependent
  accumulation (EWMA warm-up, kernel densities) — agreement is expected only
  to estimator precision.
"""

TOL_FLOAT64 = 1e-12
TOL_COMPOUNDING = 1e-9
TOL_PINNED = 1e-10
TOL_PARITY = 1e-6
TOL_ESTIMATE = 1e-4
