"""Module helpers and method decorators for statistical computations.

Provides:

- `_drawdown_series` — drawdown series from a returns series.
- `_to_float` — safe Polars aggregation result → Python float.
- `_mean` — series mean with ``None → 0.0`` fallback.
- `_std_is_negligible` — shared "is this std numerically zero?" test for
  mean/std ratio metrics.
- `columnwise_stat` — decorator: apply a metric to every asset column.
- `to_frame` — decorator: build a per-column Polars DataFrame result.

These building blocks are shared across the stats mixin modules
(`_basic`, `_performance`,
`_reporting`, `_rolling`).

Null-return convention
----------------------
- **Scalar metrics** return ``float("nan")`` when the series has no non-null
  observations (use ``_mean`` for the ``None → nan`` conversion).
- **Ratio metrics** return ``float("nan")`` when the denominator is zero
  or indeterminate.
- Use ``_mean`` for the ``None → nan`` conversion rather than
  ``cast(float, ...)``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import timedelta
from functools import wraps
from typing import Any, Concatenate, ParamSpec, TypeVar, cast, overload

import polars as pl

P = ParamSpec("P")
R = TypeVar("R")

# ── Module helpers ────────────────────────────────────────────────────────────


def _drawdown_series(series: pl.Series) -> pl.Series:
    """Compute the drawdown percentage series from a returns series.

    Builds a compound NAV (geometric cumulative product) from the returns
    series and expresses drawdown as the fraction below the running high-water
    mark.  This matches the quantstats convention.

    Args:
        series: A Polars Series of multiplicative daily returns.

    Returns:
        A Polars Float64 Series whose values are in [0, 1].  A value of 0
        means the NAV is at its all-time high; a value of 0.2 means the NAV
        is 20 % below its previous peak.

    Numerical edge cases:
        The high-water mark can only fall below the ``1e-10`` floor when
        *every* NAV value so far is below it, i.e. when the very first
        return is (effectively) -100 %.  Because ``0 <= nav <= hwm`` always
        holds, the result stays within [0, 1] even when the floor is active.
        Note that an exact -100 % first return yields ``nav == hwm == 0``
        and therefore a drawdown of 0: with no baseline, the first
        observation *is* its own high-water mark.  Metrics that need the
        quantstats convention (initial capital of 1.0 as the baseline)
        should use ``_drawdown_with_baseline`` instead.

    Examples:
        >>> import polars as pl
        >>> s = pl.Series([0.0, -0.1, 0.2])
        >>> [round(x, 10) for x in _drawdown_series(s).to_list()]
        [0.0, 0.1, 0.0]
    """
    nav = (1.0 + series.cast(pl.Float64)).cum_prod()
    hwm = nav.cum_max()
    # The floor keeps the division defined after a -100 % return wipes out
    # the NAV; since 0 <= nav <= hwm the ratio stays in [0, 1] regardless.
    hwm_safe = hwm.clip(lower_bound=1e-10)
    return ((hwm - nav) / hwm_safe).clip(lower_bound=0.0)


def _to_float(value: Any) -> float:
    """Safely convert a Polars aggregation result to float.

    Examples:
        >>> _to_float(2.0)
        2.0
        >>> _to_float(None)
        0.0
    """
    if value is None:
        return 0.0
    if isinstance(value, timedelta):
        return value.total_seconds()
    return float(cast(float, value))


def _std_is_negligible(std: float | None, mean: float) -> bool:
    """Return True when a sample standard deviation is numerically zero.

    Mean/std ratios (Sharpe and friends) are meaningless when the measured
    dispersion is smaller than the floating-point rounding noise of the
    inputs: a constant series can produce a tiny non-zero ``std`` purely from
    accumulated rounding, and dividing by it would report an absurdly large
    ratio instead of "no dispersion".  The threshold is 10 machine epsilons
    scaled by the magnitude of the mean, with an absolute floor of one
    epsilon for means at or near zero.  Callers map this case to
    ``float("nan")``.

    Args:
        std: Sample standard deviation, or ``None`` when undefined
            (fewer than two observations).
        mean: Sample mean of the same series, used to scale the threshold.

    Examples:
        >>> _std_is_negligible(None, 1.0)
        True
        >>> _std_is_negligible(0.0, 0.05)
        True
        >>> _std_is_negligible(0.01, 0.05)
        False
    """
    if std is None:
        return True
    eps = sys.float_info.epsilon
    return float(std) <= eps * max(abs(mean), eps) * 10.0


def _mean(series: pl.Series) -> float:
    """Return series mean, or ``float("nan")`` if the series is empty or all-null.

    Use this instead of ``cast(float, series.mean())`` to avoid ``None``
    leaking into arithmetic — consistent with the scalar-metric convention
    that returns ``float("nan")`` when there are no non-null observations.

    Examples:
        >>> import polars as pl
        >>> _mean(pl.Series([1.0, 3.0]))
        2.0
        >>> import math
        >>> math.isnan(_mean(pl.Series([], dtype=pl.Float64)))
        True
    """
    result = series.mean()
    return float(cast(float, result)) if result is not None else float("nan")


# ── Module-level decorators ──────────────────────────────────────────────────


@overload
def columnwise_stat(
    func: Callable[Concatenate[Any, pl.Series, P], R], *, data_attr: str = ...
) -> Callable[Concatenate[Any, P], dict[str, R]]: ...


@overload
def columnwise_stat(
    func: None = ..., *, data_attr: str = ...
) -> Callable[[Callable[Concatenate[Any, pl.Series, P], R]], Callable[Concatenate[Any, P], dict[str, R]]]: ...


def columnwise_stat(
    func: Callable[Concatenate[Any, pl.Series, P], R] | None = None, *, data_attr: str = "_data"
) -> (
    Callable[Concatenate[Any, P], dict[str, R]]
    | Callable[[Callable[Concatenate[Any, pl.Series, P], R]], Callable[Concatenate[Any, P], dict[str, R]]]
):
    """Apply a column-wise statistical function to all numeric columns.

    The decorated method must accept ``(self, series, *args, **kwargs)``; the
    wrapper drops the ``series`` parameter and preserves the remaining
    signature (via ParamSpec), returning ``{column: value}`` with the wrapped
    method's return type as the value type.

    Args:
        func (Callable | None): The function to decorate.
        data_attr: Attribute name that holds the column-wise data object.

    Returns:
        Callable: The decorated function.

    """

    def decorator(
        inner_func: Callable[Concatenate[Any, pl.Series, P], R],
    ) -> Callable[Concatenate[Any, P], dict[str, R]]:
        """Wrap *inner_func* to iterate over the configured data attribute columns."""

        @wraps(inner_func)
        def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> dict[str, R]:
            """Apply *func* to every column and return a ``{column: value}`` mapping."""
            if not hasattr(self, data_attr):
                msg = (
                    f"columnwise_stat requires host object to define '{data_attr}' "
                    f"(missing attribute on {type(self).__name__})."
                )
                raise AttributeError(msg)
            data = getattr(self, data_attr)
            return {col: inner_func(self, series, *args, **kwargs) for col, series in data.items()}

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


@overload
def to_frame(
    func: Callable[Concatenate[Any, pl.Series, P], pl.Series], *, data_attr: str = ...
) -> Callable[Concatenate[Any, P], pl.DataFrame]: ...


@overload
def to_frame(
    func: None = ..., *, data_attr: str = ...
) -> Callable[[Callable[Concatenate[Any, pl.Series, P], pl.Series]], Callable[Concatenate[Any, P], pl.DataFrame]]: ...


def to_frame(
    func: Callable[Concatenate[Any, pl.Series, P], pl.Series] | None = None, *, data_attr: str = "_data"
) -> (
    Callable[Concatenate[Any, P], pl.DataFrame]
    | Callable[[Callable[Concatenate[Any, pl.Series, P], pl.Series]], Callable[Concatenate[Any, P], pl.DataFrame]]
):
    """Apply per-column expressions and evaluates with .with_columns(...).

    The decorated method must accept ``(self, series, *args, **kwargs)`` and
    return a per-column Polars Series; the wrapper drops the ``series``
    parameter and preserves the remaining signature (via ParamSpec).

    Args:
        func (Callable | None): The function to decorate.
        data_attr: Attribute name that holds the column-wise data object.

    Returns:
        Callable: The decorated function.

    """

    def decorator(
        inner_func: Callable[Concatenate[Any, pl.Series, P], pl.Series],
    ) -> Callable[Concatenate[Any, P], pl.DataFrame]:
        """Wrap *inner_func* to build a per-column frame from the configured data attribute."""

        @wraps(inner_func)
        def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> pl.DataFrame:
            """Apply *func* per column and return the result as a Polars DataFrame."""
            if not hasattr(self, data_attr):
                msg = (
                    f"to_frame requires host object to define '{data_attr}' "
                    f"(missing attribute on {type(self).__name__})."
                )
                raise AttributeError(msg)
            data = getattr(self, data_attr)
            return cast(pl.DataFrame, self.all).select(
                [pl.col(name) for name in data.date_col]
                + [inner_func(self, series, *args, **kwargs).alias(col) for col, series in data.items()]
            )

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
