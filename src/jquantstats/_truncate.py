"""Shared bound resolution for `Data.truncate` and `Portfolio.truncate`.

Both entry points accept the same bound types and must agree on what they mean,
so the classification lives here rather than being written twice.

The rule is: **the bound type picks the axis, the index type says what is
legal.** An ``int`` bound is a 0-based row index, a ``date``/``datetime``/ISO-8601
string is a position on the temporal axis, and the two cannot be mixed. An
object with an integer index accepts row indices only.

Routing on the bound rather than on the index dtype is what makes an ``int``
bound meaningful on dated data: the previous behaviour compared the row number
against the date column, which Polars evaluated to an all-true mask, so
``truncate(start=10)`` silently returned every row.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from .exceptions import IntegerIndexBoundError, InvalidTruncateBoundError, MixedTruncateBoundsError

#: What the resolved bounds address: nothing, row indices, or the temporal axis.
Mode = Literal["none", "rows", "dates"]


def _parse_iso(param: str, value: str) -> date | datetime:
    """Parse an ISO-8601 date or datetime string.

    Args:
        param: Name of the parameter being parsed, for the error message.
        value: The candidate string.

    Returns:
        A `datetime.date` for a plain date, a `datetime.datetime` when the
        string carries a time component.

    Raises:
        InvalidTruncateBoundError: If *value* is not ISO-8601.
    """
    try:
        return date.fromisoformat(value)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise InvalidTruncateBoundError(param, value) from None


def _classify(param: str, value: Any) -> tuple[Mode, Any]:
    """Classify one bound and coerce it to the form the comparison needs.

    ``bool`` is rejected rather than accepted as an ``int``: ``truncate(start=True)``
    is far more likely to be a mistake than a request for row 1.

    Args:
        param: Name of the parameter, for the error message.
        value: The supplied bound.

    Returns:
        A ``(mode, coerced)`` pair; mode is ``"none"`` when *value* is ``None``.

    Raises:
        InvalidTruncateBoundError: If *value* is of an unsupported type, or is a
            string that is not ISO-8601.
    """
    if value is None:
        return "none", None
    if isinstance(value, bool):
        raise InvalidTruncateBoundError(param, value)
    if isinstance(value, int):
        return "rows", value
    if isinstance(value, (date, datetime)):
        return "dates", value
    if isinstance(value, str):
        return "dates", _parse_iso(param, value)
    raise InvalidTruncateBoundError(param, value)


def _resolve_row_bounds(start: Any, end: Any) -> tuple[Mode, Any, Any]:
    """Resolve bounds against an integer index, where only row numbers are legal.

    Split out of `resolve_bounds` so the non-temporal rule reads as one thing:
    a row number is the only meaning a bound can carry here, so anything else is
    reported against that expectation rather than being classified further.

    ``bool`` is rejected for the reason `_classify` rejects it — ``start=True``
    is far more likely to be a mistake than a request for row 1 — but the error
    differs: on an integer index the complaint is the index type, not the bound
    type, so `IntegerIndexBoundError` names what was expected.

    Args:
        start: Inclusive lower bound, or ``None``.
        end: Inclusive upper bound, or ``None``.

    Returns:
        ``(mode, start, end)`` with the bounds unchanged — a row index needs no
        coercion. ``mode`` is ``"none"`` when both bounds are ``None``, else
        ``"rows"``.

    Raises:
        IntegerIndexBoundError: If either bound is not an ``int``.
    """
    for param, value in (("start", start), ("end", end)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise IntegerIndexBoundError(param, type(value).__name__)
    return ("none" if start is None and end is None else "rows"), start, end


def _reject_mixed_modes(start_mode: Mode, end_mode: Mode) -> None:
    """Reject a row index paired with a date, in either order.

    The two checks are mirror images, and the argument order to
    `MixedTruncateBoundsError` is not symmetric: the first name is the bound
    that disagrees with the axis already established by the other, so the
    message reads as a complaint about the newcomer rather than about the pair.

    Args:
        start_mode: The mode `_classify` assigned to *start*.
        end_mode: The mode `_classify` assigned to *end*.

    Raises:
        MixedTruncateBoundsError: If one bound is a row index and the other a date.
    """
    if start_mode == "rows" and end_mode == "dates":
        raise MixedTruncateBoundsError("start", "end")
    if start_mode == "dates" and end_mode == "rows":
        raise MixedTruncateBoundsError("end", "start")


def resolve_bounds(start: Any, end: Any, *, temporal: bool) -> tuple[Mode, Any, Any]:
    """Decide which axis *start* and *end* address, and coerce them for it.

    Args:
        start: Inclusive lower bound, or ``None``.
        end: Inclusive upper bound, or ``None``.
        temporal: Whether the object has a temporal index.  When ``False`` the
            only legal bound is an ``int`` row index.

    Returns:
        ``(mode, start, end)``.  ``mode`` is ``"none"`` when both bounds are
        ``None`` (the caller should return the object unchanged), ``"rows"`` for
        positional slicing, or ``"dates"`` for a comparison against the date
        column.  The returned bounds are coerced — ISO strings become
        `datetime.date` or `datetime.datetime` values.

    Raises:
        IntegerIndexBoundError: If the index is not temporal and a bound is not
            an ``int``.
        InvalidTruncateBoundError: If a bound is of an unsupported type, or is a
            string that is not ISO-8601.
        MixedTruncateBoundsError: If one bound is a row index and the other a date.
    """
    if not temporal:
        return _resolve_row_bounds(start, end)

    start_mode, start_value = _classify("start", start)
    end_mode, end_value = _classify("end", end)
    _reject_mixed_modes(start_mode, end_mode)

    # With the modes agreed, whichever bound is set names the axis; when only one
    # is given the other stays "none" and contributes nothing.
    mode: Mode = start_mode if start_mode != "none" else end_mode
    return mode, start_value, end_value
