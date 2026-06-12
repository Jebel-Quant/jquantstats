"""Mutation-kill tests for the `cached_in_slot` decorator in `jquantstats._cache`.

Each test targets a specific surviving mutmut mutant:

- removal of ``@functools.wraps(fn)``  -> metadata test,
- ``TypeVar("T")`` -> ``TypeVar("XXTXX")`` and ``T = None``  -> signature test.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

from jquantstats._cache import cached_in_slot


@dataclass(frozen=True, slots=True)
class _Box:
    """Minimal frozen, slotted dataclass mirroring how Portfolio uses `cached_in_slot`."""

    value: int
    calls: list[int] = field(default_factory=list)
    _double_cache: int | None = None

    @cached_in_slot("_double_cache")
    def double(self) -> int:
        """Return twice the stored value."""
        self.calls.append(1)
        return 2 * self.value


def test_cached_in_slot_preserves_wrapped_metadata():
    """The wrapper must carry the wrapped function's name and docstring via functools.wraps.

    Kills the mutant that removes ``@functools.wraps(fn)``: without it the
    decorated method would be named ``wrapper`` and carry the wrapper's docstring.
    """
    assert _Box.double.__name__ == "double"
    assert _Box.double.__doc__ == "Return twice the stored value."
    assert _Box.double.__wrapped__.__name__ == "double"


def test_cached_in_slot_computes_once_and_returns_cached_value():
    """The wrapped method must compute exactly once and serve the slot cache afterwards."""
    box = _Box(value=3)
    assert box._double_cache is None
    assert box.double() == 6
    assert box._double_cache == 6
    assert box.double() == 6
    assert len(box.calls) == 1


def test_cached_in_slot_signature_uses_typevar_named_t():
    """The decorator's return hint must stay generic in a TypeVar named ``T``.

    Kills the mutants ``TypeVar("T") -> TypeVar("XXTXX")`` and ``T = None``:
    both change the resolved type hints of `cached_in_slot`.
    """
    hints = typing.get_type_hints(cached_in_slot)
    decorator_hint = hints["return"]  # Callable[[Callable[[Any], T]], Callable[[Any], T]]
    returned_callable = typing.get_args(decorator_hint)[-1]  # Callable[[Any], T]
    type_var = typing.get_args(returned_callable)[-1]
    assert isinstance(type_var, typing.TypeVar)
    assert type_var.__name__ == "T"
