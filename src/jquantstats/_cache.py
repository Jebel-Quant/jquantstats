"""Slot-backed caching for frozen, slotted dataclasses.

`Portfolio` is a frozen dataclass with ``slots=True``, so neither
`functools.cached_property` (needs ``__dict__``) nor plain attribute
assignment (frozen) works for memoising derived values. Instead, every cache
lives in an explicitly declared slot field that ``__post_init__`` initialises
to ``None``, and `cached_in_slot` fills it via ``object.__setattr__`` on
first access.

Caching is not thread-safe: concurrent first accesses may compute the value
redundantly, but never produce incorrect results because every thread stores
the same deterministic value.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def cached_in_slot(slot: str) -> Callable[[Callable[[Any], T]], Callable[[Any], T]]:
    """Cache a zero-argument method's result in the slot field named *slot*.

    Apply below ``@property`` so the property getter is the wrapped function:

    ```python
    @property
    @cached_in_slot("_profits_cache")
    def profits(self) -> pl.DataFrame: ...
    ```

    Args:
        slot: Name of the declared slot field used as the cache. The field
            must be initialised to ``None`` before first access (Portfolio
            does this in ``__post_init__``); a ``None`` value means
            "not yet computed".

    Returns:
        A decorator that wraps the getter with read-through caching.
    """

    def decorator(fn: Callable[[Any], T]) -> Callable[[Any], T]:
        """Wrap *fn* with read-through caching against the configured slot."""

        @functools.wraps(fn)
        def wrapper(self: Any) -> T:
            """Return the cached value, computing and storing it on first access."""
            cache = getattr(self, slot, None)
            if cache is None:
                cache = fn(self)
                # Direct write is safe: the owner is a frozen, slotted
                # dataclass that declares every cache field, so
                # object.__setattr__ cannot fail here.
                object.__setattr__(self, slot, cache)
            return cache

        return wrapper

    return decorator
