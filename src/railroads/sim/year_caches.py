"""Caches of what a year's supply and demand boards decide (railroads.sim.tiles), forgotten
together whenever the boards change."""

from collections.abc import Callable
from functools import cache

_CLEARS: list[Callable[[], None]] = []


def year_cached[F: Callable[..., object]](fn: F) -> F:
    cached = cache(fn)
    _CLEARS.append(cached.cache_clear)
    return cached  # type: ignore[return-value]


def forget() -> None:
    for clear in _CLEARS:
        clear()


def on_forget(clear: Callable[[], None]) -> None:
    """Has forget call this too."""
    _CLEARS.append(clear)
