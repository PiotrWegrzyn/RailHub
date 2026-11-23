"""Caches of what a year's supply and demand boards decide (railroads.sim.tiles), forgotten
together whenever the boards change."""

from collections.abc import Callable
from functools import cache

_CLEARS: list[Callable[[], None]] = []
_forgotten = 0


def year_cached[F: Callable[..., object]](fn: F) -> F:
    cached = cache(fn)
    _CLEARS.append(cached.cache_clear)
    return cached  # type: ignore[return-value]


def forget() -> None:
    global _forgotten
    _forgotten += 1
    for clear in _CLEARS:
        clear()


def boards_changed() -> int:
    """How many times the boards have changed, for a cache kept elsewhere to know it is stale."""
    return _forgotten


def on_forget(clear: Callable[[], None]) -> None:
    """Has forget call this too."""
    _CLEARS.append(clear)
