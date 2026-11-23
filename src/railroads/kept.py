"""Values kept between frames and worked out again only when what they were worked out from
changes (Kept, KeptFor) or once they are a while old (Fresh).

A key is compared element by element when it is a tuple: plain values (numbers, strings, enums,
None) by equality, anything else, such as a month's trade, by being the very same object, since
such objects are replaced rather than changed in place."""

import time
import weakref
from collections.abc import Callable, Hashable, Iterable
from enum import Enum

_PLAIN = (int, float, str, bool, bytes, Enum, type(None))


def same(first: object, second: object) -> bool:
    """Whether two keys stand for the same thing (see the module's docstring)."""
    if isinstance(first, tuple) and isinstance(second, tuple):
        return len(first) == len(second) and all(map(same, first, second, strict=True))
    if isinstance(first, _PLAIN) and isinstance(second, _PLAIN):
        return first == second
    return first is second


class Kept[V]:
    """One value, worked out again whenever the key it was worked out for changes."""

    __slots__ = ("_full", "_key", "_value")

    def __init__(self) -> None:
        self._full = False

    def get(self, key: object, make: Callable[[], V]) -> V:
        if not self._full or not same(self._key, key):
            self._value = make()
            self._key = key
            self._full = True
        return self._value


class KeptFor[O: Hashable, V]:
    """A Kept for each of many owners, such as stations, or game objects held weakly so that
    a game thrown away takes its values with it."""

    def __init__(self, *, weak: bool = False) -> None:
        self._kept: dict[O, Kept[V]] | weakref.WeakKeyDictionary[O, Kept[V]] = (
            weakref.WeakKeyDictionary() if weak else {}
        )

    def get(self, owner: O, key: object, make: Callable[[], V]) -> V:
        kept = self._kept.get(owner)
        if kept is None:
            kept = self._kept[owner] = Kept()
        return kept.get(key, make)

    def keep_only(self, owners: Iterable[O]) -> None:
        """Forgets the values of every owner but these."""
        wanted = set(owners)
        for owner in [owner for owner in self._kept if owner not in wanted]:
            del self._kept[owner]


class Fresh[O: Hashable, V]:
    """Values by owner, each worked out again once it is `seconds` old or its key changes."""

    def __init__(self, seconds: float, now: Callable[[], float] = time.monotonic) -> None:
        self.seconds = seconds
        self._now = now
        self._kept: dict[O, tuple[float, object, V]] = {}

    def get(self, owner: O, make: Callable[[], V], key: object = None) -> V:
        now = self._now()
        kept = self._kept.get(owner)
        if kept is None or now - kept[0] > self.seconds or not same(kept[1], key):
            kept = self._kept[owner] = (now, key, make())
        return kept[2]

    def forget_old(self) -> None:
        """Forgets every value too old to be given again, such as those of trains sold."""
        now = self._now()
        for owner in [o for o, (at, _, _) in self._kept.items() if now - at > self.seconds]:
            del self._kept[owner]
