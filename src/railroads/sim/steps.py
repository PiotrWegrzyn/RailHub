"""Work done in steps: a generator that yields between its steps, so a caller can spread them over
several frames, and returns its result at the end."""

import time
from collections.abc import Generator

type Steps[T] = Generator[None, None, T]


def finish[T](steps: Steps[T]) -> T:
    """Does every step left, one after another; returns the result."""
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def budgeted[T](steps: Steps[T], seconds: float) -> Steps[T]:
    """The same steps run several at a time, pausing only once a run of them has taken
    `seconds`, so a caller spending a frame on each pause spends about that long on it."""
    began = time.perf_counter()
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value
        if time.perf_counter() - began >= seconds:
            yield
            began = time.perf_counter()
