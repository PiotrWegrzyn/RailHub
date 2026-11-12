"""Work done in steps: a generator that yields between its steps, so a caller can spread them over
several frames, and returns its result at the end."""

from collections.abc import Generator

type Steps[T] = Generator[None, None, T]


def finish[T](steps: Steps[T]) -> T:
    """Does every step left, one after another; returns the result."""
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value
