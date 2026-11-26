from enum import Enum


class Keyed(Enum):
    """An enum hashed by identity in C rather than by its name in Python as Enum is. Its members
    are singletons compared by identity, so every dict and set keyed by them works the same, only
    faster: the month's trade looks goods, needs and kinds up tens of thousands of times."""

    __hash__ = object.__hash__
