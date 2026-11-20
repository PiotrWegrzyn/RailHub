from collections.abc import Iterator

import pytest

from railroads.sim.tiles import Tiles, activate

activate(Tiles())


@pytest.fixture(autouse=True)
def flat_boards() -> Iterator[None]:
    """Every test begins on boards that make and want of every good its base amount, and earn its
    base price, unless it makes a game or turns the boards itself."""
    activate(Tiles())
    yield
    activate(Tiles())
