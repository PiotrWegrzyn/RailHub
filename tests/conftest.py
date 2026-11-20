from collections.abc import Iterator

import pytest

from builders import open_boards
from railroads.sim.tiles import activate

activate(open_boards())


@pytest.fixture(autouse=True)
def flat_boards() -> Iterator[None]:
    """Every test begins on boards that make and want of every good its base amount, earn its base
    price and open every buyer, unless it makes a game or turns the boards itself."""
    activate(open_boards())
    yield
    activate(open_boards())
