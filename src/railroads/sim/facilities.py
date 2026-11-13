"""What the player can put up beside a station besides raising it: an engine shed, which burns
COAL_A_MONTH of coal a month and lets trains setting off from the station run ENGINE_SHED_SPEED
times as fast, and a sand tower, which uses SAND_A_MONTH of sand a month and lets them pick up
speed in SAND_TOWER_PICKUP of the time. Each takes its good at the start of every month from what
the station holds, gathered or brought, kept for trains or not, and works that month only if
it got all of it."""

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType

from railroads.sim.finance import ENGINE_SHED_PRICE, SAND_TOWER_PRICE
from railroads.sim.goods import Good
from railroads.sim.station_goods import StationGoods

ENGINE_SHED_SPEED = 1.2
SAND_TOWER_PICKUP = 0.5
COAL_A_MONTH = 50.0
SAND_A_MONTH = 15.0


class Facility(Enum):
    ENGINE_SHED = "engine shed"
    SAND_TOWER = "sand tower"


# What each facility uses a month, how much of it, and what it costs to put up.
USES: Mapping[Facility, tuple[Good, float]] = MappingProxyType(
    {
        Facility.ENGINE_SHED: (Good.COAL, COAL_A_MONTH),
        Facility.SAND_TOWER: (Good.SAND, SAND_A_MONTH),
    }
)
PRICES: Mapping[Facility, float] = MappingProxyType(
    {Facility.ENGINE_SHED: ENGINE_SHED_PRICE, Facility.SAND_TOWER: SAND_TOWER_PRICE}
)


def supply(goods: StationGoods, facility: Facility) -> bool:
    """Takes a month of what a facility uses from what a station holds, its stockpile first;
    returns whether there was all of it."""
    good, amount = USES[facility]
    if goods.held(good) < amount - 1e-9:
        return False
    stocked = min(amount, goods.stockpile.get(good, 0.0))
    goods.stockpile[good] = goods.stockpile.get(good, 0.0) - stocked
    if amount - stocked > 0:
        goods.release(good, amount - stocked)
    return True
