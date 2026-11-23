"""What the player can put up beside a station besides raising it, each using a good a month:

- an engine shed burns COAL_A_MONTH of coal and lets trains setting off from the station run
  ENGINE_SHED_SPEED times as fast;
- a sand tower uses SAND_A_MONTH of sand and lets them pick up speed in SAND_TOWER_PICKUP of the
  time;
- a post office, a refreshment room, a market hall and cattle pens make what is delivered at the
  station pay more (PAYS_MORE): mail, passengers, food and livestock;
- a warehouse adds WAREHOUSE_ROOM to the room for every kind of freight.

Each takes its good at the start of every month from what the station holds, gathered or brought,
and works that month only if it got all of it."""

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType

from railroads.sim.amounts import EPSILON
from railroads.sim.finance import (
    CATTLE_PENS_PRICE,
    ENGINE_SHED_PRICE,
    MARKET_HALL_PRICE,
    POST_OFFICE_PRICE,
    REFRESHMENT_ROOM_PRICE,
    SAND_TOWER_PRICE,
    WAREHOUSE_PRICE,
)
from railroads.sim.goods import FOODS, Good
from railroads.sim.station_goods import StationGoods

ENGINE_SHED_SPEED = 1.2
SAND_TOWER_PICKUP = 0.5
COAL_A_MONTH = 50.0
SAND_A_MONTH = 15.0
WAREHOUSE_ROOM = 500.0


class Facility(Enum):
    ENGINE_SHED = "engine shed"
    SAND_TOWER = "sand tower"
    POST_OFFICE = "post office"
    REFRESHMENT_ROOM = "refreshment room"
    MARKET_HALL = "market hall"
    CATTLE_PENS = "cattle pens"
    WAREHOUSE = "warehouse"


# What each facility uses a month, how much of it, and what it costs to put up.
USES: Mapping[Facility, tuple[Good, float]] = MappingProxyType(
    {
        Facility.ENGINE_SHED: (Good.COAL, COAL_A_MONTH),
        Facility.SAND_TOWER: (Good.SAND, SAND_A_MONTH),
        Facility.POST_OFFICE: (Good.PAPER, 10.0),
        Facility.REFRESHMENT_ROOM: (Good.BEER, 30.0),
        Facility.MARKET_HALL: (Good.PLANKS, 20.0),
        Facility.CATTLE_PENS: (Good.WHEAT, 30.0),
        Facility.WAREHOUSE: (Good.BRICKS, 20.0),
    }
)
PRICES: Mapping[Facility, float] = MappingProxyType(
    {
        Facility.ENGINE_SHED: ENGINE_SHED_PRICE,
        Facility.SAND_TOWER: SAND_TOWER_PRICE,
        Facility.POST_OFFICE: POST_OFFICE_PRICE,
        Facility.REFRESHMENT_ROOM: REFRESHMENT_ROOM_PRICE,
        Facility.MARKET_HALL: MARKET_HALL_PRICE,
        Facility.CATTLE_PENS: CATTLE_PENS_PRICE,
        Facility.WAREHOUSE: WAREHOUSE_PRICE,
    }
)

# The goods each facility makes pay more where it works, and how many times as much.
PAYS_MORE: Mapping[Facility, tuple[frozenset[Good], float]] = MappingProxyType(
    {
        Facility.POST_OFFICE: (frozenset({Good.MAIL}), 1.5),
        Facility.REFRESHMENT_ROOM: (frozenset({Good.PASSENGERS}), 1.25),
        Facility.MARKET_HALL: (frozenset(FOODS), 1.25),
        Facility.CATTLE_PENS: (
            frozenset({Good.CATTLE, Good.PIGS, Good.CHICKENS, Good.SHEEP}),
            1.3,
        ),
    }
)


def pay_factor(working: set[Facility], good: Good) -> float:
    """How many times as much a good delivered at a station pays with these facilities working."""
    factor = 1.0
    for facility in working:
        goods, more = PAYS_MORE.get(facility, (frozenset(), 1.0))
        if good in goods:
            factor *= more
    return factor


def supply(goods: StationGoods, facility: Facility) -> bool:
    """Takes a month of what a facility uses from what a station holds, its stockpile first;
    returns whether there was all of it."""
    good, amount = USES[facility]
    if goods.held(good) < amount - EPSILON:
        return False
    stocked = min(amount, goods.stockpile.get(good, 0.0))
    goods.stockpile[good] = goods.stockpile.get(good, 0.0) - stocked
    if amount - stocked > 0:
        goods.release(good, amount - stocked)
    return True
