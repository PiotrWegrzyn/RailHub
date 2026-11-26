"""What the player can put up at a station besides raising it. An engine shed, a roundhouse round
a turntable with track of its own that track is laid to as to a station, fills a cell beside the
station or one of its platforms, and cattle pens take a little room in one (railroads.sim.constructions.ANNEX_SPACE);
the others stand in the station itself and take no room on the map:

- an engine shed lets trains setting off from the station run ENGINE_SHED_SPEED times as fast,
  and trains waiting there for a full load the station holds too little to fill wait in it, on
  its own track (railroads.sim.railway);
- a sand tower lets them pick up speed in SAND_TOWER_PICKUP of the time;
- a post office, a refreshment room, a market hall and cattle pens make what is delivered at the
  station pay more (PAYS_MORE): mail, passengers, food and livestock;
- a warehouse adds WAREHOUSE_ROOM to the room for every kind of freight.

None costs anything to put up, but each uses a good every month (USES) and costs what that much
of it fetches on the market (railroads.sim.railway.Railway.market_price): the share of it the
station holds, gathered or brought, it takes from there at DELIVERED_SHARE of that price, the
rest it buys at the full price. Each works whatever it got. A station can have one of each, from
the level each asks (LEVEL)."""

from collections.abc import Callable, Mapping
from types import MappingProxyType

from railroads.sim.constructions import ANNEX_SPACE, Facility
from railroads.sim.goods import FOODS, Good
from railroads.sim.station_goods import StationGoods

__all__ = ["Facility"]

ENGINE_SHED_SPEED = 1.2
SAND_TOWER_PICKUP = 0.5
WAREHOUSE_ROOM = 500.0
REACH = 1
DELIVERED_SHARE = 0.5


# The facilities that stand in a cell beside their station rather than in the station itself.
PLACED = frozenset(ANNEX_SPACE)
# What each facility uses a month, and how much of it.
USES: Mapping[Facility, tuple[Good, float]] = MappingProxyType(
    {
        Facility.ENGINE_SHED: (Good.COAL, 50.0),
        Facility.SAND_TOWER: (Good.SAND, 15.0),
        Facility.POST_OFFICE: (Good.PAPER, 10.0),
        Facility.REFRESHMENT_ROOM: (Good.BEER, 5.0),
        Facility.MARKET_HALL: (Good.PLANKS, 20.0),
        Facility.CATTLE_PENS: (Good.WHEAT, 30.0),
        Facility.WAREHOUSE: (Good.BRICKS, 20.0),
    }
)
# The level a station must be raised to before each facility can go up beside it.
LEVEL: Mapping[Facility, int] = MappingProxyType(
    {
        Facility.ENGINE_SHED: 3,
        Facility.SAND_TOWER: 1,
        Facility.POST_OFFICE: 1,
        Facility.REFRESHMENT_ROOM: 1,
        Facility.MARKET_HALL: 1,
        Facility.CATTLE_PENS: 1,
        Facility.WAREHOUSE: 1,
    }
)

# The goods each facility makes pay more where it stands, and how many times as much.
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


def pay_factor(facilities: set[Facility], good: Good) -> float:
    """How many times as much a good delivered at a station pays with these facilities."""
    factor = 1.0
    for facility in facilities:
        goods, more = PAYS_MORE.get(facility, (frozenset(), 1.0))
        if good in goods:
            factor *= more
    return factor


def monthly_cost(facility: Facility, held: float, price: float) -> float:
    """What a month of a facility costs where its station holds `held` of what it uses, a unit
    of which fetches `price` on the market."""
    _, amount = USES[facility]
    taken = min(amount, max(held, 0.0))
    return (taken * DELIVERED_SHARE + amount - taken) * price


def supply(goods: StationGoods, facility: Facility, price: Callable[[Good], float]) -> float:
    """Takes a month of what a facility uses from what a station holds, its stockpile first, as
    much as there is, and returns what the month costs (monthly_cost) at `price`."""
    good, amount = USES[facility]
    held = goods.held(good)
    cost = monthly_cost(facility, held, price(good))
    taken = min(amount, held)
    stocked = min(taken, goods.stockpile.get(good, 0.0))
    if stocked > 0:
        goods.stockpile[good] -= stocked
    if taken - stocked > 0:
        goods.release(good, taken - stocked)
    return cost
