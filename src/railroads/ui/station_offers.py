"""What a train can take on at a station: the goods made around it and those waiting there."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from railroads.sim.goods import GOOD_YEARS, Good
from railroads.sim.grids import Cell
from railroads.sim.railway import Railway

# What a station has to send on each month, good by good (railroads.sim.station_trade).
type Supplies = Callable[[Cell], Mapping[Good, float]]


@dataclass(frozen=True)
class Offer:
    good: Good
    waiting: float
    made: float


def goods_to_load(railway: Railway, station: Cell, made: Mapping[Good, float]) -> list[Offer]:
    """The goods a train can take on at a station: what is made around it and what waits
    there, gathered or brought by other trains, leaving out goods not yet invented."""
    goods = railway.station_goods(station)
    waiting = {g: goods.stockpile.get(g, 0.0) + goods.brought.get(g, 0.0) for g in Good}
    known = {g for g, a in made.items() if a > 0} | {g for g, a in waiting.items() if a >= 0.5}
    return [
        Offer(good, waiting[good], made.get(good, 0.0))
        for good in Good
        if good in known and GOOD_YEARS.get(good, 0) <= railway.year
    ]
