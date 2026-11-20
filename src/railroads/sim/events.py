"""History: what the world's years bring, and what each changes in the game. Nothing here
depends on one game; the buildings and the news it makes in one are in
railroads.sim.happenings.

What buildings make and want grows with the supply and demand boards (railroads.sim.tiles,
rise_of): the buildings people use (RISERS: all that use goods up but power plants and railway
works) want as much more of each need as its tiles' demand gives, processors and makers of what
the land gives take in as much more as their supply gives.

EVENTS are dated, each for its `years` or, at 0, for good. While one lasts, buildings that use
goods up want so many times as much of each need in its `wants` (of every other need `rest`
times as much, for RISERS only), in homes only if it is `homes_only`, and of a need of one kind
of building its `kind_wants`; a kind of building wants some needs only while one lasts
(`adds`), as ports want guns in a war, which boost what they make. Goods can be made less of
(`makes`) and earn more (`prices`, the diesel fuel trains burn too), passengers and mail come
and go more or less (`travels`), homes want more electricity and it costs more
(`electricity_use`, `electricity_price`), and passengers getting off in the town of St Paul's
earn `pilgrims` times as much for PILGRIM_MONTHS from the event's news, which comes in its
`month`. One `east_only` happens only on maps in the old Eastern Bloc
(EASTERN_BLOC). A war (`war`) also makes the military grow WAR_GROWTH times as fast.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType

from railroads.sim import tiles
from railroads.sim.constructions import BuildingKind
from railroads.sim.geo import Bounds
from railroads.sim.goods import HOMES, MAKES, NEEDS, RECIPES, Good, Need
from railroads.sim.year_caches import year_cached

NOT_RISING = frozenset(
    {
        BuildingKind.POWER_PLANT,
        BuildingKind.DIESEL_POWER_PLANT,
        BuildingKind.GAS_POWER_PLANT,
        BuildingKind.RAILWAY_WORKS,
    }
)
WAR_GROWTH = 11
PILGRIM_MONTHS = 12
# West, east, south and north of the old Eastern Bloc, which a map's middle must fall in.
EASTERN_BLOC = (12.0, 60.0, 42.0, 60.0)


def uses_up(kind: BuildingKind) -> bool:
    """Whether a kind of building uses goods up: it needs some and neither processes nor makes
    goods."""
    return kind in NEEDS and kind not in RECIPES and kind not in MAKES


RISERS = frozenset(kind for kind in BuildingKind if uses_up(kind) and kind not in NOT_RISING)


@dataclass(frozen=True)
class Event:
    year: int
    title: str
    story: str
    # How many years it lasts; 0 for good.
    years: int = 0
    wants: Mapping[Need, float] = field(default_factory=dict)
    rest: float = 1.0
    homes_only: bool = False
    kind_wants: Mapping[tuple[BuildingKind, Need], float] = field(default_factory=dict)
    adds: Mapping[tuple[BuildingKind, Need], float] = field(default_factory=dict)
    makes: Mapping[Good, float] = field(default_factory=dict)
    prices: Mapping[Good, float] = field(default_factory=dict)
    travels: Mapping[Good, float] = field(default_factory=dict)
    electricity_use: float = 1.0
    electricity_price: float = 1.0
    # The month its news comes in, January as 0; it changes nothing by year.
    month: int = 0
    # How many times as much passengers getting off in the town of St Paul's earn for
    # PILGRIM_MONTHS from its news.
    pilgrims: float = 1.0
    east_only: bool = False
    war: bool = False
    # Skyscrapers that fall, in the biggest towns.
    towers_fall: int = 0
    # Whether its news stops the game until the player has read it.
    big: bool = False

    def lasts(self, year: int) -> bool:
        if self.years == 0:
            return year >= self.year
        return self.year <= year < self.year + self.years

    def factor(self, kind: BuildingKind, need: Need) -> float:
        """How many times as much of a need a kind of building wants while it lasts."""
        named = self.kind_wants.get((kind, need))
        if named is not None:
            return named
        if not uses_up(kind) or (self.homes_only and kind not in HOMES):
            return 1.0
        return self.wants.get(need, self.rest if kind in RISERS else 1.0)


EVENTS = (
    Event(
        1854,
        "The telegraph",
        "Telegraph wires reach every town, and news travels in minutes. Everyone wants a paper.",
        wants={Need.NEWSPAPERS: 2.0},
    ),
    Event(
        1873,
        "The Long Depression",
        "Banks fail from Vienna to New York, and people put off buying anything they can do "
        "without.",
        years=6,
        wants={Need.LUXURIES: 0.7, Need.FURNITURE: 0.7, Need.WINE: 0.7},
        big=True,
    ),
    Event(
        1882,
        "Electric light",
        "The first power stations light streets and homes. Homes no longer want wax for "
        "candles; churches, theatres and palaces still burn them.",
        wants={Need.WAX: 0.0},
        homes_only=True,
        electricity_use=1.2,
    ),
    Event(
        1914,
        "The Great War",
        "War breaks out across Europe. New garrisons go up, the army grows fast, gun factories "
        "open and ports ship arms.",
        years=5,
        wants={Need.GUNS: 2.0, Need.STEEL: 2.0, Need.LUXURIES: 0.6, Need.WINE: 0.6},
        adds={(BuildingKind.PORT, Need.GUNS): 20.0},
        travels={Good.PASSENGERS: 1.5},
        war=True,
        big=True,
    ),
    Event(
        1920,
        "Radio",
        "Broadcasting begins. People hear the news at home, and buy fewer papers.",
        wants={Need.NEWSPAPERS: 0.85},
        electricity_use=1.2,
    ),
    Event(
        1929,
        "The Great Depression",
        "The stock market crashes. People still buy food, common goods and fuel to keep warm, "
        "but little else.",
        years=4,
        wants={Need.FOOD: 1.0, Need.HEATING: 1.0, Need.COMMON: 1.0},
        rest=0.7,
        big=True,
    ),
    Event(
        1939,
        "The Second World War",
        "War again. New garrisons go up, the army grows fast, gun factories open and ports ship "
        "arms.",
        years=7,
        wants={
            Need.GUNS: 2.0,
            Need.TANKS: 2.0,
            Need.PLANES: 2.0,
            Need.STEEL: 2.0,
            Need.LUXURIES: 0.5,
            Need.WINE: 0.5,
            Need.CARS: 0.5,
        },
        adds={(BuildingKind.PORT, Need.GUNS): 40.0},
        travels={Good.PASSENGERS: 1.5},
        war=True,
        big=True,
    ),
    Event(
        1955,
        "Television",
        "A television in every living room: fewer papers, more appliances, and power stations "
        "working hard every evening.",
        wants={Need.NEWSPAPERS: 0.5, Need.APPLIANCES: 1.3},
        electricity_use=1.5,
        electricity_price=1.25,
    ),
    Event(
        1973,
        "The oil crisis",
        "Oil producers cut supply. Oil wells pump half as much, and oil and diesel fuel cost "
        "twice as much.",
        years=2,
        makes={Good.OIL: 0.5},
        prices={Good.OIL: 2.0, Good.FUEL: 2.0},
        big=True,
    ),
    Event(
        1981,
        "The Pope is shot",
        "At 17:17 on 13 May Pope John Paul II is shot in St Peter's Square, and survives. For a "
        "year pilgrims crowd the town of St Paul's to pray for him.",
        month=4,
        pilgrims=4.0,
    ),
    Event(
        1989,
        "The Iron Curtain falls",
        "Communism ends and the market opens: everyone wants a car, a television and a washing "
        "machine.",
        wants={Need.CARS: 1.3, Need.ELECTRONICS: 1.3, Need.APPLIANCES: 1.3},
        east_only=True,
    ),
    Event(
        1995,
        "The internet",
        "Homes go online. People read the news on screens and buy fewer papers.",
        wants={Need.NEWSPAPERS: 0.7, Need.ELECTRONICS: 1.3},
    ),
    Event(
        2001,
        "September 11",
        "At 8:46 on 11 September terrorists fly the first of two airliners into twin towers. Two "
        "skyscrapers of the biggest towns fall.",
        month=8,
        towers_fall=2,
        big=True,
    ),
    Event(
        2005,
        "John Paul II dies",
        "At 21:37 on 2 April Pope John Paul II dies. For a year pilgrims crowd the town of "
        "St Paul's to pray for him.",
        month=3,
        pilgrims=6.0,
    ),
    Event(
        2007,
        "Smartphones",
        "Everyone carries the news in their pocket. Nobody buys a paper any more.",
        wants={Need.NEWSPAPERS: 0.0, Need.ELECTRONICS: 1.5},
    ),
    Event(
        2008,
        "The financial crisis",
        "Banks collapse. Homes eat half as much and buy a fifth less of everything else.",
        years=2,
        wants={Need.FOOD: 0.5},
        rest=0.8,
        homes_only=True,
        big=True,
    ),
    Event(
        2020,
        "The pandemic",
        "A new virus spreads. People stay at home and send parcels instead.",
        years=3,
        travels={Good.PASSENGERS: 0.2, Good.MAIL: 1.5},
        big=True,
    ),
    Event(
        2030,
        "Electric cars",
        "Electric cars take over the roads: less diesel fuel, more electronics.",
        wants={Need.FUEL: 0.6, Need.ELECTRONICS: 1.2},
    ),
    Event(
        2040,
        "Goodbye to coal",
        "Coal power stations are run down; each burns half as much coal.",
        kind_wants={(BuildingKind.POWER_PLANT, Need.COAL): 0.5},
    ),
)


def in_eastern_bloc(bounds: Bounds) -> bool:
    """Whether a map's middle lies in the old Eastern Bloc."""
    west, east, south, north = EASTERN_BLOC
    lon, lat = bounds.center
    return west <= lon <= east and south <= lat <= north


@cache
def happening(year: int, east: bool = False) -> tuple[Event, ...]:
    """The events that last in a year, on a map in the east or not."""
    return tuple(e for e in EVENTS if e.lasts(year) and (east or not e.east_only))


@year_cached
def rise_of(kind: BuildingKind, need: Need, year: int) -> float:
    """How many times as much of a need a kind of building wants in a year as its NEEDS say, by the
    boards (railroads.sim.tiles): one people use as much more as the demand for the need's goods,
    a processor as much more as the supply of what it makes, and one making what the land gives
    as much more as the supply of what it makes."""
    boards = tiles.of_year(year)
    if kind in RISERS:
        return boards.want_factor(need)
    recipe = RECIPES.get(kind)
    if recipe is not None:
        return boards.make_factor(recipe.makes)
    made = MAKES.get(kind)
    if made:
        return sum(boards.make_factor(good) for good in made) / len(made)
    return 1.0


@year_cached
def want_factor(kind: BuildingKind, need: Need, year: int, east: bool = False) -> float:
    """How many times as much of a need a kind of building wants in a year as its NEEDS say."""
    factor = rise_of(kind, need, year)
    for event in happening(year, east):
        factor *= event.factor(kind, need)
    return factor


@cache
def added_needs(kind: BuildingKind, year: int) -> Mapping[Need, float]:
    """What a kind of building wants in a year only while an event lasts."""
    return MappingProxyType(
        {
            need: amount
            for event in happening(year)
            for (adding, need), amount in event.adds.items()
            if adding == kind
        }
    )


def _product(year: int, east: bool, of: str) -> float:
    factor = 1.0
    for event in happening(year, east):
        factor *= getattr(event, of)
    return factor


@year_cached
def make_factor(good: Good, year: int) -> float:
    """How many times as much of a good buildings make in a year: as much as its supply tile gives
    (railroads.sim.tiles), and what events make of it."""
    factor = tiles.of_year(year).make_factor(good)
    for event in happening(year):
        factor *= event.makes.get(good, 1.0)
    return factor


@cache
def price_factor(good: Good, year: int) -> float:
    """How many times as much a good earns in a year."""
    factor = 1.0
    for event in happening(year):
        factor *= event.prices.get(good, 1.0)
    return factor


@year_cached
def travel_factor(good: Good, year: int) -> float:
    """How many times as many passengers or as much mail come and go in a year: as many as their
    supply tile gives (railroads.sim.tiles), and what events make of it."""
    factor = tiles.of_year(year).make_factor(good)
    for event in happening(year):
        factor *= event.travels.get(good, 1.0)
    return factor


@cache
def electricity_use(year: int) -> float:
    """How many times as much electricity homes and factories use in a year."""
    return _product(year, False, "electricity_use")


@cache
def electricity_price(year: int) -> float:
    """How many times as much electricity costs in a year."""
    return _product(year, False, "electricity_price")


@cache
def military_growth(year: int) -> int:
    """How many times as fast the military grows in a year."""
    return WAR_GROWTH if any(event.war for event in happening(year)) else 1
