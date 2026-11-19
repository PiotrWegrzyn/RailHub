"""The trains on the railway (railroads.sim.railway runs them): the engines that can be bought, the
cars, each carrying one good up to its car_size, the stops a train calls at with the cars it pulls
from each, and the train itself, slowed by the weight of what it carries against the cars its
engine is built to pull. Every engine costs a share of its price to run each month, by what drives
it (RUNNING_SHARE). A train sets off from a stand and picks up speed evenly until it runs at full
speed after its engine's pickup time, the Planet's twelve seconds at normal speed, a light
engine's a few and a heavy or express engine's more. An express always leads its coaches with a dining car,
which carries nothing but weighs and costs like any other car.

Freight remembers where it came from as parcels: how much from each station."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from railroads.sim.finance import TRAIN_PRICE
from railroads.sim.goods import General, Good

# What a stop asks for a car of: one good, or a whole kind of goods.
type Cargo = Good | General
from railroads.sim.grids import Cell

MAX_CARS = 5


# A full car weighs this much more than an empty one, which counts 1; each car's weight slows the
# train by DRAG_PER_CAR of the engine's speed alone, so six full cars still make about half of it.
# That is for an engine built to pull MAX_CARS, like the Planet; one built to pull more is slowed
# less by each car, in proportion.
FULL_LOAD = 1.0


DRAG_PER_CAR = 0.08


# Cars come in sizes up the Fibonacci numbers, each holding about 1.6 times the one before; an
# engine (CAR_LEVELS) or tech (railroads.sim.tech.Effect.CAR_SIZE) moves a car that many sizes up.
CAR_SIZES = (5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0, 144.0, 233.0, 377.0, 610.0)
SMALL_CAR = 5.0
MEDIUM_CAR = 13.0
CAR_CAPACITY = 34.0
# Passengers ride a size up from other goods.
PASSENGER_CAR = 55.0
# Bulk goods, dug or grown in great amounts, go BULK_CAR to a car, and a full bulk car weighs as
# much as BULK_CAR / CAR_CAPACITY full cars of anything else, so only heavy engines pull a long
# train of them at a fair speed.
BULK_CAR = 89.0
BULK_CAR_GOODS = frozenset(
    {
        Good.WHEAT,
        Good.BREAD,
        Good.POTATOES,
        Good.APPLES,
        Good.ORANGES,
        Good.BANANAS,
        Good.DATES,
        Good.SUGAR,
        Good.GRAPES,
        Good.COAL,
        Good.IRON,
        Good.SAND,
        Good.CLAY,
        Good.LIMESTONE,
        Good.SALT,
        Good.LUMBER,
    }
)
# Goods carried in small or medium cars; everything else but passengers goes CAR_CAPACITY to a car.
SMALL_CAR_GOODS = frozenset(
    {
        Good.GOLD,
        Good.DIAMONDS,
        Good.LUXURIES,
        Good.INCENSE,
        Good.ELECTRONICS,
        Good.CARS,
        Good.TANKS,
        Good.PLANES,
    }
)
MEDIUM_CAR_GOODS = frozenset(
    {
        Good.STEEL,
        Good.COPPER,
        Good.COKE,
        Good.CASTINGS,
        Good.MACHINERY,
        Good.TOOLS,
        Good.GUNS,
        Good.APPLIANCES,
        Good.BRICKS,
        Good.CEMENT,
        Good.GLASS,
        Good.PORCELAIN,
        Good.STOVE_TILES,
        Good.POTS,
        Good.FURNITURE,
        Good.BARRELS,
        Good.CATTLE,
        Good.PIGS,
        Good.SHEEP,
        Good.CHICKENS,
    }
)


# Goods whose cars are this many sizes up CAR_SIZES from the rest of their kind: cattle ride in
# roomier cattle cars than the other animals.
SIZES_UP: Mapping[Good, int] = MappingProxyType({Good.CATTLE: 1})

BULK_HEFT = BULK_CAR / CAR_CAPACITY


def heft(good: Good) -> float:
    """How many times as heavy as other full cars a full car of a good is."""
    return BULK_HEFT if good in BULK_CAR_GOODS else 1.0


def car_size(good: Good) -> float:
    """How much of a good one car carries, before what tech adds to coaches."""
    if good == Good.PASSENGERS:
        return PASSENGER_CAR
    if good in BULK_CAR_GOODS:
        return BULK_CAR
    if good in SMALL_CAR_GOODS:
        return SMALL_CAR
    if good in MEDIUM_CAR_GOODS:
        return bigger(MEDIUM_CAR, SIZES_UP.get(good, 0))
    return CAR_CAPACITY


# The most a general car of a kind carries, whatever it holds: processed food rides in medium cars.
GENERAL_CAR_MOST: Mapping[General, float] = MappingProxyType({General.PROCESSED_FOOD: MEDIUM_CAR})


def general_car_size(general: General, good: Good) -> float:
    """How much of a good one general car of a kind carries, before what engines add."""
    return min(car_size(good), GENERAL_CAR_MOST.get(general, car_size(good)))


def bigger(size: float, levels: int) -> float:
    """A car size `levels` sizes up CAR_SIZES."""
    return CAR_SIZES[min(CAR_SIZES.index(size) + levels, len(CAR_SIZES) - 1)]


# About how long an engine with its tender, and each car, are on the map, in cells.
ENGINE_LENGTH = 0.8
CAR_LENGTH = 0.42


class Kind(Enum):
    """What an engine is for. Pioneers come before the Planet. Light engines are cheap to buy and
    run and are up to speed at once, but pull three or four cars, slowly, and a heavy load slows
    them most: for short lines with many stops. General ones are the Planet and its heirs. Heavy
    freight engines pull twice as many cars or more and are hardly slowed by a full load, but run
    slowly, take long to get going and cost the most to run after express engines. Express engines
    carry only passengers, much faster, with a dining car, but cost the most and take longest to
    get up to speed, so they pay on long lines."""

    PIONEER = "pioneer"
    LIGHT = "light"
    GENERAL = "general"
    HEAVY = "heavy freight"
    EXPRESS = "express"


class Power(Enum):
    STEAM = "steam"
    DIESEL = "diesel"
    ELECTRIC = "electric"
    MAGLEV = "maglev"


# Speed 1 stands for this many km/h.
KMH_PER_SPEED = 60.0
# How much a load slows an engine of a kind, against the Planet: light engines the most, heavy
# freight engines half as much.
LOAD_DRAG = {Kind.LIGHT: 1.5, Kind.HEAVY: 0.5}
# Passengers on an express pay this much more, and on a maglev MAGLEV_FARE times as much.
EXPRESS_FARE = 1.2
MAGLEV_FARE = 11.0
# How much fuel (railroads.sim.fuel_market) an engine burns a cell for each car it is built to
# pull, by what drives it: coal, diesel fuel or electricity. Light engines burn LIGHT_BURN of that.
BURN_PER_CAR = {
    Power.STEAM: 0.25,
    Power.DIESEL: 0.1,
    Power.ELECTRIC: 0.15,
    Power.MAGLEV: 0.4,
}
LIGHT_BURN = 0.5
# Diesel, electric and maglev engines pull heavier cars, this many sizes up CAR_SIZES.
CAR_LEVELS = {Power.DIESEL: 1, Power.ELECTRIC: 1, Power.MAGLEV: 1}


@dataclass(frozen=True, slots=True)
class Engine:
    """An engine model: from what year it can be bought, how fast it runs against the Planet,
    how many cars it pulls (an express's coaches, without its dining car), what it costs, what it
    is for and what drives it, and the tech nodes (railroads.sim.tech) it needs to open. Electric
    engines run only on electrified track. Once a newer engine of its line can be bought, it is
    no longer sold (railroads.sim.railway.Railway.engines)."""

    title: str
    year: int
    speed: float
    cars: int
    price: float
    kind: Kind = Kind.GENERAL
    power: Power = Power.STEAM
    # How many seconds of play at normal speed it takes to get from a stand to full speed.
    pickup: float = 10.0
    needs: tuple[str, ...] = ()
    # How many cells it can run on the coal or diesel fuel it takes on setting off, so it can
    # only call at stops that near each other; 0 for electric and maglev engines, which run as
    # far as the wires or guideway go.
    tank: float = 0.0
    # Whether it burns wood (lumber) instead of coal, as American engines did while forests
    # were near and coal far.
    wood: bool = False

    @property
    def electric(self) -> bool:
        return self.power == Power.ELECTRIC

    @property
    def maglev(self) -> bool:
        return self.power == Power.MAGLEV

    @property
    def express(self) -> bool:
        return self.kind == Kind.EXPRESS

    @property
    def burn(self) -> float:
        """How much fuel it burns a cell."""
        light = LIGHT_BURN if self.kind == Kind.LIGHT else 1.0
        return BURN_PER_CAR[self.power] * self.hauls * light

    @property
    def car_levels(self) -> int:
        """How many sizes up CAR_SIZES its cars are."""
        return CAR_LEVELS.get(self.power, 0)

    @property
    def kmh(self) -> float:
        return self.speed * KMH_PER_SPEED

    @property
    def line(self) -> Kind:
        """The line of engines it belongs to: pioneers lead up to the Planet's."""
        return Kind.GENERAL if self.kind == Kind.PIONEER else self.kind

    def replaces(self, older: Engine) -> bool:
        """Whether it takes an older engine's place on sale: a newer one of the same line that
        runs on the same track, so an electric never ends the sale of a steam or diesel engine,
        nor a maglev that of anything on rails."""
        return (
            self.line == older.line
            and self.electric == older.electric
            and self.maglev == older.maglev
            and self.year > older.year
        )

    @property
    def hauls(self) -> int:
        """How many cars it is built to pull, the dining car included: a train of that many
        full cars slows it as much as five full cars slow the Planet."""
        return self.cars + (1 if self.express else 0)


P, L, G, H, X = Kind.PIONEER, Kind.LIGHT, Kind.GENERAL, Kind.HEAVY, Kind.EXPRESS
DIESEL, ELECTRIC, MAGLEV = Power.DIESEL, Power.ELECTRIC, Power.MAGLEV
BOILERS, SUPERHEATED = ("boilers",), ("superheaters",)
EXPRESS_TECH = ("express_trains",)
ENGINES: dict[str, Engine] = {
    "penydarren": Engine(
        "Penydarren steam wagon", 1804, 0.25, 1, 1_000.0, P, pickup=14.0, tank=4.0
    ),
    "catch_me": Engine("Catch Me Who Can", 1808, 0.2, 1, 1_500.0, P, pickup=13.0, tank=5.0),
    "salamanca": Engine("Salamanca", 1812, 0.2, 1, 2_000.0, P, pickup=13.0, tank=6.0),
    "puffing_billy": Engine("Puffing Billy", 1815, 0.25, 1, 2_500.0, P, pickup=13.0, tank=7.0),
    "locomotion": Engine("Locomotion", 1820, 0.3, 2, 4_000.0, P, pickup=13.0, tank=8.0),
    "rocket": Engine("Rocket", 1826, 0.4, 2, 6_000.0, P, pickup=13.0, tank=10.0),
    "lion": Engine("Lion", 1834, 0.45, 2, 8_000.0, P, pickup=13.0, tank=12.0),
    "planet": Engine("Planet steam engine", 1840, 0.5, 3, TRAIN_PRICE, G, pickup=12.0, tank=21.0),
    "tank": Engine("Tank engine", 1840, 0.4, 2, 3_000.0, L, pickup=3.0, tank=13.0),
    "goods": Engine(
        "Goods engine", 1853, 0.35, 4, 15_000.0, H, pickup=30.0, tank=20.0, needs=BOILERS
    ),
    "american": Engine(
        "American 4-4-0", 1856, 0.7, 4, 30_000.0, G, pickup=11.0, tank=25.0, wood=True
    ),
    "single": Engine(
        "Single-driver express",
        1858,
        1.0,
        3,
        40_000.0,
        X,
        pickup=25.0,
        tank=30.0,
        needs=EXPRESS_TECH,
    ),
    "light_tank": Engine("Light tank engine", 1862, 0.55, 2, 9_000.0, L, pickup=3.0, tank=16.0),
    "mogul": Engine(
        "Mogul freight engine", 1865, 0.45, 6, 45_000.0, H, pickup=28.0, tank=30.0, needs=BOILERS
    ),
    "atlantic": Engine(
        "Atlantic express",
        1900,
        1.5,
        4,
        120_000.0,
        X,
        pickup=22.0,
        tank=50.0,
        needs=EXPRESS_TECH + SUPERHEATED,
    ),
    "mallet": Engine(
        "Mallet articulated",
        1905,
        0.6,
        8,
        135_000.0,
        H,
        pickup=25.0,
        tank=50.0,
        needs=BOILERS + SUPERHEATED,
    ),
    "pacific": Engine(
        "Pacific engine", 1910, 1.0, 5, 90_000.0, G, pickup=9.0, tank=50.0, needs=SUPERHEATED
    ),
    "boxcab": Engine(
        "Boxcab electric", 1925, 1.6, 5, 270_000.0, G, ELECTRIC, pickup=6.0, needs=("traction",)
    ),
    "railbus": Engine(
        "Railbus", 1935, 1.0, 3, 27_000.0, L, DIESEL, pickup=2.0, tank=60.0, needs=("diesel",)
    ),
    "streamliner": Engine(
        "Streamliner",
        1935,
        2.3,
        5,
        360_000.0,
        X,
        pickup=20.0,
        tank=60.0,
        needs=EXPRESS_TECH + SUPERHEATED,
    ),
    "big_boy": Engine(
        "Big articulated",
        1940,
        0.7,
        10,
        405_000.0,
        H,
        pickup=22.0,
        tank=60.0,
        needs=BOILERS + SUPERHEATED,
    ),
    "mixed_diesel": Engine(
        "Mixed-traffic diesel",
        1950,
        1.3,
        6,
        800_000.0,
        G,
        DIESEL,
        pickup=7.0,
        tank=100.0,
        needs=("diesel",),
    ),
    "heavy_diesel": Engine(
        "Heavy diesel",
        1960,
        0.9,
        12,
        1_200_000.0,
        H,
        DIESEL,
        pickup=18.0,
        tank=100.0,
        needs=BOILERS + ("diesel",),
    ),
    "bullet": Engine(
        "Bullet train",
        1964,
        3.5,
        6,
        1_100_000.0,
        X,
        ELECTRIC,
        pickup=15.0,
        needs=EXPRESS_TECH + ("thyristors",),
    ),
    "express": Engine(
        "Universal electric",
        1965,
        2.4,
        7,
        2_400_000.0,
        G,
        ELECTRIC,
        pickup=5.0,
        needs=("thyristors",),
    ),
    "sprinter": Engine(
        "Sprinter unit",
        1985,
        1.6,
        3,
        81_000.0,
        L,
        DIESEL,
        pickup=2.0,
        tank=100.0,
        needs=("diesel",),
    ),
    "high_speed": Engine(
        "High-speed electric",
        1990,
        5.0,
        8,
        3_200_000.0,
        X,
        ELECTRIC,
        pickup=14.0,
        needs=EXPRESS_TECH + ("thyristors",),
    ),
    "twin_diesel": Engine(
        "Twin-unit diesel",
        1995,
        1.0,
        16,
        3_600_000.0,
        H,
        DIESEL,
        pickup=18.0,
        tank=150.0,
        needs=BOILERS + ("diesel",),
    ),
    "modern_electric": Engine(
        "Modern electric", 2000, 3.2, 8, 7_200_000.0, G, ELECTRIC, pickup=4.0, needs=("thyristors",)
    ),
    "super_diesel": Engine(
        "Super-heavy diesel",
        2040,
        1.1,
        24,
        10_800_000.0,
        H,
        DIESEL,
        pickup=20.0,
        tank=200.0,
        needs=BOILERS + ("diesel",),
    ),
    "maglev": Engine(
        "Maglev",
        2030,
        500.0 / KMH_PER_SPEED,
        10,
        10_000_000.0,
        X,
        MAGLEV,
        pickup=13.0,
        needs=EXPRESS_TECH + ("maglev",),
    ),
}


FIRST_ENGINE = "planet"


def successor(name: str) -> str | None:
    """The first newer engine that ends an engine's sale (Engine.replaces) once it can be
    bought; None for the last of its line."""
    newer = [other for other in ENGINES if ENGINES[other].replaces(ENGINES[name])]
    return min(newer, key=lambda other: ENGINES[other].year, default=None)


type Point = tuple[float, float]


# How much of some cargo came from each station; None where that is not known.
type Parcels = dict[Cell | None, float]


def take_parcels(parcels: Parcels, amount: float, total: float) -> Parcels:
    """Takes `amount` evenly out of parcels of cargo holding `total` in all."""
    if total <= 0 or amount <= 0:
        return {}
    share = min(1.0, amount / total)
    taken = {origin: held * share for origin, held in parcels.items() if held > 0}
    for origin, part in taken.items():
        parcels[origin] -= part
    unknown = amount - sum(taken.values())
    if unknown > 1e-9:
        taken[None] = taken.get(None, 0.0) + unknown
    return taken


@dataclass
class Car:
    """A car for one good, how much of it it carries, and where that came from. A general car
    takes any good of its kind, one at a time: `good` is the one it carries, or carried last."""

    good: Good
    load: float = 0.0
    origins: Parcels = field(default_factory=dict)
    capacity: float = 0.0
    # An express's dining car, which carries nothing.
    dining: bool = False
    general: General | None = None

    def __post_init__(self) -> None:
        if not self.capacity:
            self.capacity = car_size(self.good)

    @property
    def cargo(self) -> Cargo:
        """What a stop asks for to get this car: its kind of goods if it is general."""
        return self.general or self.good

    def fill(self, parcels: Parcels) -> None:
        for origin, amount in parcels.items():
            self.origins[origin] = self.origins.get(origin, 0.0) + amount
            self.load += amount

    def empty(self, amount: float) -> Parcels:
        taken = take_parcels(self.origins, amount, self.load)
        self.load -= amount
        return taken


@dataclass
class Stop:
    """A station a train calls at, and the goods of the cars it pulls when it leaves for the
    next stop."""

    cell: Cell
    cars: list[Cargo] = field(default_factory=list)
    # Whether the train waits here until its cars are full.
    wait_for_load: bool = False
    # Whether the train only takes on fuel here, for a leg longer than it carries fuel for: it
    # unloads and loads nothing and goes on with the cars it came with to the next stop that
    # isn't one, where its passengers and mail are bound.
    refuel_only: bool = False

    def add_car(self, cargo: Cargo, most: int = MAX_CARS) -> bool:
        if len(self.cars) >= most:
            return False
        self.cars.append(cargo)
        return True


@dataclass
class Train:
    name: str
    stops: list[Stop] = field(default_factory=list)
    # The cars it pulls now.
    cars: list[Car] = field(default_factory=list)
    # Which stop it is heading for, the way there as points on the map, and how far along it is.
    heading: int = 1
    # Whether it has called at the stop it is leaving, taking on that stop's cars, and how many
    # stops it has reached since it last set off from its first.
    called: bool = False
    arrivals: int = 0
    # How much longer it will wait at the stop it is leaving for its cars to fill, in months.
    waiting: float = 0.0
    # The station whose cars it pulls, where they were filled.
    loaded_at: Cell | None = None
    path: list[Point] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    # How far along the path each of its points lies, with the path it was worked out for
    # (TrackNetwork.ends).
    path_ends: tuple[list[Point], list[float]] | None = None
    along: float = 0.0
    status: str = "Waiting for stops"
    engine: str = FIRST_ENGINE
    # How fast it runs as a share of its full speed, which it picks up after setting off.
    pace: float = 0.0
    # The train it has stopped to let by, if any, and how long it has stood letting trains by
    # since it last set off from a stop, in months.
    giving_way: str = ""
    waited: float = 0.0
    # Whether it set off from a station whose engine shed had its coal, and whose sand tower had
    # its sand, this month (railroads.sim.facilities).
    fuelled: bool = False
    sanded: bool = False
    # The platform of its station the train stands at, which it sets off from: the station's own
    # cell, or one of the platforms added to it; None for the station's own cell.
    platform: Cell | None = None
    # Whether it stands at its first stop until the player is done setting it up, so it does not
    # set off empty with half a route.
    held: bool = False

    @property
    def load(self) -> dict[Good, float]:
        """What the train carries, by good."""
        carried: dict[Good, float] = {}
        for car in self.cars:
            if car.load > 0:
                carried[car.good] = carried.get(car.good, 0.0) + car.load
        return carried

    @property
    def weight(self) -> float:
        """How heavy the cars are, in empty cars: an empty car weighs 1, a full one 1 + FULL_LOAD,
        and a full bulk car 1 + FULL_LOAD * BULK_HEFT."""
        return sum(1.0 + FULL_LOAD * heft(car.good) * car.load / car.capacity for car in self.cars)

    @property
    def pull(self) -> float:
        """How much of the engine's top speed the train makes with its cars behind it."""
        engine = ENGINES[self.engine]
        drag = DRAG_PER_CAR * MAX_CARS / engine.hauls * LOAD_DRAG.get(engine.kind, 1.0)
        return 1.0 / (1.0 + drag * self.weight)

    @property
    def full(self) -> bool:
        return all(car.dining or car.load >= car.capacity - 1e-9 for car in self.cars)

    @property
    def length(self) -> float:
        """How long the train is on the map, engine and cars, in cells."""
        return ENGINE_LENGTH + CAR_LENGTH * len(self.cars)

    def speed_up(self, months: float, pickup: float) -> float:
        """Picks up speed for `months`, evenly from a stand to full speed in `pickup` months;
        returns its average pace meanwhile."""
        start = self.pace
        if start >= 1.0 or months <= 0:
            return start
        rise = months / pickup
        if start + rise <= 1.0:
            self.pace = start + rise
            return start + rise / 2
        rising = (1.0 - start) * pickup
        self.pace = 1.0
        return (rising * (start + 1.0) / 2 + months - rising) / months

    def onward(self, index: int) -> int:
        """Where among its stops the train is next bound after the stop at `index`: the next
        one that isn't only for refuelling, or simply the next if every other one is."""
        count = len(self.stops)
        for step in range(1, count + 1):
            following = (index + step) % count
            if not self.stops[following].refuel_only:
                return following
        return (index + 1) % count

    def before(self, index: int) -> int:
        """The stop the cars that reach the stop at `index` were taken on at: the last one
        before it that isn't only for refuelling, or simply the one before if every other one
        is."""
        count = len(self.stops)
        for step in range(1, count + 1):
            earlier = (index - step) % count
            if not self.stops[earlier].refuel_only:
                return earlier
        return (index - 1) % count

    def restart(self) -> None:
        """Sends the train back to its first stop after its stops change."""
        self.platform = None
        self.heading = 1
        self.called = False
        self.arrivals = 0
        self.waiting = 0.0
        self.path = []
        self.pace = 0.0
        self.giving_way = ""
        self.waited = 0.0

    def cars_changed(self, index: int) -> None:
        """Takes in new cars for the stop at `index`. A train standing at that stop calls there
        again to swap its cars; otherwise it keeps going and takes them the next time it calls
        there."""
        if self.path or not self.stops:
            return
        if (self.heading - 1) % len(self.stops) == index % len(self.stops):
            self.called = False
            self.waiting = 0.0
