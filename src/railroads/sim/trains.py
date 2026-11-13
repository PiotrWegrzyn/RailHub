"""The trains on the railway (railroads.sim.railway runs them): the engines that can be bought, the
cars, each carrying one good up to its car_size, the stops a train calls at with the cars it pulls
from each, and the train itself, slowed by the weight of what it carries against the cars its
engine is built to pull. A train sets off from a stand and picks up speed evenly until it runs at
full speed after its engine's pickup time, the Planet's ten seconds at normal speed, better
engines' less and earlier ones' more. An express always leads its coaches with a dining car,
which carries nothing but weighs and costs like any other car.

Freight remembers where it came from as parcels: how much from each station."""

from dataclasses import dataclass, field
from enum import Enum

from railroads.sim.finance import TRAIN_PRICE
from railroads.sim.goods import Good
from railroads.sim.grids import Cell

MAX_CARS = 5


# A full car weighs this much more than an empty one, which counts 1; each car's weight slows the
# train by DRAG_PER_CAR of the engine's speed alone, so six full cars still make about half of it.
# That is for an engine built to pull MAX_CARS, like the Planet; one built to pull more is slowed
# less by each car, in proportion.
FULL_LOAD = 1.0


DRAG_PER_CAR = 0.08


CAR_CAPACITY = 50.0
MEDIUM_CAR = 25.0
SMALL_CAR = 10.0
# Goods carried in small or medium cars; everything else goes CAR_CAPACITY to a car.
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


def car_size(good: Good) -> float:
    """How much of a good one car carries, before what tech adds to coaches."""
    if good in SMALL_CAR_GOODS:
        return SMALL_CAR
    if good in MEDIUM_CAR_GOODS:
        return MEDIUM_CAR
    return CAR_CAPACITY


# About how long an engine with its tender, and each car, are on the map, in cells.
ENGINE_LENGTH = 0.8
CAR_LENGTH = 0.42


class Kind(Enum):
    """What an engine is for. Pioneers come before the Planet; light engines are cheap and pull
    few cars, general ones are the Planet and its heirs, heavy freight engines pull more cars more
    slowly, and express engines carry only passengers, faster and dearer, with a dining car."""

    PIONEER = "pioneer"
    LIGHT = "light"
    GENERAL = "general"
    HEAVY = "heavy freight"
    EXPRESS = "express"


class Power(Enum):
    STEAM = "steam"
    DIESEL = "diesel"
    ELECTRIC = "electric"


# Speed 1, the Planet's, stands for this many km/h.
KMH_PER_SPEED = 60.0
# Passengers on an express pay this much more.
EXPRESS_FARE = 1.2


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

    @property
    def electric(self) -> bool:
        return self.power == Power.ELECTRIC

    @property
    def express(self) -> bool:
        return self.kind == Kind.EXPRESS

    @property
    def kmh(self) -> float:
        return self.speed * KMH_PER_SPEED

    @property
    def line(self) -> Kind:
        """The line of engines it belongs to: pioneers lead up to the Planet's."""
        return Kind.GENERAL if self.kind == Kind.PIONEER else self.kind

    def replaces(self, older: Engine) -> bool:
        """Whether it takes an older engine's place on sale: a newer one of the same line that
        runs on the same track, so an electric never ends the sale of a steam or diesel engine."""
        return (
            self.line == older.line and self.electric == older.electric and self.year > older.year
        )

    @property
    def hauls(self) -> int:
        """How many cars it is built to pull, the dining car included: a train of that many
        full cars slows it as much as five full cars slow the Planet."""
        return self.cars + (1 if self.express else 0)


P, L, G, H, X = Kind.PIONEER, Kind.LIGHT, Kind.GENERAL, Kind.HEAVY, Kind.EXPRESS
DIESEL, ELECTRIC = Power.DIESEL, Power.ELECTRIC
ENGINES: dict[str, Engine] = {
    "penydarren": Engine("Penydarren steam wagon", 1804, 0.3, 1, 500.0, P, pickup=14.0),
    "catch_me": Engine("Catch Me Who Can", 1808, 0.45, 1, 800.0, P, pickup=13.0),
    "salamanca": Engine("Salamanca", 1812, 0.4, 2, 1_200.0, P, pickup=13.0),
    "puffing_billy": Engine("Puffing Billy", 1813, 0.5, 2, 1_500.0, P, pickup=12.0),
    "locomotion": Engine("Locomotion", 1825, 0.6, 3, 2_500.0, P, pickup=12.0),
    "rocket": Engine("Rocket", 1829, 0.8, 3, 3_500.0, P, pickup=11.0),
    "lion": Engine("Lion", 1838, 0.85, 4, 4_000.0, P, pickup=11.0),
    "planet": Engine("Planet steam engine", 1850, 1.0, MAX_CARS, TRAIN_PRICE, G, pickup=10.0),
    "tank": Engine("Tank engine", 1850, 0.7, 3, 2_000.0, L, pickup=9.0),
    "goods": Engine("Goods engine", 1865, 0.8, 7, 7_000.0, H, pickup=10.0, needs=("boilers",)),
    "single": Engine(
        "Single-driver express", 1865, 1.5, 4, 12_000.0, X, pickup=8.0, needs=("express_trains",)
    ),
    "american": Engine("American 4-4-0", 1870, 1.15, 6, 8_000.0, G, pickup=9.0),
    "mogul": Engine(
        "Mogul freight engine", 1880, 0.9, 8, 11_000.0, H, pickup=8.0, needs=("boilers",)
    ),
    "light_tank": Engine("Light tank engine", 1895, 0.9, 4, 4_000.0, L, pickup=8.0),
    "atlantic": Engine(
        "Atlantic express",
        1900,
        1.8,
        5,
        25_000.0,
        X,
        pickup=6.0,
        needs=("express_trains", "superheaters"),
    ),
    "mallet": Engine(
        "Mallet articulated",
        1905,
        0.95,
        10,
        22_000.0,
        H,
        pickup=8.0,
        needs=("boilers", "superheaters"),
    ),
    "pacific": Engine(
        "Pacific engine", 1910, 1.4, 7, 16_000.0, G, pickup=6.0, needs=("superheaters",)
    ),
    "boxcab": Engine(
        "Boxcab electric", 1925, 1.4, 8, 30_000.0, G, ELECTRIC, pickup=4.0, needs=("traction",)
    ),
    "railbus": Engine("Railbus", 1935, 1.3, 4, 9_000.0, L, DIESEL, pickup=5.0, needs=("diesel",)),
    "streamliner": Engine(
        "Streamliner",
        1935,
        2.3,
        6,
        60_000.0,
        X,
        pickup=5.0,
        needs=("express_trains", "superheaters"),
    ),
    "big_boy": Engine(
        "Big articulated", 1940, 1.1, 12, 45_000.0, H, pickup=7.0, needs=("boilers", "superheaters")
    ),
    "mixed_diesel": Engine(
        "Mixed-traffic diesel", 1950, 1.8, 8, 40_000.0, G, DIESEL, pickup=4.0, needs=("diesel",)
    ),
    "heavy_diesel": Engine(
        "Heavy diesel", 1960, 1.3, 14, 70_000.0, H, DIESEL, pickup=5.0, needs=("boilers", "diesel")
    ),
    "bullet": Engine(
        "Bullet train",
        1964,
        3.5,
        7,
        150_000.0,
        X,
        ELECTRIC,
        pickup=3.0,
        needs=("express_trains", "thyristors"),
    ),
    "express": Engine(
        "Universal electric",
        1965,
        2.2,
        10,
        70_000.0,
        G,
        ELECTRIC,
        pickup=3.0,
        needs=("thyristors",),
    ),
    "sprinter": Engine(
        "Sprinter unit", 1985, 2.0, 5, 18_000.0, L, DIESEL, pickup=3.0, needs=("diesel",)
    ),
    "high_speed": Engine(
        "High-speed electric",
        1990,
        5.0,
        9,
        300_000.0,
        X,
        ELECTRIC,
        pickup=2.5,
        needs=("express_trains", "thyristors"),
    ),
    "twin_diesel": Engine(
        "Twin-unit diesel",
        1995,
        1.4,
        20,
        140_000.0,
        H,
        DIESEL,
        pickup=5.0,
        needs=("boilers", "diesel"),
    ),
    "modern_electric": Engine(
        "Modern electric", 2000, 3.0, 12, 120_000.0, G, ELECTRIC, pickup=2.5, needs=("thyristors",)
    ),
    "super_diesel": Engine(
        "Super-heavy diesel",
        2040,
        1.5,
        40,
        300_000.0,
        H,
        DIESEL,
        pickup=6.0,
        needs=("boilers", "diesel"),
    ),
    "maglev": Engine(
        "500 km/h electric",
        2040,
        500.0 / KMH_PER_SPEED,
        11,
        700_000.0,
        X,
        ELECTRIC,
        pickup=2.0,
        needs=("express_trains", "thyristors"),
    ),
}


FIRST_ENGINE = "planet"


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
    """A car for one good, how much of it it carries, and where that came from."""

    good: Good
    load: float = 0.0
    origins: Parcels = field(default_factory=dict)
    capacity: float = 0.0
    # An express's dining car, which carries nothing.
    dining: bool = False

    def __post_init__(self) -> None:
        if not self.capacity:
            self.capacity = car_size(self.good)

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
    cars: list[Good] = field(default_factory=list)
    # Whether the train waits here, up to WAIT_MONTHS, for its cars to fill.
    wait_for_load: bool = True

    def add_car(self, good: Good, most: int = MAX_CARS) -> bool:
        if len(self.cars) >= most:
            return False
        self.cars.append(good)
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
        """How heavy the cars are, in empty cars: an empty car weighs 1, a full one 1 + FULL_LOAD."""
        return sum(1.0 + FULL_LOAD * car.load / car.capacity for car in self.cars)

    @property
    def pull(self) -> float:
        """How much of the engine's top speed the train makes with its cars behind it."""
        drag = DRAG_PER_CAR * MAX_CARS / ENGINES[self.engine].hauls
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

    def restart(self) -> None:
        """Sends the train back to its first stop after its stops change."""
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
