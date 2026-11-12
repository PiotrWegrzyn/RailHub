"""The trains on the railway (railroads.sim.railway runs them): the engines that can be bought, the
cars, each carrying one good up to CAR_CAPACITY, the stops a train calls at with the cars it pulls
from each, and the train itself, slowed by the weight of what it carries. A train sets off from a
stand and picks up speed evenly until it runs at full speed after its engine's pickup time, the
first engine's ten seconds at normal speed and better engines' less.

Freight remembers where it came from as parcels: how much from each station."""

from dataclasses import dataclass, field

from railroads.sim.finance import TRAIN_PRICE
from railroads.sim.goods import Good
from railroads.sim.grids import Cell

MAX_CARS = 5


# A full car weighs this much more than an empty one, which counts 1; each car's weight slows the
# train by DRAG_PER_CAR of the engine's speed alone, so six full cars still make about half of it.
FULL_LOAD = 1.0


DRAG_PER_CAR = 0.08


CAR_CAPACITY = 50.0
# About how long an engine with its tender, and each car, are on the map, in cells.
ENGINE_LENGTH = 0.8
CAR_LENGTH = 0.42


@dataclass(frozen=True, slots=True)
class Engine:
    """An engine model: from what year it can be bought, how fast it runs against the first,
    how many cars it pulls, what it costs, and whether it runs only on electrified track. All but
    the first also need a tech node (railroads.sim.tech) to open."""

    title: str
    year: int
    speed: float
    cars: int
    price: float
    electric: bool = False
    # How many seconds of play at normal speed it takes to get from a stand to full speed.
    pickup: float = 10.0


ENGINES: dict[str, Engine] = {
    "planet": Engine("Planet steam engine", 1850, 1.0, MAX_CARS, TRAIN_PRICE, pickup=10.0),
    "mogul": Engine("Mogul freight engine", 1880, 1.0, 8, 12_000.0, pickup=8.0),
    "pacific": Engine("Pacific express engine", 1910, 1.5, 8, 25_000.0, pickup=6.0),
    "boxcab": Engine("Boxcab electric", 1925, 1.4, 10, 40_000.0, electric=True, pickup=4.0),
    "express": Engine("Express electric", 1965, 2.2, 12, 90_000.0, electric=True, pickup=3.0),
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
    capacity: float = CAR_CAPACITY

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
        return 1.0 / (1.0 + DRAG_PER_CAR * self.weight)

    @property
    def full(self) -> bool:
        return all(car.load >= car.capacity - 1e-9 for car in self.cars)

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
