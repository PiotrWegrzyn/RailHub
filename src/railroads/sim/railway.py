"""Trains: they run along laid track from stop to stop, starting at the first. Each stop says
which cars, up to MAX_CARS, the train pulls on to the next one; each car carries one good, up to
CAR_CAPACITY of it.

At each stop a train unloads its cars into the station, to be handed out, swaps them for that
stop's cars and fills those from the station's stockpile. Stations deal with the buildings around them once a month, when goods
are shared out (railroads.sim.economy): they hand what trains brought to buildings within
DELIVERY_RADIUS cells, and take every spare good made within COLLECTION_RADIUS cells into their
stockpile, up to STOCKPILE goods in all.
"""

import heapq
import itertools
import math
from dataclasses import dataclass, field

from railroads.sim.constructions import Constructions, Station, Track
from railroads.sim.economy import Good
from railroads.sim.grids import Cell
from railroads.sim.track_rules import track_rules_for

TRAIN_COST_NAME = "Steam engine"
# How far a train runs in a month, in cells, on flat ground.
CELLS_PER_MONTH = 60.0
MAX_CARS = 6
# A full car weighs this much more than an empty one, which counts 1; each car's weight slows the
# train by DRAG_PER_CAR of the engine's speed alone, so six full cars still make about half of it.
FULL_LOAD = 1.0
DRAG_PER_CAR = 0.08
CAR_CAPACITY = 10.0
# Longest a train waits at a stop for its cars to fill, in months.
WAIT_MONTHS = 1.0
# Most goods a station's stockpile holds, of all kinds together.
STOCKPILE = 100.0

type Point = tuple[float, float]


@dataclass
class Car:
    """A car for one good, and how much of it it carries."""

    good: Good
    load: float = 0.0


@dataclass
class Stop:
    """A station a train calls at, and the goods of the cars it pulls when it leaves for the
    next stop."""

    cell: Cell
    cars: list[Good] = field(default_factory=list)
    # Whether the train waits here, up to WAIT_MONTHS, for its cars to fill.
    wait_for_load: bool = True

    def add_car(self, good: Good) -> bool:
        if len(self.cars) >= MAX_CARS:
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
        return sum(1.0 + FULL_LOAD * car.load / CAR_CAPACITY for car in self.cars)

    @property
    def pull(self) -> float:
        """How much of the engine's top speed the train makes with its cars behind it."""
        return 1.0 / (1.0 + DRAG_PER_CAR * self.weight)

    @property
    def full(self) -> bool:
        return all(car.load >= CAR_CAPACITY - 1e-9 for car in self.cars)

    def restart(self) -> None:
        """Sends the train back to its first stop after its stops change."""
        self.heading = 1
        self.called = False
        self.arrivals = 0
        self.waiting = 0.0
        self.path = []

    def cars_changed(self) -> None:
        """Takes in new cars at a stop. A train standing at a stop calls there again to swap
        its cars, and one still on its first trip goes back to its first stop for them, so the
        cars set up after the stops still carry goods that trip; later on, changes show at the
        next call."""
        if not self.path:
            self.called = False
            self.waiting = 0.0
        elif self.arrivals == 0:
            self.restart()


@dataclass
class StationGoods:
    """What a station holds: goods trains brought, waiting to be handed out, and its stockpile of
    spare goods from around it, waiting for trains."""

    brought: dict[Good, float] = field(default_factory=dict)
    stockpile: dict[Good, float] = field(default_factory=dict)

    @property
    def stocked(self) -> float:
        return sum(self.stockpile.values())


class Railway:
    """The trains on one map, and the goods held at its stations."""

    def __init__(self, constructions: Constructions) -> None:
        self.constructions = constructions
        self.trains: list[Train] = []
        self.goods: dict[Cell, StationGoods] = {}
        self._names = itertools.count(1)

    def stations(self) -> list[Cell]:
        return sorted(built.cell for built in self.constructions if isinstance(built, Station))

    def buy(self) -> Train:
        train = Train(f"{TRAIN_COST_NAME} {next(self._names)}")
        self.trains.append(train)
        return train

    def sell(self, train: Train) -> None:
        self.trains.remove(train)

    def station_goods(self, station: Cell) -> StationGoods:
        return self.goods.setdefault(station, StationGoods())

    def run(self, months: float) -> None:
        """Moves every train on by this many months."""
        for train in self.trains:
            self._run(train, months)

    def route(self, start: Cell, end: Cell) -> tuple[list[Point], list[float]] | None:
        """The way along laid track from one station to another, as points on the map and the
        speed on each stretch between them; None without track joining them. Trains pass
        through stations on the way, which join track at any of their sides."""
        grid = self.constructions.tile_map.grid
        steps = grid.track_steps()
        counter = itertools.count()
        frontier: list[tuple[float, int, Cell, int | None, tuple]] = [
            (0.0, next(counter), start, None, ())
        ]
        seen: set[tuple[Cell, int | None]] = set()
        while frontier:
            cost, _, cell, entry, trail = heapq.heappop(frontier)
            if cell == end and trail:
                return self._way(trail, (cell, entry or 0))
            if (cell, entry) in seen:
                continue
            seen.add((cell, entry))
            for exit_point in self._exits(cell, entry):
                nxt, nxt_entry = grid.track_exit(cell, exit_point)
                if not grid.contains(*nxt) or not (nxt == end or self._joins(nxt, nxt_entry)):
                    continue
                length = float(math.hypot(*steps[exit_point]))
                heapq.heappush(
                    frontier,
                    (cost + length, next(counter), nxt, nxt_entry, (*trail, (cell, exit_point))),
                )
        return None

    def _exits(self, cell: Cell, entry: int | None) -> list[int]:
        """The points a train that came in by `entry` can leave the cell by: any side of a
        station, else the other end of each piece of track that starts where it came in."""
        built = self.constructions.at(cell)
        if any(isinstance(b, Station) for b in built):
            points = range(len(self.constructions.tile_map.grid.track_points()))
            return [point for point in points if point != entry]
        return [
            piece.ends[1] if piece.ends[0] == entry else piece.ends[0]
            for piece in built
            if isinstance(piece, Track) and entry in piece.ends
        ]

    def _joins(self, cell: Cell, entry: int) -> bool:
        """Whether a train coming in by `entry` can go on from this cell."""
        return any(
            isinstance(b, Station) or (isinstance(b, Track) and entry in b.ends)
            for b in self.constructions.at(cell)
        )

    def _way(self, trail: tuple, arrival: tuple[Cell, int]) -> tuple[list[Point], list[float]]:
        grid = self.constructions.tile_map.grid
        steps = grid.track_steps()
        start_cell = trail[0][0]
        points: list[Point] = [grid.center(*start_cell)]
        speeds: list[float] = []
        for cell, exit_point in trail:
            cx, cy = grid.center(*cell)
            sx, sy = steps[exit_point] / 2
            if cell != start_cell:
                points.append((float(cx), float(cy)))
                speeds.append(self._speed(cell))
            points.append((float(cx + sx), float(cy + sy)))
            speeds.append(self._speed(cell))
        points.append(tuple(float(v) for v in grid.center(*arrival[0])))  # type: ignore[arg-type]
        speeds.append(1.0)
        return points, speeds

    def _speed(self, cell: Cell) -> float:
        tile = self.constructions.tile_map.tile(cell)
        pieces = [b for b in self.constructions.at(cell) if isinstance(b, Track)]
        if not pieces:
            return 1.0
        return max(0.2, track_rules_for(tile, pieces[0].bed).speed_factor)

    def _run(self, train: Train, months: float) -> None:
        if len(train.stops) < 2:
            train.status = "Needs two stops"
            train.path = []
            return
        if not train.path:
            leaving = train.stops[(train.heading - 1) % len(train.stops)]
            if not train.called:
                self._arrive(train, leaving)
            if train.waiting > 0:
                self._load(train, leaving)
                if train.full:
                    train.waiting = 0.0
                else:
                    train.waiting -= months
                    train.status = f"Loading at {leaving.cell[0]}, {leaving.cell[1]}"
                    return
            here, there = leaving.cell, train.stops[train.heading % len(train.stops)].cell
            way = self.route(here, there)
            if way is None:
                train.status = "No track to its next stop"
                return
            train.path, train.speeds = way
            train.along = 0.0
            train.status = "Running"
        train.along += months * CELLS_PER_MONTH * self._speed_at(train) * train.pull
        if train.along >= self._length(train.path):
            self._arrive(train, train.stops[train.heading % len(train.stops)])
            train.arrivals += 1
            train.heading = (train.heading + 1) % len(train.stops)
            train.path = []

    def _arrive(self, train: Train, stop: Stop) -> None:
        """Calls at a stop: unloads every car, takes on the stop's cars and fills them."""
        train.called = True
        if not any(isinstance(b, Station) for b in self.constructions.at(stop.cell)):
            train.status = "Its stop has no station"
            return
        goods = self.station_goods(stop.cell)
        if train.loaded_at == stop.cell:
            train.cars = self._swap_cars(train.cars, stop.cars, goods)
        else:
            for car in train.cars:
                if car.load > 0:
                    goods.brought[car.good] = goods.brought.get(car.good, 0.0) + car.load
            train.cars = [Car(good) for good in stop.cars]
        train.loaded_at = stop.cell
        self._load(train, stop)
        if stop.wait_for_load and train.cars and not train.full:
            train.waiting = WAIT_MONTHS

    @staticmethod
    def _swap_cars(cars: list[Car], wanted: list[Good], goods: StationGoods) -> list[Car]:
        """Swaps a train's cars for the wanted ones at the station that filled them: each wanted
        car keeps the fullest car of its good, and what dropped cars held goes back into the
        stockpile."""
        left = sorted(cars, key=lambda car: -car.load)
        kept = []
        for good in wanted:
            car = next((car for car in left if car.good == good), None)
            if car is None:
                kept.append(Car(good))
            else:
                left.remove(car)
                kept.append(car)
        for car in left:
            if car.load > 0:
                goods.stockpile[car.good] = goods.stockpile.get(car.good, 0.0) + car.load
        return kept

    def _load(self, train: Train, stop: Stop) -> None:
        """Tops each car up from the stop's stockpile."""
        goods = self.station_goods(stop.cell)
        for car in train.cars:
            take = min(CAR_CAPACITY - car.load, goods.stockpile.get(car.good, 0.0))
            if take > 0:
                car.load += take
                goods.stockpile[car.good] -= take

    def position(self, train: Train, back: float = 0.0) -> tuple[Point, Point] | None:
        """Where the train is on the map, or a point `back` cells behind its front, and which way
        it faces there, while it runs; None for a point still inside the station it left."""
        if len(train.path) < 2 or train.along - back < 0:
            return None
        left = train.along - back
        for a, b in itertools.pairwise(train.path):
            length = math.dist(a, b)
            if left <= length or b == train.path[-1]:
                share = min(1.0, left / length) if length else 0.0
                at = (a[0] + (b[0] - a[0]) * share, a[1] + (b[1] - a[1]) * share)
                return at, (b[0] - a[0], b[1] - a[1])
            left -= length
        return None

    def _speed_at(self, train: Train) -> float:
        left = train.along
        for (a, b), speed in zip(itertools.pairwise(train.path), train.speeds, strict=False):
            length = math.dist(a, b)
            if left <= length:
                return speed
            left -= length
        return 1.0

    @staticmethod
    def _length(path: list[Point]) -> float:
        return sum(math.dist(a, b) for a, b in itertools.pairwise(path))
