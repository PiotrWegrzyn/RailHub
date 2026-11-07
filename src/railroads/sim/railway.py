"""Trains: they run along laid track from stop to stop, pulling up to MAX_CARS cars. Each car
carries one good, up to CAR_CAPACITY of it.

At each stop a train unloads its cars into the station, to be handed out, and fills them from
the station's stockpile. Stations deal with the buildings around them once a month, when goods
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
CELLS_PER_MONTH = 30.0
MAX_CARS = 6
# A full car weighs this much more than an empty one, which counts 1; each car's weight slows the
# train by DRAG_PER_CAR of the engine's speed alone, so six full cars still make about half of it.
FULL_LOAD = 1.0
DRAG_PER_CAR = 0.08
CAR_CAPACITY = 10.0
# Most goods a station's stockpile holds, of all kinds together.
STOCKPILE = 100.0

type Point = tuple[float, float]


@dataclass
class Car:
    """A car for one good, and how much of it it carries."""

    good: Good
    load: float = 0.0


@dataclass
class Train:
    name: str
    stops: list[Cell] = field(default_factory=list)
    cars: list[Car] = field(default_factory=list)
    # Which stop it is heading for, the way there as points on the map, and how far along it is.
    heading: int = 0
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

    def add_car(self, good: Good) -> bool:
        if len(self.cars) >= MAX_CARS:
            return False
        self.cars.append(Car(good))
        return True


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
        speed on each stretch between them; None without track joining them."""
        grid = self.constructions.tile_map.grid
        points = len(grid.track_points())
        steps = grid.track_steps()
        counter = itertools.count()
        frontier: list[tuple[float, int, Cell, int, tuple]] = []
        for point in range(points):
            nxt, entry = grid.track_exit(start, point)
            if grid.contains(*nxt):
                heapq.heappush(frontier, (0.0, next(counter), nxt, entry, ((start, point),)))
        seen: set[tuple[Cell, int]] = set()
        while frontier:
            cost, _, cell, entry, trail = heapq.heappop(frontier)
            if cell == end:
                return self._way(trail, (cell, entry))
            if (cell, entry) in seen:
                continue
            seen.add((cell, entry))
            for piece in self.constructions.at(cell):
                if not isinstance(piece, Track) or entry not in piece.ends:
                    continue
                exit_point = piece.ends[1] if piece.ends[0] == entry else piece.ends[0]
                nxt, nxt_entry = grid.track_exit(cell, exit_point)
                if not grid.contains(*nxt):
                    continue
                continues = any(
                    isinstance(b, Track) and nxt_entry in b.ends for b in self.constructions.at(nxt)
                )
                if nxt != end and not continues:
                    continue
                length = float(math.hypot(*steps[exit_point]))
                heapq.heappush(
                    frontier,
                    (cost + length, next(counter), nxt, nxt_entry, (*trail, (cell, exit_point))),
                )
        return None

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
            here = train.stops[(train.heading - 1) % len(train.stops)]
            there = train.stops[train.heading % len(train.stops)]
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
            train.heading = (train.heading + 1) % len(train.stops)
            train.path = []

    def _arrive(self, train: Train, station: Cell) -> None:
        if not any(isinstance(b, Station) for b in self.constructions.at(station)):
            train.status = "Its stop has no station"
            return
        goods = self.station_goods(station)
        for car in train.cars:
            if car.load > 0:
                goods.brought[car.good] = goods.brought.get(car.good, 0.0) + car.load
                car.load = 0.0
        for car in train.cars:
            take = min(CAR_CAPACITY, goods.stockpile.get(car.good, 0.0))
            if take > 0:
                car.load = take
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
