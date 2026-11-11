"""The way along laid track (railroads.sim.railway uses it): which stations track joins, the route a
train takes from one station to another with the speed on each stretch, and where a train is on
its way. Trains pass through stations, which join track only by their two ends, and may turn round
in them."""

import heapq
import itertools
import math

from railroads.sim.constructions import Constructions, Station, Track
from railroads.sim.grids import Cell
from railroads.sim.track_rules import track_rules_for
from railroads.sim.trains import Point, Train


class TrackNetwork:
    def __init__(self, constructions: Constructions) -> None:
        self.constructions = constructions
        self._connected: tuple[int, set[Cell]] | None = None

    def connected(self) -> set[Cell]:
        """The stations that track joins to at least one other station, worked out again only
        when track or stations have changed."""
        version = self.constructions.railway_version
        if self._connected is None or self._connected[0] != version:
            self._connected = (version, self._join_stations())
        return set(self._connected[1])

    def _join_stations(self) -> set[Cell]:
        stations = set(self.constructions.stations())
        joined: set[Cell] = set()
        for station in stations:
            if station not in joined and (other := self._next_station(station, stations)):
                joined |= {station, other}
        return joined

    def _next_station(self, start: Cell, stations: set[Cell]) -> Cell | None:
        """Another station track leads to from this one, if there is any."""
        grid = self.constructions.tile_map.grid
        frontier: list[tuple[Cell, int | None]] = [(start, None)]
        seen = set(frontier)
        while frontier:
            cell, entry = frontier.pop()
            for exit_point in self._exits(cell, entry):
                nxt, nxt_entry = grid.track_exit(cell, exit_point)
                if not grid.contains(*nxt) or not self._joins(nxt, nxt_entry):
                    continue
                if nxt in stations and nxt != start:
                    return nxt
                if (nxt, nxt_entry) not in seen:
                    seen.add((nxt, nxt_entry))
                    frontier.append((nxt, nxt_entry))
        return None

    def route(
        self, start: Cell, end: Cell, electric: bool = False
    ) -> tuple[list[Point], list[float]] | None:
        """The way along laid track, only electrified track if `electric`, from one station to
        another, as points on the map and the speed on each stretch between them; None without
        track joining them. Trains pass through stations on the way, which join track only by
        their two ends, and may turn round in them."""
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
            for exit_point in self._exits(cell, entry, electric):
                nxt, nxt_entry = grid.track_exit(cell, exit_point)
                if not grid.contains(*nxt) or not self._joins(nxt, nxt_entry):
                    continue
                length = float(math.hypot(*steps[exit_point]))
                heapq.heappush(
                    frontier,
                    (cost + length, next(counter), nxt, nxt_entry, (*trail, (cell, exit_point))),
                )
        return None

    def _exits(self, cell: Cell, entry: int | None, electric: bool = False) -> list[int]:
        """The points a train that came in by `entry` can leave the cell by: either end of a
        station, the one it came in by too, turning round there; else the other end of each
        piece of track, electrified if `electric`, that starts where it came in."""
        built = self.constructions.at(cell)
        if any(isinstance(b, Station) for b in built):
            return list(self.constructions.station_ends(cell))
        return [
            piece.ends[1] if piece.ends[0] == entry else piece.ends[0]
            for piece in built
            if isinstance(piece, Track) and entry in piece.ends and (not electric or piece.electric)
        ]

    def _joins(self, cell: Cell, entry: int) -> bool:
        """Whether a train coming in by `entry` can go on from this cell: by one of a station's
        ends, or onto track that starts there."""
        if self.constructions.station_at(cell) is not None:
            return entry in self.constructions.station_ends(cell)
        return any(isinstance(b, Track) and entry in b.ends for b in self.constructions.at(cell))

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

    def speed_at(self, train: Train) -> float:
        left = train.along
        for (a, b), speed in zip(itertools.pairwise(train.path), train.speeds, strict=False):
            length = math.dist(a, b)
            if left <= length:
                return speed
            left -= length
        return 1.0

    @staticmethod
    def length(path: list[Point]) -> float:
        return sum(math.dist(a, b) for a, b in itertools.pairwise(path))
