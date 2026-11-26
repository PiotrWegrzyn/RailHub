"""The way along laid track (railroads.sim.railway uses it): which stations track joins, the route a
train takes from one station to another with the speed on each stretch, and where a train is on
its way. Trains pass through stations, which join track only by their two ends, and may turn round
in them; in a station with more platforms they may cross to a platform beside the one they are on,
and they reach a station at whichever of its platforms their way comes to first. An engine shed
joins track by its ends as a station does, but only as the end of a way: trains go into it to
wait and come out of it again, never through it."""

import bisect
import heapq
import itertools
import math
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from railroads.sim.constructions import Constructions, Station, Track, TrackWidth
from railroads.sim.grids import Cell, Grid
from railroads.sim.track_rules import track_rules_for
from railroads.sim.trains import Point, Train

# A route: the points along it and the speed on each stretch between them.
type Way = tuple[list[Point], list[float]]
# The step of a route that crosses from one platform of a station to the one beside it.
CROSSING = -1


# The track point each point on a cell's edge lies at (TrackNetwork.entry_point), by grid, for
# at most ENTRIES_KEPT of them.
ENTRIES_KEPT = 200_000
_ENTRIES: dict[Grid, dict[tuple[Cell, Point], int]] = {}


@dataclass
class _Memo:
    """What finding a way asks of each cell (TrackNetwork._memo), kept until track or stations
    change, as after a change many trains look for their ways again over the same cells: the
    railway version it was found for, the exits, whether a train can go on, the platforms
    beside, whether an engine shed stands there, the searches for ways from each station
    (_Search), and how fast trains run and how much they burn through each cell."""

    version: int
    exits: dict[tuple[Cell, int | None, bool, bool], list[int]] = field(default_factory=dict)
    joins: dict[tuple[Cell, int], bool] = field(default_factory=dict)
    beside: dict[Cell, list[Cell]] = field(default_factory=dict)
    sheds: dict[Cell, bool] = field(default_factory=dict)
    searches: dict[tuple[Cell, int | None, bool, bool], _Search] = field(default_factory=dict)
    speeds: dict[Cell, float] = field(default_factory=dict)
    burns: dict[Cell, float] = field(default_factory=dict)


class _Search:
    """A search for ways from a station, or a cell come into by a point (TrackNetwork._search),
    kept going as far
    as each way asked for from it needs and taken up again for the next: it takes its steps in
    the same order whatever it is looking for, so a way found here is the one a search of its
    own would find, and many ways asked for from one station take one search. Each cell is kept
    with where it was first reached from, and in what order."""

    def __init__(self, start: Cell, entry: int | None = None) -> None:
        self.counter = itertools.count()
        self.taken: list[tuple[int, Cell, int]] = []
        self.frontier: list[tuple[float, int, Cell, int | None, int]] = [
            (0.0, next(self.counter), start, entry, -1)
        ]
        self.seen: set[tuple[Cell, int | None]] = set()
        self.first: dict[Cell, tuple[int, int | None, int]] = {}
        self.order = 0

    def reach(
        self, network: TrackNetwork, arrivals: set[Cell], electric: bool, maglev: bool
    ) -> tuple[Cell, int | None, int] | None:
        """The first of the arrivals the search reaches, with the point it comes in by and its
        last step; None if it reaches none of them."""
        reached = [(self.first[cell], cell) for cell in arrivals if cell in self.first]
        if reached:
            (_, entry, last), cell = min(reached)
            return cell, entry, last
        network._find_exits()
        grid = network.constructions.tile_map.grid
        lengths = [float(math.hypot(*step)) for step in grid.track_steps()]
        frontier, taken, seen = self.frontier, self.taken, self.seen
        while frontier:
            cost, _, cell, entry, last = heapq.heappop(frontier)
            arrived = False
            if last >= 0 and cell not in self.first:
                self.first[cell] = (self.order, entry, last)
                self.order += 1
                arrived = cell in arrivals
            if (cell, entry) not in seen:
                seen.add((cell, entry))
                if not (last >= 0 and network._shed(cell)):
                    for beside in network._platforms_beside(cell):
                        taken.append((last, cell, CROSSING))
                        heapq.heappush(
                            frontier, (cost + 1.0, next(self.counter), beside, None, len(taken) - 1)
                        )
                    for exit_point in network._exits(cell, entry, electric, maglev):
                        nxt, nxt_entry = network._track_exit(cell, exit_point)
                        if not grid.contains(*nxt) or not network._joins(nxt, nxt_entry):
                            continue
                        taken.append((last, cell, exit_point))
                        heapq.heappush(
                            frontier,
                            (
                                cost + lengths[exit_point],
                                next(self.counter),
                                nxt,
                                nxt_entry,
                                len(taken) - 1,
                            ),
                        )
            if arrived:
                return cell, entry, last
        return None


# Each network's _Memo, kept off the network so a saved game does not hold it, and the network
# last asked for its memo with that memo, found again without the weak dictionary.
_MEMOS: weakref.WeakKeyDictionary[TrackNetwork, _Memo] = weakref.WeakKeyDictionary()
_LAST_MEMO: list = [None, None]


class TrackNetwork:
    # The cell under each point on the map asked about, kept as they never change.
    _cells: dict[Point, Cell] | None = None
    # The routes found since track or stations last changed, by their ends and track kind, and
    # where track leaving each cell by each point runs into, as the grid never changes. Kept on
    # the class until first used, so games saved without them load.
    _routes: tuple[int, dict[tuple, Way | None]] = (-1, {})
    _exits_to: dict[tuple[Cell, int], tuple[Cell, int]] | None = None
    # The railway version the exits of every cell with track or a station were last found for.
    _exits_for: int = -1
    # For each point on the map asked about, whether it lies in a station and on double track,
    # with the railway version found for.
    _passing: tuple[int, dict[Point, tuple[bool, bool]]] = (-1, {})
    # How long each way found is, by its ends and track kind as in _routes, with the railway
    # version found for: a train that can't set off asks every frame.
    _lengths: tuple[int, dict[tuple, float]] = (-1, {})
    _fuel_lengths: tuple[int, dict[tuple, float]] = (-1, {})

    def __init__(self, constructions: Constructions) -> None:
        self.constructions = constructions
        self._connected: tuple[int, set[Cell]] | None = None

    def _memo(self) -> _Memo:
        """What finding a way has asked of each cell since track or stations last changed."""
        version = self.constructions.railway_version
        network, memo = _LAST_MEMO
        if network is self and memo is not None and memo.version == version:
            return memo
        memo = _MEMOS.get(self)
        if memo is None or memo.version != version:
            memo = _MEMOS[self] = _Memo(version)
        _LAST_MEMO[:] = [self, memo]
        return memo

    def _shed(self, cell: Cell) -> bool:
        sheds = self._memo().sheds
        found = sheds.get(cell)
        if found is None:
            found = sheds[cell] = self.constructions.shed_at(cell) is not None
        return found

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
        """Another station track leads to from this one, by any of its platforms, if there is
        any."""
        constructions = self.constructions
        grid = constructions.tile_map.grid
        frontier: list[tuple[Cell, int | None]] = [
            (cell, None) for cell in constructions.station_cells(start)
        ]
        seen = set(frontier)
        while frontier:
            cell, entry = frontier.pop()
            for exit_point in self._exits(cell, entry):
                nxt, nxt_entry = self._track_exit(cell, exit_point)
                if not grid.contains(*nxt) or not self._joins(nxt, nxt_entry):
                    continue
                main = constructions.main_station(nxt)
                if main is not None and main in stations and main != start:
                    return main
                if (nxt, nxt_entry) not in seen:
                    seen.add((nxt, nxt_entry))
                    frontier.append((nxt, nxt_entry))
        return None

    def route(
        self, start: Cell, end: Cell, electric: bool = False, maglev: bool = False
    ) -> Way | None:
        """The way along laid track, only electrified track if `electric`, only maglev track if
        `maglev` and never maglev track if not, from one station to another, as points on the
        map and the speed on each stretch between them; None without track joining them. Trains
        pass through stations on the way, which join track only by their two ends, and may turn
        round in them. A way is found once until track or stations change, and is shared by the
        trains that take it, which must not change it."""
        version = self.constructions.railway_version
        if self._routes[0] != version:
            self._routes = (version, {})
        found = self._routes[1]
        key = (start, end, electric, maglev)
        if key not in found:
            found[key] = self._find_way(start, end, electric, maglev)
        return found[key]

    def route_length(
        self, start: Cell, end: Cell, electric: bool = False, maglev: bool = False
    ) -> float | None:
        """How long the way route finds is, in cells; None without one."""
        version = self.constructions.railway_version
        if self._lengths[0] != version:
            self._lengths = (version, {})
        lengths = self._lengths[1]
        key = (start, end, electric, maglev)
        length = lengths.get(key)
        if length is None:
            points = self._route_points(start, end, electric, maglev)
            if points is None:
                return None
            length = lengths[key] = sum(math.dist(a, b) for a, b in itertools.pairwise(points))
        return length

    def _route_points(
        self, start: Cell, end: Cell, electric: bool, maglev: bool
    ) -> list[Point] | None:
        """The points of the way route finds, without the speeds on it when it was not found
        yet: planning refuelling stops asks only how long thousands of ways are."""
        version = self.constructions.railway_version
        if self._routes[0] == version:
            way = self._routes[1].get((start, end, electric, maglev), False)
            if way is not False:
                return None if way is None else way[0]
        found = self._found(start, end, electric, maglev)
        return None if found is None else self._points(*found)

    def fuel_length(
        self, start: Cell, end: Cell, electric: bool = False, maglev: bool = False
    ) -> float | None:
        """How many cells of fuel a train burns on the way route finds: its length, each stretch
        weighed by the burn factor of the cell it runs through; None without one."""
        way = self.route(start, end, electric, maglev)
        if way is None:
            return None
        version = self.constructions.railway_version
        if self._fuel_lengths[0] != version:
            self._fuel_lengths = (version, {})
        lengths = self._fuel_lengths[1]
        key = (start, end, electric, maglev)
        length = lengths.get(key)
        if length is None:
            length = lengths[key] = self.path_fuel_length(way[0])
        return length

    def path_fuel_length(self, path: list[Point]) -> float:
        """How many cells of fuel a train burns along a path: its length, each stretch weighed
        by the burn factor of the cell it runs through."""
        return sum(
            math.dist(a, b) * self._burn(self.cell_at(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)))
            for a, b in itertools.pairwise(path)
        )

    def _find_way(self, start: Cell, end: Cell, electric: bool, maglev: bool) -> Way | None:
        found = self._found(start, end, electric, maglev)
        return None if found is None else self._way(*found)

    def _found(
        self, start: Cell, end: Cell, electric: bool, maglev: bool
    ) -> tuple[list[tuple[Cell, int]], tuple[Cell, int]] | None:
        """The steps of the way from one station to another and where it arrives, if any."""
        constructions = self.constructions
        if constructions.shed_at(end) is not None:
            arrivals = {end}
        else:
            arrivals = set(constructions.station_cells(constructions.main_station(end) or end))
        return self._reach(start, None, arrivals, electric, maglev)

    def _reach(
        self,
        start: Cell,
        start_entry: int | None,
        arrivals: set[Cell],
        electric: bool,
        maglev: bool,
    ) -> tuple[list[tuple[Cell, int]], tuple[Cell, int]] | None:
        """The steps of the way _search finds and where it arrives, if any, taking up the search
        from the same start where an earlier way since track or stations changed left it."""
        searches = self._memo().searches
        key = (start, start_entry, electric, maglev)
        search = searches.get(key)
        if search is None:
            search = searches[key] = _Search(start, start_entry)
        found = search.reach(self, arrivals, electric, maglev)
        if found is None:
            return None
        cell, entry, last = found
        return _trail(search.taken, last), (cell, entry or 0)

    def way_on(
        self, cell: Cell, entry: int, arrival: Cell, electric: bool = False, maglev: bool = False
    ) -> Way | None:
        """The way on for a train coming into a cell of track by this point, to one station cell
        or platform: its points from the middle of the cell on, and the speed on each stretch
        between them; None without track that leads there."""
        found = self._reach(cell, entry, {arrival}, electric, maglev)
        return None if found is None else self._way(*found)

    def ways_avoiding(
        self,
        cell: Cell,
        entry: int,
        arrivals: set[Cell],
        extra: Callable[[Cell, int, int], float],
        electric: bool = False,
        maglev: bool = False,
        most: float = math.inf,
    ) -> dict[Cell, tuple[float, Way]]:
        """The quickest way on to each of the arrivals for a train coming into a cell by this
        point (way_on), each way across a cell outside them, in by one point and out by
        another, counting `extra` cells longer, with what it costs: a way round track other
        trains hold; only those costing no more than `most`. Not kept, as the trains move on."""
        self._find_exits()
        grid = self.constructions.tile_map.grid
        lengths = [float(math.hypot(*step)) for step in grid.track_steps()]
        counter = itertools.count()
        taken: list[tuple[int, Cell, int]] = []
        frontier: list[tuple[float, int, Cell, int | None, int]] = [
            (0.0, next(counter), cell, entry, -1)
        ]
        seen: set[tuple[Cell, int | None]] = set()
        found: dict[Cell, tuple[float, Way]] = {}
        while frontier and len(found) < len(arrivals):
            cost, _, here, into, last = heapq.heappop(frontier)
            if cost > most:
                break
            if here in arrivals and last >= 0 and here not in found:
                found[here] = (cost, self._way(_trail(taken, last), (here, into or 0)))
            if (here, into) in seen:
                continue
            seen.add((here, into))
            if last >= 0 and self._shed(here):
                continue
            for beside in self._platforms_beside(here):
                taken.append((last, here, CROSSING))
                heapq.heappush(frontier, (cost + 1.0, next(counter), beside, None, len(taken) - 1))
            for exit_point in self._exits(here, into, electric, maglev):
                nxt, nxt_entry = self._track_exit(here, exit_point)
                if not grid.contains(*nxt) or not self._joins(nxt, nxt_entry):
                    continue
                step = lengths[exit_point]
                if into is not None and here not in arrivals:
                    step += extra(here, into, exit_point)
                taken.append((last, here, exit_point))
                heapq.heappush(
                    frontier, (cost + step, next(counter), nxt, nxt_entry, len(taken) - 1)
                )
        return found

    def entry_point(self, cell: Cell, point: Point) -> int:
        """The track point of a cell a point on its edge, as on a train's path, lies at; kept,
        as every train's way is checked against the track again whenever it changes."""
        grid = self.constructions.tile_map.grid
        entries = _ENTRIES.get(grid)
        if entries is None or len(entries) >= ENTRIES_KEPT:
            entries = _ENTRIES[grid] = {}
        found = entries.get((cell, point))
        if found is None:
            cx, cy = grid.center(*cell)
            offsets = grid.track_steps() / 2
            gaps = np.hypot(cx + offsets[:, 0] - point[0], cy + offsets[:, 1] - point[1])
            found = entries[cell, point] = int(np.argmin(gaps))
        return found

    def speed_in(self, cell: Cell) -> float:
        """How fast trains run through a cell, as a share of their speed."""
        return self._speed(cell)

    def _search(
        self,
        start: Cell,
        start_entry: int | None,
        arrivals: set[Cell],
        electric: bool,
        maglev: bool,
    ) -> Way | None:
        self._find_exits()
        grid = self.constructions.tile_map.grid
        lengths = [float(math.hypot(*step)) for step in grid.track_steps()]
        counter = itertools.count()
        # Each step taken: the step before it (-1 for none), the cell and the point left by.
        taken: list[tuple[int, Cell, int]] = []
        frontier: list[tuple[float, int, Cell, int | None, int]] = [
            (0.0, next(counter), start, start_entry, -1)
        ]
        seen: set[tuple[Cell, int | None]] = set()
        while frontier:
            cost, _, cell, entry, last = heapq.heappop(frontier)
            if cell in arrivals and last >= 0:
                return self._way(_trail(taken, last), (cell, entry or 0))
            if (cell, entry) in seen:
                continue
            seen.add((cell, entry))
            if last >= 0 and self._shed(cell):
                continue
            for beside in self._platforms_beside(cell):
                taken.append((last, cell, CROSSING))
                heapq.heappush(frontier, (cost + 1.0, next(counter), beside, None, len(taken) - 1))
            for exit_point in self._exits(cell, entry, electric, maglev):
                nxt, nxt_entry = self._track_exit(cell, exit_point)
                if not grid.contains(*nxt) or not self._joins(nxt, nxt_entry):
                    continue
                taken.append((last, cell, exit_point))
                heapq.heappush(
                    frontier,
                    (cost + lengths[exit_point], next(counter), nxt, nxt_entry, len(taken) - 1),
                )
        return None

    def _platforms_beside(self, cell: Cell) -> list[Cell]:
        """The other platforms of the station in a cell that lie beside it."""
        beside = self._memo().beside
        found = beside.get(cell)
        if found is None:
            found = beside[cell] = self._find_platforms_beside(cell)
        return found

    def _find_platforms_beside(self, cell: Cell) -> list[Cell]:
        constructions = self.constructions
        main = constructions.main_station(cell)
        if main is None:
            return []
        cells = constructions.station_cells(main)
        if len(cells) < 2:
            return []
        near = set(constructions.tile_map.grid.within(cell, 1))
        return [other for other in cells if other != cell and other in near]

    def _exits(
        self, cell: Cell, entry: int | None, electric: bool = False, maglev: bool = False
    ) -> list[int]:
        """The points a train that came in by `entry` can leave the cell by: either end of a
        station, the one it came in by too, turning round there; else the other end of each
        piece of track, electrified if `electric` and maglev just when `maglev`, that starts
        where it came in."""
        exits = self._memo().exits
        key = (cell, entry, electric, maglev)
        found = exits.get(key)
        if found is None:
            found = exits[key] = self._find_exits_of(cell, entry, electric, maglev)
        return found

    def _find_exits_of(
        self, cell: Cell, entry: int | None, electric: bool, maglev: bool
    ) -> list[int]:
        built = self.constructions.at(cell)
        if any(isinstance(b, Station) for b in built) or self.constructions.shed_at(cell):
            return list(self.constructions.station_ends(cell))
        return [
            piece.other_end(entry)
            for piece in built
            if isinstance(piece, Track)
            and entry in piece.ends
            and (not electric or piece.electric)
            and piece.maglev == maglev
        ]

    def leads(
        self, cell: Cell, entry: int, exit_point: int, electric: bool = False, maglev: bool = False
    ) -> bool:
        """Whether a train that came into a cell by `entry` can leave it by `exit_point` on what is
        built there now (_exits)."""
        return exit_point in self._exits(cell, entry, electric, maglev)

    def _find_exits(self) -> None:
        """Where track leaving each cell with track or a station by each point runs into, for
        all of them at once (Grid.track_exit): one at a time is slow, and finding a way asks
        for thousands."""
        version = self.constructions.railway_version
        if self._exits_for == version:
            return
        if self._exits_to is None:
            self._exits_to = {}
        self._exits_for = version
        cells = sorted(
            {
                *self.constructions.every_station_cell(),
                *self.constructions.every_shed_cell(),
                *(p.cell for p in self.constructions.track()),
            }
            - {cell for cell, point in self._exits_to if point == 0}
        )
        if not cells:
            return
        grid = self.constructions.tile_map.grid
        steps = grid.track_steps()
        count = len(steps)
        rows, cols = np.array(cells).T
        cx, cy = grid.center(rows, cols)
        to_rows, to_cols = grid.cell_at(
            np.asarray(cx)[:, None] + steps[None, :, 0], np.asarray(cy)[:, None] + steps[None, :, 1]
        )
        for cell, row, col in zip(cells, to_rows.tolist(), to_cols.tolist(), strict=True):
            for point in range(count):
                self._exits_to[(cell, point)] = (
                    (row[point], col[point]),
                    (point + count // 2) % count,
                )

    def _track_exit(self, cell: Cell, point: int) -> tuple[Cell, int]:
        if self._exits_to is None:
            self._exits_to = {}
        key = (cell, point)
        found = self._exits_to.get(key)
        if found is None:
            found = self._exits_to[key] = self.constructions.tile_map.grid.track_exit(cell, point)
        return found

    def _joins(self, cell: Cell, entry: int) -> bool:
        """Whether a train coming in by `entry` can go on from this cell: by one of a station's
        ends, or onto track that starts there."""
        joins = self._memo().joins
        key = (cell, entry)
        found = joins.get(key)
        if found is None:
            found = joins[key] = self._find_joins(cell, entry)
        return found

    def _find_joins(self, cell: Cell, entry: int) -> bool:
        if self.constructions.is_stop(cell):
            return entry in self.constructions.station_ends(cell)
        return any(isinstance(b, Track) and entry in b.ends for b in self.constructions.at(cell))

    def _way(self, trail: list[tuple[Cell, int]], arrival: tuple[Cell, int]) -> Way:
        start_cell = trail[0][0]
        speeds: list[float] = []
        for cell, exit_point in trail:
            if cell != start_cell:
                speeds.append(self._speed(cell))
            if exit_point != CROSSING:
                speeds.append(self._speed(cell))
        speeds.append(1.0)
        return self._points(trail, arrival), speeds

    def _points(self, trail: list[tuple[Cell, int]], arrival: tuple[Cell, int]) -> list[Point]:
        """The points of a way found: the middle of each cell it passes and where it leaves it."""
        grid = self.constructions.tile_map.grid
        steps = grid.track_steps()
        start_cell = trail[0][0]
        points: list[Point] = [grid.center(*start_cell)]
        for cell, exit_point in trail:
            cx, cy = grid.center(*cell)
            if cell != start_cell:
                points.append((float(cx), float(cy)))
            if exit_point == CROSSING:
                continue
            sx, sy = steps[exit_point] / 2
            points.append((float(cx + sx), float(cy + sy)))
        points.append(tuple(float(v) for v in grid.center(*arrival[0])))  # type: ignore[arg-type]
        return points

    def _speed(self, cell: Cell) -> float:
        speeds = self._memo().speeds
        found = speeds.get(cell)
        if found is None:
            found = speeds[cell] = self._find_speed(cell)
        return found

    def _find_speed(self, cell: Cell) -> float:
        tile = self.constructions.tile_map.tile(cell)
        pieces = [b for b in self.constructions.at(cell) if isinstance(b, Track)]
        if not pieces:
            return 1.0
        return max(0.2, track_rules_for(tile, pieces[0].bed).speed_factor)

    def _burn(self, cell: Cell) -> float:
        burns = self._memo().burns
        found = burns.get(cell)
        if found is None:
            found = burns[cell] = self._find_burn(cell)
        return found

    def _find_burn(self, cell: Cell) -> float:
        tile = self.constructions.tile_map.tile(cell)
        pieces = [b for b in self.constructions.at(cell) if isinstance(b, Track)]
        if not pieces:
            return 1.0
        return track_rules_for(tile, pieces[0].bed).burn_factor

    def position(self, train: Train, back: float = 0.0) -> tuple[Point, Point] | None:
        """Where the train is on the map, or a point `back` cells behind its front, and which way
        it faces there, while it runs; None for a point still inside the station it left."""
        if len(train.path) < 2 or train.along - back < 0:
            return None
        return _placed(train.path, self.ends(train), train.along - back)

    def speed_at(self, train: Train) -> float:
        path = train.path
        if len(path) < 2:
            return 1.0
        ends = self.ends(train)
        if train.along > ends[-1]:
            return 1.0
        index = _stretch_at(ends, train.along)
        return train.speeds[index] if index < len(train.speeds) else 1.0

    def passing(self, point: Point) -> tuple[bool, bool]:
        """Whether a point on the map lies in a station, and whether on double track: where
        trains can pass each other. Kept until track or stations change, as trains that meet
        ask every frame."""
        version = self.constructions.railway_version
        if self._passing[0] != version:
            self._passing = (version, {})
        found = self._passing[1]
        kept = found.get(point)
        if kept is None:
            cell = self.cell_at(point)
            built = self.constructions.at(cell)
            kept = found[point] = (
                self.constructions.is_stop(cell),
                any(isinstance(b, Track) and b.width >= TrackWidth.DOUBLE for b in built),
            )
        return kept

    def cell_at(self, point: Point) -> Cell:
        """The cell a point on the map lies in."""
        if self._cells is None:
            self._cells = {}
        cell = self._cells.get(point)
        if cell is None:
            grid = self.constructions.tile_map.grid
            row, col = grid.cell_at(np.array(point[0]), np.array(point[1]))
            cell = self._cells[point] = (int(row), int(col))
        return cell

    def body(self, train: Train, step: float) -> list[tuple[Point, Point, Point]]:
        """Points along the running train from its front back to its tail, `step` cells
        apart, leaving out what is still inside the station it left (TrackNetwork.spot)."""
        front = train.along
        tail = max(0.0, front - train.length)
        if len(train.path) < 2 or front <= 0:
            return []
        count = int((front - tail) / step) + 1
        return [self.spot(train, front - step * k) for k in range(count)]

    def spot(self, train: Train, along: float) -> tuple[Point, Point, Point]:
        """The point this far along the train's path, which way the train faces there, and the
        middle of the stretch of track it is on."""
        path = train.path
        ends = self.ends(train)
        index = _stretch_at(ends, along)
        a, b = path[index], path[index + 1]
        length = ends[index + 1] - ends[index]
        share = min(1.0, (along - ends[index]) / length) if length else 0.0
        at = (a[0] + (b[0] - a[0]) * share, a[1] + (b[1] - a[1]) * share)
        middle = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        return at, (b[0] - a[0], b[1] - a[1]), middle

    def can_pass(self, one: tuple[Point, Point, Point], other: tuple[Point, Point, Point]) -> bool:
        """Whether two trains at these spots (TrackNetwork.spot) can pass each other: where
        either is within a station, or on double track going opposite ways. Either is enough,
        as spots close to where a station or double track ends may lie on each side of it."""
        (_, way, middle), (_, other_way, other_middle) = one, other
        in_station, double = self.passing(middle)
        other_in_station, other_double = self.passing(other_middle)
        if in_station or other_in_station:
            return True
        opposite = way[0] * other_way[0] + way[1] * other_way[1] < 0
        return opposite and (double or other_double)

    def length(self, train: Train) -> float:
        """How long the way the train is on is."""
        return self.ends(train)[-1] if train.path else 0.0

    @staticmethod
    def ends(train: Train) -> list[float]:
        """How far along the train's path each of its points lies, worked out once for each
        path: trains ask every frame."""
        kept = train.path_ends
        if kept is not None and kept[0] is train.path:
            return kept[1]
        path = train.path
        ends = [0.0, *itertools.accumulate(math.dist(a, b) for a, b in itertools.pairwise(path))]
        train.path_ends = (path, ends)
        return ends


def _trail(taken: list[tuple[int, Cell, int]], last: int) -> list[tuple[Cell, int]]:
    """The steps of a way found, from the first to `last`."""
    trail = []
    while last >= 0:
        before, cell, exit_point = taken[last]
        trail.append((cell, exit_point))
        last = before
    trail.reverse()
    return trail


def _stretch_at(ends: list[float], along: float) -> int:
    """The stretch of a path, by the index of the point it starts at, that holds the point
    `along` into it; the last stretch for points past its end."""
    return min(max(bisect.bisect_left(ends, along) - 1, 0), len(ends) - 2)


def _placed(path: list[Point], ends: list[float], along: float) -> tuple[Point, Point]:
    index = _stretch_at(ends, along)
    a, b = path[index], path[index + 1]
    length = ends[index + 1] - ends[index]
    share = min(1.0, (along - ends[index]) / length) if length else 0.0
    return (a[0] + (b[0] - a[0]) * share, a[1] + (b[1] - a[1]) * share), (b[0] - a[0], b[1] - a[1])
