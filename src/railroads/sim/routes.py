"""Planning track between two cells, the way a player lays it: the cheapest route that turns no
more than MAX_TURN_DEG in any cell, bridging rivers and lakes and tunnelling through mountains,
and keeping to empty land where it can: each building that would have to move out of its way
costs BUILDING_COST and more for the room it takes (SPACE_COST), so track goes round farms and
works rather than through them.

Track leaves a cell through the middle of an edge, into the cell next to it, or through a corner,
along the edge between two cells into the cell beyond; corners let track run straight where edges
would zigzag. Stations at either end are joined without track in their own cell; a station in
between is never crossed.

A route that starts or ends where track is already laid joins it: its first or last piece runs
from a point where that track carries on, turning no more than MAX_TURN_DEG, so trains can run
from one onto the other, and a dead end there is turned to where the route goes instead of being
left as a stub. Following laid track is cheap (REUSE_COST), and the track a route runs along is
widened or electrified to match it. Elsewhere an open end runs straight on across its cell.
"""

import heapq
import itertools
import math
from dataclasses import dataclass, field
from functools import cache

from railroads.sim.constructions import (
    MAX_TURN_DEG,
    Constructions,
    Station,
    Track,
    TrackWidth,
)
from railroads.sim.demolition import clearing, clearing_price, station_site_price
from railroads.sim.finance import track_price
from railroads.sim.grids import Cell, Grid
from railroads.sim.growth import Growth
from railroads.sim.railway import Railway
from railroads.sim.tilemap import Relief, Terrain, Tile
from railroads.sim.track_rules import TrackBed, track_rules_for

# How much each building that must move or come down adds to a cell's cost, against a cell of new
# track on open flat land, and how much more for each unit of room it takes.
BUILDING_COST = 2.5
SPACE_COST = 0.5
# How far beyond the straight line between the ends a route may wander, in cells.
DETOUR_CELLS = 12
# What running along track already laid costs, against new track on open flat land.
REUSE_COST = 0.5
# What each bend costs the planner, against a cell of new track on open flat land, so that
# routes run straight and turn seldom instead of zigzagging.
BEND_COST = 0.6

# The planner numbers cells row by row with this many cells of border round the map, so that a
# step from any cell on it, through a corner too, lands on a numbered cell.
_BORDER = 2
# The search keeps a state as a cell's number times 16 plus the point track came in by, or plus
# this where track may leave by any point: at the start of a route from open land or a station.
_ANYWHERE = 15

type State = tuple[Cell, int | None]
# What a cell offers track: the bed it would be laid on, the beds it could take (a bit for each,
# 1 << bed), and what new track on its bed costs the planner before any buildings, None where
# none can go.
type _Ground = tuple[TrackBed | None, int, float | None]
# What leaving a cell takes, as `_Planner._leave` works it out.
type _Leaving = tuple[float, float, tuple[int, ...], tuple[tuple[int, ...], ...], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class Route:
    """Planned track: the pieces it runs along, in order, as they will be once it is laid, and
    what it costs to plan. `replaces` maps a piece to the track it takes the place of: a dead end
    turned towards where the route goes, or track widened or electrified."""

    cells: list[Cell]
    pieces: list[Track]
    cost: float
    replaces: dict[Track, Track] = field(default_factory=dict)

    def changes(self, constructions: Constructions) -> list[tuple[Track, Track | None]]:
        """The pieces that laying the route builds, each with the track it takes the place of."""
        return [
            (piece, self.replaces.get(piece))
            for piece in self.pieces
            if piece not in constructions.at(piece.cell)
        ]


def bed_for(constructions: Constructions, cell: Cell) -> TrackBed | None:
    """What track in this cell would be laid on, or None where none can go."""
    return _bed_on(constructions.tile_map.tile(cell))


def _bed_on(tile: Tile) -> TrackBed | None:
    if tile.terrain in (Terrain.RIVER, Terrain.LAKE):
        return TrackBed.BRIDGE
    if tile.terrain == Terrain.SEA:
        return None
    return TrackBed.TUNNEL if tile.relief == Relief.MOUNTAINS else TrackBed.GROUND


def plan_route(
    constructions: Constructions,
    start: Cell,
    end: Cell,
    width: TrackWidth = TrackWidth.SINGLE,
    electric: bool = False,
    maglev: bool = False,
) -> Route | None:
    """The cheapest track of this width, electric, maglev or neither, from `start` to `end`, or
    None if there is no way."""
    if start == end:
        return None
    return _Planner(constructions, start, end, width, electric and not maglev, maglev).plan()


def route_price(constructions: Constructions, route: Route) -> float:
    """What laying the route costs: the price of each new piece, less the price of the track it
    takes the place of, and the price of clearing the buildings in its way."""
    tile_map = constructions.tile_map
    total = 0.0
    room: dict[Cell, int] = {}
    for piece, old in route.changes(constructions):
        tile = tile_map.tile(piece.cell)
        total += track_price(piece, tile) - (0.0 if old is None else track_price(old, tile))
        freed = old.space if old is not None and old in constructions.at(piece.cell) else 0
        room[piece.cell] = room.get(piece.cell, 0) + piece.space - freed
    return total + sum(
        clearing_price(constructions, cell, units) or 0.0 for cell, units in room.items()
    )


def lay_route(growth: Growth, route: Route) -> set[Cell]:
    """Lays the planned track, making room for it as it goes; returns the cells that changed."""
    constructions = growth.constructions
    avoid = frozenset(route.cells)
    changed: set[Cell] = set()
    for piece, old in route.changes(constructions):
        replacing = old is not None and old in constructions.at(piece.cell)
        if replacing:
            constructions.remove(old)
        changed |= growth.room.clear(piece.cell, piece.space, avoid)
        if constructions.why_not(piece) is None:
            constructions.build(piece)
            changed.add(piece.cell)
        elif replacing:
            constructions.build(old)
    return changed


def station_way(constructions: Constructions, cell: Cell) -> tuple[int, int]:
    """The way a station built in the cell would take its track through it: along straight
    track through the cell, else towards track that ends pointing into it, else across."""
    grid = constructions.tile_map.grid
    count = len(grid.track_points())
    for piece in constructions.track_at(cell):
        start, end = piece.ends
        if (start + count // 2) % count == end and start in grid.track_ends():
            return start, end
    for point in grid.track_ends():
        beyond, entry = grid.track_exit(cell, point)
        if grid.contains(*beyond) and any(
            entry in piece.ends for piece in constructions.track_at(beyond)
        ):
            return tuple(sorted((point, (point + count // 2) % count)))  # type: ignore[return-value]
    return 0, count // 2


def build_station(
    growth: Growth, cell: Cell, ends: tuple[int, int] | None = None
) -> set[Cell] | None:
    """Builds a station in the cell, its track running between `ends` (by default the way of
    station_way), moving or pulling down the buildings there and taking the place of any track
    through it; None if a station cannot go there (water, mountains, a station already, a way
    that is not straight, or a building that cannot be taken down). Clearing the cell costs
    station_site_price."""
    constructions = growth.constructions
    station = Station(cell, station_way(constructions, cell) if ends is None else ends)
    built = constructions.at(cell)
    tile = constructions.tile_map.tile(cell)
    if any(isinstance(b, Station) for b in built) or tile.terrain != Terrain.LAND:
        return None
    if station_site_price(constructions, cell) is None:
        return None
    if tile.relief == Relief.MOUNTAINS:
        return None
    if ends is not None and constructions._way_problem(ends) is not None:
        return None
    for piece in constructions.track_at(cell):
        constructions.remove(piece)
    changed = growth.room.clear(cell, station.space, frozenset({cell}))
    constructions.build(station)
    return changed | {cell}


def move_station(
    growth: Growth, railway: Railway, old: Cell, new: Cell, ends: tuple[int, int] | None = None
) -> set[Cell] | None:
    """Moves a station to another cell, clearing it as a new station would, its track running
    between `ends` there (by default the way of station_way); its trains, name and goods go with
    it, its level and storage start again. Returns the cells that changed, or None if it cannot
    go there."""
    constructions = growth.constructions
    station = constructions.station_at(old)
    if new == old or station is None:
        return None
    if station_site_price(constructions, new) is None:
        return None
    constructions.remove(station)
    changed = build_station(growth, new, ends)
    if changed is None:
        constructions.build(station)
        return None
    railway.moved(old, new)
    return changed | {old}


@dataclass(frozen=True, slots=True)
class _Geometry:
    """How track runs from cell to cell on one grid, worked out once for every search on it.
    The planner numbers cells row by row, `columns` to a row: the map's and a border on either
    side. `turns` holds how far track bends between two points, as Grid.track_turn, and
    `exits`, for each point track comes in by (and _ANYWHERE), the points it may leave by,
    turning no more than MAX_TURN_DEG, and `bends` what leaving by each costs the planner
    for turning. By the point it leaves by, `lengths` is how far it runs
    to the centre of the next cell and `entries` the point it comes in there by; for even and
    odd rows, `steps` is what to add to a cell's number for that next cell and `flanks` for the
    two cells a corner runs between. `ends` lists the points track may use at all
    (Grid.track_ends), `edges` has a bit for each of them in the middle of an edge, 1 << point,
    and `corners` lists the others."""

    points: int
    columns: int
    ends: tuple[int, ...]
    edges: int
    corners: tuple[int, ...]
    turns: tuple[tuple[float, ...], ...]
    exits: tuple[tuple[int, ...], ...]
    bends: tuple[tuple[float, ...], ...]
    lengths: tuple[float, ...]
    entries: tuple[int, ...]
    steps: tuple[tuple[int, ...], ...]
    flanks: tuple[tuple[tuple[int, ...], ...], ...]


@cache
def _geometry(grid: Grid) -> _Geometry:
    points = len(grid.track_points())
    ends = grid.track_ends()
    columns = grid.cols + 2 * _BORDER
    turns = tuple(
        tuple(grid.track_turn(start, end) for end in range(points)) for start in range(points)
    )
    exits: list[tuple[int, ...]] = [()] * (_ANYWHERE + 1)
    for entry in ends:
        exits[entry] = tuple(
            point for point in ends if point != entry and turns[entry][point] <= MAX_TURN_DEG + 1e-6
        )
    exits[_ANYWHERE] = ends
    bends = [tuple(0.0 if turn < 1e-6 else BEND_COST for turn in row) for row in turns]
    bends += [(0.0,) * points] * (_ANYWHERE + 1 - points)
    origins = [(0, 0), (1, 0)]

    def offset(origin: Cell, cell: Cell) -> int:
        return (cell[0] - origin[0]) * columns + cell[1] - origin[1]

    return _Geometry(
        points=points,
        columns=columns,
        ends=ends,
        edges=sum(1 << point for point in ends if point % 2 == 0),
        corners=tuple(point for point in ends if point % 2 == 1),
        turns=turns,
        exits=tuple(exits),
        bends=tuple(bends),
        lengths=tuple(float(math.hypot(*step)) for step in grid.track_steps()),
        entries=tuple(grid.track_exit(origins[0], point)[1] for point in range(points)),
        steps=tuple(
            tuple(offset(origin, grid.track_exit(origin, point)[0]) for point in range(points))
            for origin in origins
        ),
        flanks=tuple(
            tuple(
                tuple(offset(origin, side) for side in grid.track_flanks(origin, point))
                for point in range(points)
            )
            for origin in origins
        ),
    )


@dataclass
class _Planner:
    """Plans one route. What the search needs to know of a cell is worked out the first time it
    asks and kept by the cell's number (see _Geometry), since it asks again for every point
    track can come into the cell by."""

    constructions: Constructions
    start: Cell
    end: Cell
    width: TrackWidth
    electric: bool
    maglev: bool
    _costs: dict[Cell, float | None] = field(default_factory=dict, init=False)
    _reach: float = field(default=0.0, init=False)
    _centers: tuple[tuple[float, float], tuple[float, float]] = field(init=False)
    _geometry: _Geometry = field(init=False)
    _to_end: list[float | None] = field(init=False)
    _leaving: list[_Leaving | None] = field(init=False)
    _grounds: list[_Ground | None] = field(init=False)
    _exits: dict[int, tuple[tuple[int, ...], ...]] = field(default_factory=dict, init=False)
    # The ends of the stations the route starts and ends at, if it does: it leaves the first by
    # running straight through it and out of one end, and comes into the last by one end.
    _ways_out: tuple[int, ...] = field(default=(), init=False)
    _ways_in: tuple[int, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        grid = self.constructions.tile_map.grid
        start, end = grid.center(*self.start), grid.center(*self.end)
        self._reach = math.dist(start, end) + DETOUR_CELLS  # type: ignore[arg-type]
        self._centers = (start, end)  # type: ignore[assignment]
        self._geometry = _geometry(grid)
        numbered = (grid.rows + 2 * _BORDER) * self._geometry.columns
        self._to_end = [None] * numbered
        self._leaving = [None] * numbered
        self._grounds = [None] * numbered

    def plan(self) -> Route | None:
        constructions = self.constructions
        grid = constructions.tile_map.grid
        if not (grid.contains(*self.start) and grid.contains(*self.end)):
            return None
        if not self._enterable(self.end):
            return None
        start_station = _is_station(constructions, self.start)
        if not start_station and self._cell_cost(self.start) is None:
            return None
        if start_station:
            self._ways_out = constructions.station_ends(self.start)
        if _is_station(constructions, self.end):
            self._ways_in = constructions.station_ends(self.end)
        seeds: list[int | None] = [*self._ways_out] if start_station else [*self._joins(self.start)]
        along = {
            point
            for piece in constructions.track_at(self.start)
            for point in piece.ends
            if point not in seeds
        }
        joins = [] if _is_station(constructions, self.end) else self._joins(self.end)
        route = self._search([*(seeds or [None]), *sorted(along)], joins, along)
        if route is None and (joins or (seeds and not start_station)):
            route = self._search([*self._ways_out] or [None], [], set())
        return route

    def _search(self, seeds: list[int | None], joins: list[int], along: set[int]) -> Route | None:
        """A* over cells and the point track came into them by. `seeds` are the points the
        first piece may start from (None for anywhere); from those in `along` it can only run
        along track already laid. The last piece must lead out by one of `joins` (with none,
        anywhere), unless the route comes in along laid track and runs on along it."""
        lengths, entries = self._geometry.lengths, self._geometry.entries
        bends = self._geometry.bends
        to_end, leaving = self._to_end, self._leaving
        start, end = self._number(self.start), self._number(self.end)
        push, pop, count, inf = heapq.heappush, heapq.heappop, itertools.count().__next__, math.inf
        frontier: list[tuple[float, int, int]] = []
        best: dict[int, float] = {}
        for seed in seeds:
            key = start << 4 | (_ANYWHERE if seed is None else seed)
            push(frontier, (0.0, count(), key))
            best[key] = 0.0
        known = best.get
        came: dict[int, int] = {}
        done = bytearray(len(to_end) << 4)
        while frontier:
            key = pop(frontier)[2]
            if done[key]:
                continue
            done[key] = 1
            index, entry = key >> 4, key & 15
            if index == end:
                assert entry != _ANYWHERE
                if self._ways_in and entry not in self._ways_in:
                    continue
                join = self._join(entry, joins)
                if join is None and self._came_along(came, key):
                    join = self._run_on(entry)
                if not joins or join is not None:
                    return self._route(came, key, best[key], join)
                continue
            cost, along_cost, laid, exits, steps = leaving[index] or self._leave(index)
            runs_on = laid[entry]
            leads = exits[entry]
            if index == start and key not in came and entry in along:
                leads = tuple(point for point in leads if runs_on >> point & 1)
            if index == start and key not in came and self._ways_out:
                leads = tuple(point for point in self._ways_out if point != entry)
            so_far = best[key]
            turning = bends[entry]
            for exit_point in leads:
                nxt = index + steps[exit_point]
                left = to_end[nxt]
                if left is None:
                    left = self._open(nxt)
                if left < 0.0:
                    continue
                piece = along_cost if runs_on >> exit_point & 1 else cost
                total = so_far + piece * lengths[exit_point] + turning[exit_point]
                nxt_key = nxt << 4 | entries[exit_point]
                if total < known(nxt_key, inf):
                    best[nxt_key] = total
                    came[nxt_key] = key
                    push(frontier, (total + left, count(), nxt_key))
        return None

    def _number(self, cell: Cell) -> int:
        return (cell[0] + _BORDER) * self._geometry.columns + cell[1] + _BORDER

    def _cell(self, index: int) -> Cell:
        row, col = divmod(index, self._geometry.columns)
        return row - _BORDER, col - _BORDER

    def _state(self, key: int) -> State:
        entry = key & 15
        return self._cell(key >> 4), None if entry == _ANYWHERE else entry

    def _open(self, index: int) -> float:
        """How far the numbered cell is from the end as the crow flies, which the search takes
        as its guess of what is left to pay from there; -1 where the route may not go into it:
        off the map, too far out of its way, a station other than the end, or where no track
        can go."""
        cell = self._cell(index)
        grid = self.constructions.tile_map.grid
        left = -1.0
        if grid.contains(*cell):
            here = grid.center(*cell)
            start, end = self._centers
            to_end = math.dist(here, end)  # type: ignore[arg-type]
            near = math.dist(here, start) + to_end <= 2 * self._reach  # type: ignore[arg-type]
            if near and self._enterable(cell):
                left = to_end
        self._to_end[index] = left
        return left

    def _leave(self, index: int) -> _Leaving:
        """What track leaving the numbered cell costs a unit of length, new and along track
        laid there; by the point it comes in by, the points laid track runs on to (a bit for
        each, 1 << point) and the points it may leave by, turning no more than MAX_TURN_DEG and
        as far as corners go; and the steps to the cells beyond each point."""
        constructions = self.constructions
        cell = self._cell(index)
        laid = [0] * (_ANYWHERE + 1)
        for piece in constructions.track_at(cell):
            first, second = piece.ends
            laid[first] |= 1 << second
            laid[second] |= 1 << first
        if _is_station(constructions, cell):
            cost, along_cost = 0.0, 0.0
        else:
            new_cost = self._cell_cost(cell)
            assert new_cost is not None
            cost, along_cost = new_cost, REUSE_COST
        fits = self._fits(index)
        if fits not in self._exits:
            self._exits[fits] = tuple(
                tuple(point for point in exits if fits >> point & 1)
                for exits in self._geometry.exits
            )
        steps = self._geometry.steps[cell[0] % 2]
        leaving = (cost, along_cost, tuple(laid), self._exits[fits], steps)
        self._leaving[index] = leaving
        return leaving

    def _fits(self, index: int) -> int:
        """The points track may leave the numbered cell by, as bits, as far as corners go:
        through a corner, the cells on both sides must take track on the beds of the cells it
        runs between."""
        geometry = self._geometry
        parity = self._cell(index)[0] % 2
        steps, flanks = geometry.steps[parity], geometry.flanks[parity]
        ground = self._ground
        bed = ground(index)[0]
        fits = geometry.edges
        if bed is None:
            return fits
        for point in geometry.corners:
            beyond = ground(index + steps[point])[0]
            if beyond is None:
                continue
            beds = 1 << bed | 1 << beyond
            left, right = flanks[point]
            both_take = ground(index + left)[1] & ground(index + right)[1]
            if (both_take & beds) == beds:
                fits |= 1 << point
        return fits

    def _ground(self, index: int) -> _Ground:
        ground = self._grounds[index]
        if ground is None:
            cell = self._cell(index)
            if self.constructions.tile_map.grid.contains(*cell):
                tile = self.constructions.tile_map.tile(cell)
                bed = _bed_on(tile)
                takes = sum(
                    1 << other for other in TrackBed if track_rules_for(tile, other).can_build
                )
                buildable = bed is not None and takes >> bed & 1
                factor = track_rules_for(tile, bed).cost_factor if buildable else None
                ground = (bed, takes, factor)
            else:
                ground = (None, 0, None)
            self._grounds[index] = ground
        return ground

    def _cell_cost(self, cell: Cell) -> float | None:
        """What new track in the cell costs the planner, or None where none can go."""
        if cell not in self._costs:
            factor = self._ground(self._number(cell))[2]
            if factor is None:
                self._costs[cell] = None
            else:
                crowd = max(0, int(self.width) - self.constructions.space_left(cell))
                leaving = clearing(self.constructions, cell, int(self.width)) if crowd else []
                if leaving is None:
                    self._costs[cell] = None
                else:
                    moves = sum(BUILDING_COST + SPACE_COST * b.space for b in leaving)
                    self._costs[cell] = factor + moves
        return self._costs[cell]

    def _enterable(self, cell: Cell) -> bool:
        if _is_station(self.constructions, cell):
            return cell == self.end
        return self._cell_cost(cell) is not None

    def _joins(self, cell: Cell) -> list[int]:
        """The points of the cell where laid track or a station carries on beyond it, which new
        track there can join."""
        constructions = self.constructions
        fits = self._fits(self._number(cell))
        return [
            point
            for point in range(self._geometry.points)
            if _leads_on(constructions, cell, point) and fits >> point & 1
        ]

    def _join(self, entry: int, joins: list[int]) -> int | None:
        """The point the last piece leads out by to join laid track, coming in by `entry`:
        along track already there if it can, else turning as little as it can."""
        turns = self._geometry.turns[entry]
        fits = [point for point in joins if point != entry and turns[point] <= MAX_TURN_DEG + 1e-6]
        return min(
            fits,
            key=lambda point: (
                self.constructions.track_between(self.end, (entry, point)) is None,
                turns[point],
            ),
            default=None,
        )

    def _came_along(self, came: dict[int, int], key: int) -> bool:
        """Whether the route came into this state along laid track, or out of a station."""
        if key not in came:
            return False
        cell, entry = self._state(came[key])
        if _is_station(self.constructions, cell):
            return True
        exit_point = self._geometry.entries[key & 15]
        return (
            entry is not None
            and self.constructions.track_between(cell, (entry, exit_point)) is not None
        )

    def _run_on(self, entry: int) -> int | None:
        """The far end of a piece of laid track in the last cell that starts where the route
        comes in, if there is one."""
        for piece in self.constructions.track_at(self.end):
            if entry in piece.ends:
                return piece.ends[1] if piece.ends[0] == entry else piece.ends[0]
        return None

    def _route(self, came: dict[int, int], key: int, cost: float, join: int | None) -> Route:
        constructions = self.constructions
        entries = self._geometry.entries
        steps: list[tuple[Cell, int | None, int | None]] = [(*self._state(key), None)]
        while key in came:
            previous = came[key]
            steps.append((*self._state(previous), entries[key & 15]))
            key = previous
        steps.reverse()
        pieces: list[Track] = []
        replaces: dict[Track, Track] = {}
        for index, (cell, entry, exit_point) in enumerate(steps):
            if _is_station(constructions, cell):
                continue
            bed = bed_for(constructions, cell)
            assert bed is not None
            joined = entry if index == 0 else join if exit_point is None else None
            if entry is None:
                assert exit_point is not None
                ends = self._loose_end(cell, exit_point, bed)
            elif exit_point is None:
                ends = (entry, join) if join is not None else self._loose_end(cell, entry, bed)
            else:
                ends = (entry, exit_point)
            piece = Track(cell, ends, self.width, bed, self.electric, self.maglev)
            old = constructions.track_between(cell, ends)
            if old is None and joined is not None:
                old = _dead_end(constructions, cell, joined, piece.ends)
            if old is not None:
                piece = _merged(piece, old)
                if piece != old:
                    replaces[piece] = old
            pieces.append(piece)
        return Route([cell for cell, _, _ in steps], pieces, cost, replaces)

    def _loose_end(self, cell: Cell, joined: int, bed: TrackBed) -> tuple[int, int]:
        """The ends of the piece at an open end of a route: from the point it joins by, out the
        other side as straight as the map allows, along track already there if there is some."""
        constructions = self.constructions
        points = self._geometry.points
        turns = self._geometry.turns[joined]
        straight = (joined + points // 2) % points
        others = sorted(
            (point for point in self._geometry.ends if point != joined), key=turns.__getitem__
        )
        crowded = f"Not enough room: needs {int(self.width)}, {constructions.space_left(cell)} left"
        for point in [straight, *others]:
            if turns[point] > MAX_TURN_DEG + 1e-6:
                continue
            piece = Track(cell, (joined, point), self.width, bed, self.electric, self.maglev)
            laid = constructions.track_between(cell, piece.ends) is not None
            if laid or constructions.why_not(piece) in (None, crowded):
                return piece.ends
        return joined, straight


def _leads_on(constructions: Constructions, cell: Cell, point: int) -> bool:
    """Whether a train leaving the cell by this point would run on: into a station by one of
    its ends, or onto track that comes into the cell beyond by that point."""
    grid = constructions.tile_map.grid
    beyond, entry = grid.track_exit(cell, point)
    if not grid.contains(*beyond):
        return False
    if _is_station(constructions, beyond):
        return entry in constructions.station_ends(beyond)
    return any(entry in piece.ends for piece in constructions.track_at(beyond))


def _dead_end(
    constructions: Constructions, cell: Cell, joined: int, ends: tuple[int, int]
) -> Track | None:
    """A piece in the cell from `joined` to a point that leads nowhere and no other piece uses:
    the dead end that new track joining at `joined`, between `ends`, turns into."""
    pieces = constructions.track_at(cell)
    for piece in pieces:
        if joined not in piece.ends or piece.ends == ends:
            continue
        loose = piece.ends[1] if piece.ends[0] == joined else piece.ends[0]
        shared = any(loose in other.ends for other in pieces if other is not piece)
        if not shared and not _leads_on(constructions, cell, loose):
            return piece
    return None


def _merged(piece: Track, old: Track) -> Track:
    """The piece, as wide as the track it replaces, maglev if either is and else electric if
    either is: a guideway laid over rails takes their place, and rails laid along a guideway
    keep it."""
    maglev = piece.maglev or old.maglev
    return Track(
        piece.cell,
        piece.ends,
        max(piece.width, old.width),
        piece.bed,
        (piece.electric or old.electric) and not maglev,
        maglev,
    )


def _is_station(constructions: Constructions, cell: Cell) -> bool:
    return any(isinstance(built, Station) for built in constructions.at(cell))
