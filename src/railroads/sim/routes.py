"""Planning track between two cells, the way a player lays it: the cheapest route that turns no
more than MAX_TURN_DEG in any cell, bridging rivers and lakes and tunnelling through mountains,
and keeping to empty land where it can (each building in the way costs BUILDING_COST).

Track leaves a cell through the middle of an edge, into the cell next to it, or through a corner,
along the edge between two cells into the cell beyond; corners let track run straight where edges
would zigzag. Stations at either end are joined without track in their own cell; a station in
between is never crossed.
"""

import heapq
import itertools
import math
from dataclasses import dataclass

from railroads.sim.constructions import MAX_TURN_DEG, Building, Constructions, Station, Track
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth
from railroads.sim.tilemap import Relief, Terrain
from railroads.sim.track_rules import TrackBed, track_rules_for

# How much each building that must move or come down adds to a cell's cost.
BUILDING_COST = 0.4
# How far beyond the straight line between the ends a route may wander, in cells.
DETOUR_CELLS = 12


@dataclass(frozen=True, slots=True)
class Route:
    """Planned track: the pieces to lay, in order, and what it costs."""

    cells: list[Cell]
    pieces: list[Track]
    cost: float


def bed_for(constructions: Constructions, cell: Cell) -> TrackBed | None:
    """What track in this cell would be laid on, or None where none can go."""
    tile = constructions.tile_map.tile(cell)
    if tile.terrain in (Terrain.RIVER, Terrain.LAKE):
        return TrackBed.BRIDGE
    if tile.terrain == Terrain.SEA:
        return None
    return TrackBed.TUNNEL if tile.relief == Relief.MOUNTAINS else TrackBed.GROUND


def plan_route(constructions: Constructions, start: Cell, end: Cell) -> Route | None:
    """The cheapest track from `start` to `end`, or None if there is no way."""
    if start == end:
        return None
    grid = constructions.tile_map.grid
    exits = list(range(len(grid.track_points())))
    steps = grid.track_steps()
    sx, sy = grid.center(*start)
    ex, ey = grid.center(*end)
    reach = math.dist((sx, sy), (ex, ey)) + DETOUR_CELLS

    def near_enough(cell: Cell) -> bool:
        x, y = grid.center(*cell)
        return math.dist((x, y), (sx, sy)) + math.dist((x, y), (ex, ey)) <= 2 * reach

    def step_cost(cell: Cell) -> float | None:
        if cell == end and _is_station(constructions, cell):
            return 0.0
        if _is_station(constructions, cell):
            return None
        bed = bed_for(constructions, cell)
        if bed is None:
            return None
        rules = track_rules_for(constructions.tile_map.tile(cell), bed)
        if not rules.can_build:
            return None
        crowd = max(0, 1 - constructions.space_left(cell))
        return rules.cost_factor + BUILDING_COST * crowd

    first_cost = step_cost(start) if not _is_station(constructions, start) else 0.0
    if first_cost is None:
        return None
    counter = itertools.count()
    # States are a cell and the edge track entered it by (None at the start).
    frontier: list[tuple[float, int, Cell, int | None]] = [(0.0, next(counter), start, None)]
    best: dict[tuple[Cell, int | None], float] = {(start, None): first_cost}
    came: dict[tuple[Cell, int | None], tuple[Cell, int | None, int]] = {}
    while frontier:
        _, _, cell, entry = heapq.heappop(frontier)
        state = (cell, entry)
        if cell == end:
            return _route(constructions, came, state, best[state])
        for exit_point in exits:
            if entry is not None and grid.track_turn(entry, exit_point) > MAX_TURN_DEG + 1e-6:
                continue
            if entry == exit_point:
                continue
            nxt, nxt_entry = grid.track_exit(cell, exit_point)
            if not grid.contains(*nxt) or not near_enough(nxt):
                continue
            cost = step_cost(nxt)
            if cost is None:
                continue
            if exit_point % 2:
                beds = {bed_for(constructions, cell), bed_for(constructions, nxt)}
                flanks = grid.track_flanks(cell, exit_point)
                if any(not _takes(constructions, side, beds) for side in flanks):
                    continue
            total = best[state] + cost * float(math.hypot(*steps[exit_point]))
            key = (nxt, nxt_entry)
            if total < best.get(key, math.inf):
                best[key] = total
                came[key] = (cell, entry, exit_point)
                nx, ny = grid.center(*nxt)
                heapq.heappush(
                    frontier, (total + math.dist((nx, ny), (ex, ey)), next(counter), nxt, nxt_entry)
                )
    return None


def lay_route(growth: Growth, route: Route) -> set[Cell]:
    """Lays the planned track, making room for it as it goes; returns the cells that changed."""
    constructions = growth.constructions
    avoid = frozenset(route.cells)
    changed: set[Cell] = set()
    for piece in route.pieces:
        if piece in constructions.at(piece.cell):
            continue
        changed |= growth.clear_room(piece.cell, piece.space, avoid)
        if constructions.why_not(piece) is None:
            constructions.build(piece)
            changed.add(piece.cell)
    return changed


def build_station(growth: Growth, cell: Cell) -> set[Cell] | None:
    """Builds a station in the cell, moving or pulling down what stands there; None if a station
    cannot go there (water, mountains, or track already through it)."""
    constructions = growth.constructions
    station = Station(cell)
    blocked = any(not isinstance(built, Building) for built in constructions.at(cell))
    tile = constructions.tile_map.tile(cell)
    if blocked or tile.terrain != Terrain.LAND or tile.relief == Relief.MOUNTAINS:
        return None
    changed = growth.clear_room(cell, station.space, frozenset({cell}))
    constructions.build(station)
    return changed | {cell}


def _route(
    constructions: Constructions,
    came: dict[tuple[Cell, int | None], tuple[Cell, int | None, int]],
    state: tuple[Cell, int | None],
    cost: float,
) -> Route:
    steps: list[tuple[Cell, int | None, int | None]] = [(state[0], state[1], None)]
    while state in came:
        cell, entry, exit_point = came[state]
        steps.append((cell, entry, exit_point))
        state = (cell, entry)
    steps.reverse()
    cells = [cell for cell, _, _ in steps]
    pieces = []
    for cell, entry, exit_point in steps:
        if _is_station(constructions, cell):
            continue
        bed = bed_for(constructions, cell)
        assert bed is not None
        joined = entry if entry is not None else exit_point
        assert joined is not None
        if entry is None or exit_point is None:
            loose = _loose_end(constructions, cell, joined, bed)
            if loose is not None:
                pieces.append(loose)
        else:
            pieces.append(Track(cell, (entry, exit_point), bed=bed))
    return Route(cells, pieces, cost)


def _loose_end(
    constructions: Constructions, cell: Cell, joined: int, bed: TrackBed
) -> Track | None:
    """The piece at an open end of a route: from the point it joins by, out the other side as
    straight as the map allows; None where none fits, at the very edge of the map."""
    grid = constructions.tile_map.grid
    points = len(grid.track_points())
    straight = (joined + points // 2) % points
    others = sorted(
        (point for point in range(points) if point != joined),
        key=lambda point: grid.track_turn(joined, point),
    )
    for point in [straight, *others]:
        piece = Track(cell, (joined, point), bed=bed)
        if grid.track_turn(joined, point) <= MAX_TURN_DEG + 1e-6 and constructions.why_not(
            piece
        ) in (
            None,
            f"Not enough room: needs {piece.space}, {constructions.space_left(cell)} left",
        ):
            return piece
    return Track(cell, (joined, straight), bed=bed)


def _takes(constructions: Constructions, cell: Cell, beds: set[TrackBed | None]) -> bool:
    """Whether track on each of these beds could run along the edge of this cell."""
    if not constructions.tile_map.grid.contains(*cell) or None in beds:
        return False
    tile = constructions.tile_map.tile(cell)
    return all(bed is not None and track_rules_for(tile, bed).can_build for bed in beds)


def _is_station(constructions: Constructions, cell: Cell) -> bool:
    return any(isinstance(built, Station) for built in constructions.at(cell))
