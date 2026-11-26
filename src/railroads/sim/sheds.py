"""Where trains waiting in an engine shed stand (railroads.sim.railway sends them there): the shed
is a roundhouse, its stalls fanned round the back of a turntable and its lead track running out by
the end of its own track that track joins (lead_end). A waiting train stands in a stall facing the
turntable, its front NOSE out of the stall's door, its engine drawn small enough to fit
(railroads.ui.pixel.train_layer), and sets off from there across the turntable and out along the
lead; going in or out it is out of sight inside the shed's cell, coming out of the shed onto its
lead. The roundhouse's drawing
(railroads.ui.pixel.art.facilities) is built from the same measures, so trains stand at the stalls
drawn.

Measures are in the roundhouse's own cells, u along its lead and v across it, ROUNDHOUSE_SIZE
times smaller than on the map."""

import math

from railroads.sim.constructions import Constructions
from railroads.sim.grids import Cell
from railroads.sim.trains import Point

# How much bigger the roundhouse is drawn than built, a cell across between opposite sides, so its
# stalls hold a small engine each and it spills a little past its cell; where its turntable lies
# and how big it is, at which headings its stalls stand
# round it, how far out their doors are and how far their backs.
ROUNDHOUSE_SIZE = 1.8
TURNTABLE = (-0.02, 0.0)
TURNTABLE_RADIUS = 0.09
STALL_HEADINGS = tuple(math.radians(a) for a in range(84, 277, 24))
STALL_FROM = 0.13
STALL_TO = 0.33
# The stalls a roundhouse has at each of its levels, by their place in STALL_HEADINGS, from the
# middle out: the middle three, then three more beside them, then all nine.
BUILT = {1: (3, 4, 5), 2: (2, 3, 4, 5, 6, 7), 3: tuple(range(9))}


def stalls(level: int) -> list[int]:
    """The stalls of a roundhouse of `level`, the middle ones first, as trains take them."""
    middle = len(STALL_HEADINGS) // 2
    return sorted(BUILT[level], key=lambda stall: (abs(stall - middle), stall))


# How far a waiting train's front stands out of its stall's door, in cells.
NOSE = 0.035


def lead_end(constructions: Constructions, cell: Cell, towards: Point | None = None) -> int:
    """The end of the engine shed's track in the cell its lead runs out by: the one track or a
    station joins, else the one nearer `towards`, a point on the map, by default the cell of its
    station nearest it."""
    grid = constructions.tile_map.grid
    ends = constructions.station_ends(cell)
    for end in ends:
        beyond, entry = grid.track_exit(cell, end)
        if not grid.contains(*beyond):
            continue
        if constructions.is_stop(beyond):
            if entry in constructions.station_ends(beyond):
                return end
        elif any(entry in piece.ends for piece in constructions.track_at(beyond)):
            return end
    x, y = grid.center(*cell)
    if towards is None:
        shed = constructions.shed_at(cell)
        cells = [cell] if shed is None else constructions.station_cells(shed.station)
        nearest = min(cells, key=lambda own: math.dist(grid.center(*own), (x, y)))
        towards = grid.center(*nearest)
    points = grid.track_points()
    return min(
        ends,
        key=lambda end: math.dist((x + float(points[end][0]), y + float(points[end][1])), towards),
    )


def _on_map(constructions: Constructions, cell: Cell, u: float, v: float) -> Point:
    """Where a point of the roundhouse in the cell lies on the map."""
    grid = constructions.tile_map.grid
    x, y = grid.center(*cell)
    point = grid.track_points()[lead_end(constructions, cell)]
    angle = math.atan2(float(point[1]), float(point[0]))
    c, s = math.cos(angle), math.sin(angle)
    return (
        float(x) + ROUNDHOUSE_SIZE * (u * c - v * s),
        float(y) + ROUNDHOUSE_SIZE * (u * s + v * c),
    )


def stall_way(constructions: Constructions, cell: Cell, stall: int) -> tuple[Point, Point, Point]:
    """The back of a stall of the engine shed in the cell, by its place in STALL_HEADINGS, its
    door, and the middle of the turntable a train standing in it sets off for."""
    heading = STALL_HEADINGS[stall]
    du, dv = TURNTABLE
    c, s = math.cos(heading), math.sin(heading)
    back = _on_map(constructions, cell, du + STALL_TO * c, dv + STALL_TO * s)
    door = _on_map(constructions, cell, du + STALL_FROM * c, dv + STALL_FROM * s)
    return back, door, _on_map(constructions, cell, du, dv)


def standing(constructions: Constructions, cell: Cell, stall: int) -> list[Point]:
    """The way a train standing in a stall stands on: from the back of the stall out of its door
    and NOSE on towards the turntable, the train's front there (stood_at)."""
    back, door, turntable = stall_way(constructions, cell, stall)
    length = math.dist(door, turntable) or 1.0
    share = NOSE / length
    return [
        back,
        door,
        (door[0] + (turntable[0] - door[0]) * share, door[1] + (turntable[1] - door[1]) * share),
    ]


def stood_at(way: list[Point]) -> float:
    """How far along the way it stands on (standing) a train in a stall has its front."""
    return math.dist(way[0], way[1]) + math.dist(way[1], way[2])
