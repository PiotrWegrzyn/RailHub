"""The cobbled roads of towns: where they run over the map's plane.

Everything that does not grow from the land (GROWN_FROM_LAND: homes, works, shops, churches, ports)
gets a road along the edges of its cell that the slots it stands on face
(railroads.ui.building_spots), and two such cells side by side share a road along their common
edge. A loose piece of road joins the nearest other piece by the shortest way round the cells, at
most JOIN_STEPS edges, along edges with land on both sides. Corners of hexes are kept as whole numbers (corner_key), so the corner two cells share is the
same corner whichever cell it is worked out from."""

import math
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import CELL_SPACE, Building, Constructions
from railroads.sim.grids import HEX_RADIUS, Cell, HexGrid
from railroads.sim.tilemap import Terrain
from railroads.ui.building_spots import first_edge
from railroads.ui.pixel.ground_palette import GROWN_FROM_LAND

JOIN_STEPS = 4

Corner = tuple[int, int]
Edge = tuple[Corner, Corner]
# The steps from a corner to the three corners next to it, in corner_key units: from the top
# corner of a hex up and to the two below either side, from the bottom corner down and to the
# two above.
_STEPS = (((0, -2), (1, 1), (-1, 1)), ((0, 2), (1, -1), (-1, -1)))


def corner_key(x: float, y: float) -> Corner:
    """A hex corner on the plane as whole numbers: half cells across, half radii down."""
    return round(x * 2), round(y / (HEX_RADIUS / 2))


def corner_point(corner: Corner) -> tuple[float, float]:
    return corner[0] / 2, corner[1] * HEX_RADIUS / 2


@dataclass(frozen=True)
class Roads:
    """Each road as a row of where it starts and ends on the plane (x0, y0, x1, y1)."""

    segments: npt.NDArray[np.float64]

    @classmethod
    def none(cls) -> Roads:
        return cls(np.zeros((0, 4)))


def town_roads(constructions: Constructions) -> Roads:
    """The roads of every town on the map."""
    tile_map = constructions.tile_map
    grid = tile_map.grid
    if not isinstance(grid, HexGrid):
        return Roads.none()
    count = np.zeros(grid.shape, dtype=np.int32)
    for building in _roaded(constructions):
        count[building.cell] += 1
    edges: set[Edge] = set()
    for row, col in np.argwhere(count > 0).tolist():
        cell = (row, col)
        corners = [corner_key(x, y) for x, y in grid.corners(row, col)]
        first = 0
        for built in constructions.at(cell):
            start, first = first, first + built.space
            if not isinstance(built, Building) or built.kind in GROWN_FROM_LAND:
                continue
            for slot in range(start, start + built.space):
                side = (slot + first_edge(row, col)) % CELL_SPACE
                edges.add(_edge(corners[side], corners[(side + 1) % CELL_SPACE]))
        for other in grid.neighbours(row, col):
            if count[other] and cell < other:
                shared = set(corners) & {corner_key(x, y) for x, y in grid.corners(*other)}
                if len(shared) == 2:
                    edges.add(_edge(*shared))
    land = tile_map.terrain == Terrain.LAND
    edges = _joined(edges, grid, land)
    segments = np.array(
        [(*corner_point(a), *corner_point(b)) for a, b in sorted(edges)], dtype=np.float64
    ).reshape(-1, 4)
    return Roads(segments)


def _roaded(constructions: Constructions) -> Iterator[Building]:
    for built in constructions:
        if isinstance(built, Building) and built.kind not in GROWN_FROM_LAND:
            yield built


def _edge(a: Corner, b: Corner) -> Edge:
    return (a, b) if a < b else (b, a)


def _cells_beside(a: Corner, b: Corner, grid: HexGrid) -> tuple[Cell, Cell] | None:
    """The two cells an edge between corners `a` and `b` lies between, or None if it is no
    edge of a hex."""
    (ax, ay), (bx, by) = corner_point(a), corner_point(b)
    mx, my = (ax + bx) / 2, (ay + by) / 2
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    nx, ny = -dy / length * 0.1, dx / length * 0.1
    one, two = _cell_at(mx + nx, my + ny), _cell_at(mx - nx, my - ny)
    if one == two:
        return None
    for cell in (one, two):
        if not grid.contains(*cell):
            return None
        if {a, b} - {corner_key(x, y) for x, y in grid.corners(*cell)}:
            return None
    return one, two


def _cell_at(x: float, y: float) -> Cell:
    """HexGrid.cell_at for one point, without numpy, which is many times slower for one."""
    px, py = x - 0.5, y - HEX_RADIUS
    q, r = px - py / math.sqrt(3), py * 2 / math.sqrt(3)
    s = -q - r
    round_q, round_r, round_s = round(q), round(r), round(s)
    error_q, error_r, error_s = abs(round_q - q), abs(round_r - r), abs(round_s - s)
    if error_q > error_r and error_q > error_s:
        round_q = -round_r - round_s
    elif error_r > error_s:
        round_r = -round_q - round_s
    return round_r, round_q + (round_r - (round_r & 1)) // 2


def _passable(a: Corner, b: Corner, grid: HexGrid, land: npt.NDArray[np.bool_]) -> bool:
    cells = _cells_beside(a, b, grid)
    return cells is not None and all(land[cell] for cell in cells)


def _joined(edges: set[Edge], grid: HexGrid, land: npt.NDArray[np.bool_]) -> set[Edge]:
    """The roads with every loose piece joined to the nearest other piece, smallest first, by
    the shortest way along the edges of hexes with land on both sides, at most JOIN_STEPS
    edges."""
    edges = set(edges)
    parent: dict[Corner, Corner] = {}
    known: dict[Edge, bool] = {}

    def passable(a: Corner, b: Corner) -> bool:
        edge = _edge(a, b)
        if edge not in known:
            known[edge] = _passable(a, b, grid, land)
        return known[edge]

    def find(corner: Corner) -> Corner:
        while parent.setdefault(corner, corner) != corner:
            parent[corner] = parent[parent[corner]]
            corner = parent[corner]
        return corner

    for a, b in edges:
        parent[find(a)] = find(b)
    pieces: dict[Corner, list[Corner]] = {}
    for corner in list(parent):
        pieces.setdefault(find(corner), []).append(corner)
    for piece in sorted(pieces.values(), key=lambda corners: (len(corners), min(corners))):
        owner = find(piece[0])
        came: dict[Corner, Corner | None] = dict.fromkeys(piece)
        queue = deque((corner, 0) for corner in sorted(piece))
        reached = None
        while queue and reached is None:
            corner, steps = queue.popleft()
            if steps >= JOIN_STEPS:
                continue
            for dx, dy in _STEPS[corner[1] % 3]:
                there = (corner[0] + dx, corner[1] + dy)
                if there in came or not passable(corner, there):
                    continue
                came[there] = corner
                if there in parent and find(there) != find(owner):
                    reached = there
                    break
                queue.append((there, steps + 1))
        if reached is None:
            continue
        corner = reached
        while (before := came[corner]) is not None:
            edges.add(_edge(corner, before))
            parent[find(corner)] = find(before)
            corner = before
    return edges
