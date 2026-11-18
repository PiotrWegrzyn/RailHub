"""Land the player can buy next to the map, a lot at a time.

Lots are big hexagons laid over the map of the game's place (railroads.newgame.place_map), point
up like its cells, and numbered by axial coordinates: lot (0, 0) is centred on the middle of the
place, and the six around it are its neighbours. A game starts on that lot and the six around it,
a honeycomb holding START_SHARE of the place's cells, so each lot is a seventh of the land a game
starts on. A cell belongs to the lot its centre lies in. Any lot sharing a side with the player's
land can be bought, the first for the FIRST_FIBONACCI-th Fibonacci number of thousands of dollars
and each after it for the next.

The game's map covers the smallest rectangle of cells around the player's lots, starting on an
even row of the place's map so every hex keeps its neighbours; its cells in lots the player has
not bought are locked (railroads.sim.tilemap.TileMap.locked)."""

import math
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import HEX_RADIUS, HEX_ROW_STEP
from railroads.sim.tilemap import START_SHARE, TileMap

type Lot = tuple[int, int]
type Point = tuple[float, float]
type Window = tuple[int, int, int, int]

# The six lots around a lot, by axial steps.
AROUND = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
START_LOTS = frozenset({(0, 0), *AROUND})
# The first lot costs the FIRST_FIBONACCI-th Fibonacci number of PRICE_UNIT dollars, $377,000, and
# every lot bought after it the next one: $610,000, $987,000 and so on.
FIRST_FIBONACCI = 14
PRICE_UNIT = 1000
# The least share of a lot's cells that must be land for it to be offered.
LEAST_LAND = 0.02
SQRT3 = math.sqrt(3)


def fibonacci(n: int) -> int:
    """The n-th Fibonacci number, counting 1, 1, 2, 3, 5 from n = 1."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def cell_point(row: int, col: int) -> Point:
    """A hex cell's centre on the plane of its map, in cell units (railroads.sim.grids.HexGrid)."""
    return col + 0.5 + 0.5 * (row % 2), HEX_RADIUS + row * HEX_ROW_STEP


@dataclass
class Lots:
    """The lots of one game: how big they are, where lot (0, 0) is centred on the place's map,
    which the player owns, how many were bought and which are on offer."""

    # Each lot's radius, centre to corner, in cell widths.
    radius: float
    # The place's map cell lot (0, 0) is centred on, on an even row.
    middle: tuple[int, int]
    owned: set[Lot] = field(default_factory=lambda: set(START_LOTS))
    bought: int = 0
    # The lots on offer, as offer() last found them.
    offers: list[Lot] = field(default_factory=list)

    @classmethod
    def around_middle(cls, place: TileMap, share: float = START_SHARE) -> Self:
        """The lots of a game starting on the middle of a place's map, the seven it starts on
        holding `share` of its cells."""
        rows, cols = place.grid.shape
        return cls(
            radius=cls.radius_for(rows * cols, share), middle=(rows // 2 // 2 * 2, cols // 2)
        )

    @staticmethod
    def radius_for(cells: int, share: float = START_SHARE) -> float:
        """The radius of lots seven of which hold `share` of a place of so many cells: a lot of
        radius r holds about 3 r² cells."""
        return math.sqrt(cells * share / len(START_LOTS) / 3)

    def centre(self, lot: Lot) -> Point:
        """A lot's centre on the place's map's plane."""
        x, y = cell_point(*self.middle)
        q, r = lot
        return x + self.radius * SQRT3 * (q + r / 2), y + self.radius * 1.5 * r

    def corners(self, lot: Lot) -> list[Point]:
        """A lot's six corners on the place's map's plane, clockwise from its right top one."""
        x, y = self.centre(lot)
        angles = [math.radians(60 * i - 30) for i in range(6)]
        return [(x + self.radius * math.cos(a), y + self.radius * math.sin(a)) for a in angles]

    def lot_at_point(self, x: float, y: float) -> Lot:
        """The lot a point on the place's map's plane lies in."""
        q, r = self._lots_at(np.array([x]), np.array([y]))
        return int(q[0]), int(r[0])

    def lots_of(self, window: Window) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """The lot of every cell of a window of the place's map: two arrays of its shape."""
        top, left, rows, cols = window
        row, col = np.indices((rows, cols))
        row, col = row + top, col + left
        x = col + 0.5 + 0.5 * (row % 2)
        y = HEX_RADIUS + row * HEX_ROW_STEP
        return self._lots_at(x, y)

    def _lots_at(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        mx, my = cell_point(*self.middle)
        dx, dy = (x - mx) / self.radius, (y - my) / self.radius
        q, r = SQRT3 / 3 * dx - dy / 3, 2 / 3 * dy
        s = -q - r
        rq, rr, rs = np.rint(q), np.rint(r), np.rint(s)
        eq, er, es = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
        fix_q = (eq > er) & (eq > es)
        fix_r = ~fix_q & (er > es)
        rq = np.where(fix_q, -rr - rs, rq)
        rr = np.where(fix_r, -rq - rs, rr)
        return rq.astype(np.int64), rr.astype(np.int64)

    def window_of(self, lots: set[Lot] | frozenset[Lot]) -> Window:
        """The smallest window of the place's map holding every cell of these lots, from an even
        row: its top row, left column, rows and columns."""
        points = [corner for lot in lots for corner in self.corners(lot)]
        xs, ys = [x for x, _ in points], [y for _, y in points]
        left = math.floor(min(xs) - 1)
        right = math.ceil(max(xs))
        top = math.floor((min(ys) - HEX_RADIUS) / HEX_ROW_STEP) - 1
        top -= top % 2
        bottom = math.ceil((max(ys) - HEX_RADIUS) / HEX_ROW_STEP) + 1
        return top, left, bottom - top + 1, right - left + 1

    def window(self) -> Window:
        """Where the game's map lies on its place's map."""
        return self.window_of(self.owned)

    def in_lots(self, window: Window, lots: set[Lot] | frozenset[Lot]) -> npt.NDArray[np.bool_]:
        """Each cell of a window of the place's map that lies in one of these lots."""
        q, r = self.lots_of(window)
        held = np.zeros(q.shape, dtype=bool)
        for lot_q, lot_r in lots:
            held |= (q == lot_q) & (r == lot_r)
        return held

    def locked(self) -> npt.NDArray[np.bool_] | None:
        """Each cell of the game's map in a lot not owned, or None where there is none."""
        locked = ~self.in_lots(self.window(), self.owned)
        return locked if locked.any() else None

    def locked_lots(self) -> list[Lot]:
        """The lots not owned with cells on the game's map, in order."""
        q, r = self.lots_of(self.window())
        found = set(zip(q.ravel().tolist(), r.ravel().tolist(), strict=True))
        return sorted(found - self.owned)

    def map_point(self, point: Point) -> Point:
        """A point on the place's map's plane, on the game's map's plane."""
        top, left, _, _ = self.window()
        return point[0] - left, point[1] - top * HEX_ROW_STEP

    def place_point(self, point: Point) -> Point:
        """A point on the game's map's plane, on the place's map's plane."""
        top, left, _, _ = self.window()
        return point[0] + left, point[1] + top * HEX_ROW_STEP

    def lot_at(self, row: int, col: int) -> Lot:
        """The lot a cell of the game's map lies in; past the map's edges too."""
        return self.lot_at_point(*self.place_point(cell_point(row, col)))

    def middle_cell(self, lot: Lot) -> tuple[int, int]:
        """The cell of the game's map nearest a lot's centre."""
        x, y = self.centre(lot)
        top, left, _, _ = self.window()
        row = round((y - HEX_RADIUS) / HEX_ROW_STEP)
        col = round(x - 0.5 - 0.5 * (row % 2))
        return row - top, col - left

    def next_to(self) -> list[Lot]:
        """Every lot not owned that shares a side with one owned, in order."""
        around = {(q + dq, r + dr) for q, r in self.owned for dq, dr in AROUND}
        return sorted(around - self.owned)

    def offered(self, land: npt.NDArray[np.bool_]) -> list[Lot]:
        """The lots next to the player's land with at least LEAST_LAND of land, as `land` tells
        it: True for each land cell of the place's map."""
        return [lot for lot in self.next_to() if self.land_share(lot, land) >= LEAST_LAND]

    def offer(self, land: npt.NDArray[np.bool_]) -> None:
        """Finds the lots on offer (offered) and keeps them, so they are known without the
        place's map."""
        self.offers = self.offered(land)

    def land_share(self, lot: Lot, land: npt.NDArray[np.bool_]) -> float:
        """The share of a lot's cells that are land on the place's map, none past its edges."""
        window = self.window_of({lot})
        top, left, rows, cols = window
        inside = self.in_lots(window, {lot})
        row, col = np.indices((rows, cols))
        row, col = row + top, col + left
        on_map = (row >= 0) & (row < land.shape[0]) & (col >= 0) & (col < land.shape[1])
        here = np.zeros((rows, cols), dtype=bool)
        here[on_map] = land[row[on_map], col[on_map]]
        return float((here & inside).sum()) / max(int(inside.sum()), 1)

    def price(self) -> int:
        """What the next lot costs, in dollars."""
        return fibonacci(FIRST_FIBONACCI + self.bought) * PRICE_UNIT

    def buy(self, lot: Lot) -> None:
        if lot not in self.next_to():
            raise ValueError(f"Lot {lot} doesn't share a side with the player's land")
        self.owned.add(lot)
        self.bought += 1
