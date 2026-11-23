"""A made-up place for tuning growth: every land the game knows side by side, each big enough to
count what grows on it.

The game's own square of a new game here, the seven lots in the middle of the place
(railroads.sim.lots), is split into nine lands holding the same share of it: three bands from
north to south, cold, temperate and hot (BANDS), each split from west to east into barren land,
grassland and forest (COVERS). So the north holds ice, tundra and taiga, the middle steppe,
farmland and woods, and the south desert, savanna and jungle; a cell's temperature and
vegetation tell which land it is in. Big rivers run between the lands, and each land holds the
same features in the same places, by its share of the square: a range of hills round a core of
mountains, another patch of hills, a lake and a bay of the sea, plains for the rest, and two minor
rivers, one running down from the mountains and one west from the lake. The lots around the
square go on as the land beside them, plains and the big rivers only.

Only the ragged edges are made with noise, the same every time; a game's seed still decides the
deposits and everything that grows.
"""

import itertools

import numpy as np
import numpy.typing as npt

from railroads.sim.geo import Bounds
from railroads.sim.grids import KM_PER_CELL, Cell, HexGrid
from railroads.sim.lots import Lots
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation

# Where the place is said to lie, for the world map and the names of its stations: a box a tenth
# bigger each way than Poland's, about it.
BOUNDS = Bounds(west=13.55, east=24.65, south=48.65, north=55.15)
BANDS = (Temperature.COLD, Temperature.NORMAL, Temperature.HOT)
COVERS = (Vegetation.BARREN, Vegetation.GRASS, Vegetation.FOREST)
# The share of a land's cells in the square each feature takes, and where its middle lies in the
# land, as shares of the way down and across the box round the land's cells in the square.
MOUNTAINS = (0.07, (0.35, 0.6))
HILLS_ROUND_MOUNTAINS = 0.14
MORE_HILLS = (0.07, (0.7, 0.25))
LAKE = (0.03, (0.25, 0.25))
BAY = (0.03, (0.85, 0.85))
# How far the big rivers sway either side of their line, in cells, and how many cells along they
# take to sway one radian; the minor rivers sway less.
RIVER_SWAY = (2.0, 9.0)
STREAM_SWAY = (1.0, 5.0)
# How far a minor river starts from the middle of its mountains or lake, across its course.
STREAM_ASIDE = 2
NOISE_SEED = 7
NOISE_CELLS = 6.0
# Heights in metres: plains, hills and mountains above sea level, and the water.
PLAINS_M = (100, 60)
HILLS_M = (350, 300)
MOUNTAINS_M = (1300, 1400)
RIVER_M, LAKE_M = 80, 90


def place_grid() -> HexGrid:
    """The place's grid, covering BOUNDS in cells KM_PER_CELL across."""
    return HexGrid.covering(BOUNDS.width_km, BOUNDS.height_km, KM_PER_CELL)


def square(grid: HexGrid) -> npt.NDArray[np.bool_]:
    """The cells of the place a new game starts on: the seven lots in its middle."""
    lots = Lots.in_middle_of(grid)
    window = lots.window()
    top, left, height, width = window
    held = np.zeros(grid.shape, dtype=bool)
    held[top : top + height, left : left + width] = lots.in_lots(window, lots.owned)
    return held


def lands_map() -> TileMap:
    """The whole place, the square the game starts on and the land for sale around it."""
    grid = place_grid()
    held = square(grid)
    rows, cols = np.indices(grid.shape)
    noise = _noise(grid.shape, NOISE_SEED, NOISE_CELLS)
    band_cuts = _thirds(held.sum(axis=1))
    band = sum((rows > _sway(cols, cut, k)).astype(int) for k, cut in enumerate(band_cuts))
    cover_cuts = [_thirds((held & (band == b)).sum(axis=0)) for b in range(len(BANDS))]
    cover = np.zeros(grid.shape, dtype=int)
    for b, cuts in enumerate(cover_cuts):
        across = sum((cols > _sway(rows, cut, 3 * b + k)).astype(int) for k, cut in enumerate(cuts))
        cover[band == b] = across[band == b]
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    for k, cut in enumerate(band_cuts):
        terrain[_across(grid, cut, k)] = Terrain.RIVER
    for b, cuts in enumerate(cover_cuts):
        top = 0 if b == 0 else round(band_cuts[b - 1]) - 3
        bottom = grid.rows - 1 if b == len(BANDS) - 1 else round(band_cuts[b]) + 3
        for k, cut in enumerate(cuts):
            terrain[_down(grid, cut, 3 * b + k, top, bottom)] = Terrain.RIVER
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    stream = np.zeros(grid.shape, dtype=np.uint8)
    land = band * len(COVERS) + cover
    for which in range(len(BANDS) * len(COVERS)):
        _shape_land(grid, held & (land == which), land == which, noise, terrain, relief, stream)
    water = terrain != Terrain.LAND
    relief[water] = Relief.FLAT
    stream[water] = 0
    temperature = np.array(BANDS, dtype=np.uint8)[band]
    vegetation = np.array(COVERS, dtype=np.uint8)[cover]
    vegetation[water] = Vegetation.BARREN
    return TileMap(
        name="Every land",
        grid=grid,
        bounds=BOUNDS,
        terrain=terrain,
        vegetation=vegetation,
        temperature=temperature,
        height_m=_heights(terrain, relief, noise),
        relief=relief,
        stream=stream,
    )


def _shape_land(
    grid: HexGrid,
    held: npt.NDArray[np.bool_],
    whole: npt.NDArray[np.bool_],
    noise: npt.NDArray[np.float64],
    terrain: npt.NDArray[np.uint8],
    relief: npt.NDArray[np.uint8],
    stream: npt.NDArray[np.uint8],
) -> None:
    """Raises one land's mountains and hills, fills its lake and bay and runs its minor rivers,
    by the share of its cells in the square."""
    cells = np.flatnonzero(held & (terrain == Terrain.LAND))
    if not cells.size:
        return
    flat_relief, flat_terrain = relief.reshape(-1), terrain.reshape(-1)
    share, at = MOUNTAINS
    peak = _middle(grid, cells, at)
    mountains = _blob(grid, cells, peak, share, noise)
    hills = _blob(grid, cells, peak, share + HILLS_ROUND_MOUNTAINS, noise)
    flat_relief[hills] = Relief.HILLS
    flat_relief[mountains] = Relief.MOUNTAINS
    share, at = MORE_HILLS
    plains = cells[flat_relief[cells] == Relief.FLAT]
    flat_relief[_blob(grid, plains, _middle(grid, cells, at), share, noise, len(cells))] = (
        Relief.HILLS
    )
    plains = cells[flat_relief[cells] == Relief.FLAT]
    share, at = LAKE
    lake_middle = _middle(grid, plains, at)
    flat_terrain[_blob(grid, plains, lake_middle, share, noise, len(cells))] = Terrain.LAKE
    plains = cells[(flat_relief[cells] == Relief.FLAT) & (flat_terrain[cells] == Terrain.LAND)]
    share, at = BAY
    flat_terrain[_blob(grid, plains, _middle(grid, plains, at), share, noise, len(cells))] = (
        Terrain.SEA
    )
    peak_cell = divmod(int(peak), grid.cols)
    lake_cell = divmod(int(lake_middle), grid.cols)
    down = (peak_cell[0], peak_cell[1] - STREAM_ASIDE)
    west = (lake_cell[0] + STREAM_ASIDE, lake_cell[1])
    _run_stream(grid, _stream_down(down, grid), whole, terrain, relief, stream)
    _run_stream(grid, _stream_west(west, grid), whole, terrain, relief, stream)


def _thirds(counts: npt.NDArray[np.int64]) -> tuple[float, float]:
    """The two indices that split these counts, in order, into three equal shares."""
    total = np.cumsum(counts)
    return float(np.searchsorted(total, total[-1] / 3)), float(
        np.searchsorted(total, total[-1] * 2 / 3)
    )


def _sway(along: npt.NDArray[np.int64] | int, line: float, phase: int) -> npt.NDArray[np.float64]:
    """Where a big river on a line lies at so many cells along it."""
    sway, every = RIVER_SWAY
    return line + sway * np.sin(np.asarray(along) / every + phase)


def _across(grid: HexGrid, line: float, phase: int) -> npt.NDArray[np.bool_]:
    """A big river from west to east about a row, one cell wide, each cell beside the last."""
    river = np.zeros(grid.shape, dtype=bool)
    last = None
    for col in range(grid.cols):
        at = round(float(_sway(col, line, phase)))
        low, high = (at, at) if last is None else (min(last, at), max(last, at))
        river[max(low, 0) : min(high, grid.rows - 1) + 1, col] = True
        last = at
    return river


def _down(grid: HexGrid, line: float, phase: int, top: int, bottom: int) -> npt.NDArray[np.bool_]:
    """A big river from north to south about a column, from row `top` to `bottom`."""
    river = np.zeros(grid.shape, dtype=bool)
    last = None
    for row in range(max(top, 0), min(bottom, grid.rows - 1) + 1):
        at = round(float(_sway(row, line, phase)))
        low, high = (at, at) if last is None else (min(last, at), max(last, at))
        river[row, max(low, 0) : min(high, grid.cols - 1) + 1] = True
        last = at
    return river


def _middle(grid: HexGrid, cells: npt.NDArray[np.intp], at: tuple[float, float]) -> int:
    """Of these flat cells, the one nearest to a point given as shares of the way down and
    across them: as far from their middle as the same point on a box they filled evenly, which
    keeps it inside a land the corner of the square cuts."""
    rows, cols = np.divmod(cells, grid.cols)
    row = rows.mean() + (at[0] - 0.5) * np.sqrt(12) * rows.std()
    col = cols.mean() + (at[1] - 0.5) * np.sqrt(12) * cols.std()
    return int(cells[np.argmin((rows - row) ** 2 + (cols - col) ** 2)])


def _blob(
    grid: HexGrid,
    cells: npt.NDArray[np.intp],
    middle: int,
    share: float,
    noise: npt.NDArray[np.float64],
    of: int | None = None,
) -> npt.NDArray[np.intp]:
    """Of these flat cells, the `share` of `of` of them (all of them when not given) nearest to
    a middle one, its edge made ragged by the noise."""
    xs, ys = grid.centers()
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    far = np.hypot(xs[cells] - xs[middle], ys[cells] - ys[middle])
    far = far * (0.75 + 0.5 * noise.reshape(-1)[cells])
    wanted = round(share * (len(cells) if of is None else of))
    return cells[np.argsort(far, kind="stable")[:wanted]]


def _stream_down(start: Cell, grid: HexGrid) -> list[Cell]:
    """The cells of a minor river from a cell down to the bottom of the place, in order."""
    sway, every = STREAM_SWAY
    course: list[Cell] = []
    last = start[1]
    for row in range(start[0], grid.rows):
        at = round(start[1] + sway * np.sin((row - start[0]) / every))
        step = 1 if at >= last else -1
        course += [(row, col) for col in range(last, at + step, step)] if row > start[0] else []
        course.append((row, at))
        last = at
    return [cell for cell in dict.fromkeys(course) if grid.contains(*cell)]


def _stream_west(start: Cell, grid: HexGrid) -> list[Cell]:
    """The cells of a minor river from a cell west to the edge of the place, in order."""
    sway, every = STREAM_SWAY
    course: list[Cell] = []
    last = start[0]
    for col in range(start[1], -1, -1):
        at = round(start[0] + sway * np.sin((start[1] - col) / every))
        step = 1 if at >= last else -1
        course += [(row, col) for row in range(last, at + step, step)] if col < start[1] else []
        course.append((at, col))
        last = at
    return [cell for cell in dict.fromkeys(course) if grid.contains(*cell)]


def _run_stream(
    grid: HexGrid,
    course: list[Cell],
    whole: npt.NDArray[np.bool_],
    terrain: npt.NDArray[np.uint8],
    relief: npt.NDArray[np.uint8],
    stream: npt.NDArray[np.uint8],
) -> None:
    """Lets a minor river flow along its course through the land's plains and hills, from where
    it leaves its mountains or lake to the first water past them or the edge of the land."""
    flowing: list[Cell] = []
    for cell in course:
        if not whole[cell]:
            break
        if terrain[cell] != Terrain.LAND:
            if flowing:
                flowing.append(cell)
                break
            continue
        if relief[cell] == Relief.MOUNTAINS:
            if flowing:
                break
            continue
        flowing.append(cell)
    for here, there in itertools.pairwise(flowing):
        _link(grid, here, there, stream, terrain)


def _link(
    grid: HexGrid,
    here: Cell,
    there: Cell,
    stream: npt.NDArray[np.uint8],
    terrain: npt.NDArray[np.uint8],
) -> None:
    """Joins two cells beside each other by a minor river, marking the edge between them in each
    one on land."""
    if there not in grid.neighbours(*here):
        return
    normals = grid.edge_normals()
    x0, y0 = grid.center(*here)
    x1, y1 = grid.center(*there)
    towards = int(np.argmax(normals @ np.array([x1 - x0, y1 - y0])))
    back = (towards + len(normals) // 2) % len(normals)
    for cell, bit in ((here, towards), (there, back)):
        if terrain[cell] == Terrain.LAND:
            stream[cell] |= np.uint8(1 << bit)


def _heights(
    terrain: npt.NDArray[np.uint8],
    relief: npt.NDArray[np.uint8],
    noise: npt.NDArray[np.float64],
) -> npt.NDArray[np.int16]:
    """Each cell's height in metres, rolling a little on the plains and more in the hills."""
    height = np.select(
        [relief == Relief.MOUNTAINS, relief == Relief.HILLS],
        [MOUNTAINS_M[0] + MOUNTAINS_M[1] * noise, HILLS_M[0] + HILLS_M[1] * noise],
        PLAINS_M[0] + PLAINS_M[1] * noise,
    )
    height = np.where(terrain == Terrain.RIVER, RIVER_M, height)
    height = np.where(terrain == Terrain.LAKE, LAKE_M, height)
    height = np.where(terrain == Terrain.SEA, 0, height)
    return height.astype(np.int16)


def _noise(shape: tuple[int, int], seed: int, size: float) -> npt.NDArray[np.float64]:
    """Smooth noise from 0 to 1 with features about `size` cells across."""
    rng = np.random.default_rng(seed)
    coarse = rng.random((int(shape[0] / size) + 2, int(shape[1] / size) + 2))

    def smooth(n: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        at = np.arange(n) / size
        whole = at.astype(int)
        part = at - whole
        return whole, part * part * (3 - 2 * part)

    (row, down), (col, across) = smooth(shape[0]), smooth(shape[1])
    top = coarse[row][:, col] * (1 - across) + coarse[row][:, col + 1] * across
    bottom = coarse[row + 1][:, col] * (1 - across) + coarse[row + 1][:, col + 1] * across
    return top * (1 - down[:, None]) + bottom * down[:, None]
