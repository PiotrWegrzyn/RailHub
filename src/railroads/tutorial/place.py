"""The tutorial's place: a small map drawn by hand, not grown from a seed, so every player sees the
same land, towns and buildings.

The place is PLACE_SIZE cells a side. The land is drawn on the SQUARE_SIZE square in its middle,
in (row, column) cells of the square; the game plays on the honeycomb of seven lots in the middle
of the place, like any game (railroads.sim.lots), which holds the whole drawing, and the land
around it is for sale. Two towns stand on it, Ashford in the west and Brookby in the east, with a
coal mine in the hills to the north-west and a lumber mill in the forest to the north-east. The
cells below are cells of the game's map, the honeycomb's window on the place (MAP_TOP, MAP_LEFT).
"""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

import numpy as np

from railroads.places import Places, StationNames
from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.geo import KM_PER_DEGREE_LAT, Bounds
from railroads.sim.grids import Cell, HexGrid
from railroads.sim.lots import Lots
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation

PLACE_SIZE = 64
SQUARE_SIZE = 32
SQUARE_TOP = SQUARE_LEFT = (PLACE_SIZE - SQUARE_SIZE) // 2


def start_lots() -> Lots:
    """The lots of the tutorial's place, the seven in its middle the game's own."""
    grid = HexGrid(rows=PLACE_SIZE, cols=PLACE_SIZE, km_per_cell=2.0)
    rows, cols = grid.shape
    return Lots(radius=Lots.radius_for(rows * cols), middle=(rows // 2 // 2 * 2, cols // 2))


MAP_TOP, MAP_LEFT, MAP_ROWS, MAP_COLS = start_lots().window()


def on_map(cell: Cell) -> Cell:
    """The game's map cell a cell of the drawn square is."""
    return cell[0] + SQUARE_TOP - MAP_TOP, cell[1] + SQUARE_LEFT - MAP_LEFT


KM_PER_CELL = 2.0
# Where the place lies in the world: the English Midlands, for the minimap's world map.
CENTER_LON, CENTER_LAT = -1.5, 52.5
# The year play begins in, and what history the map pretends to have.
START_YEAR = 1850

ASHFORD = on_map((15, 8))
BROOKBY = on_map((16, 21))
# Where the tutorial asks for the two stations: in the middle of Ashford's farms and of
# Brookby's houses.
ASHFORD_STATION = on_map((16, 8))
BROOKBY_STATION = BROOKBY
# The farm whose details and shares the tutorial points at.
SHOWN_FARM = on_map((17, 8))
# The cells left free between the two stations, for the track.
LINE = frozenset(
    (ASHFORD_STATION[0], col) for col in range(ASHFORD_STATION[1] + 1, BROOKBY_STATION[1])
)
# How far out from its station Ashford's fields reach, and Brookby's houses.
FARMS_REACH = 5
TOWN_REACH = 3
# The chance a cell that far out from its station is built up: fields thin out towards the
# edge of the farmland and houses towards the edge of the town, so neither ends in a clean ring.
FIELD_CHANCE = (1.0, 1.0, 0.95, 0.85, 0.6, 0.35)
HOUSE_CHANCE = (1.0, 1.0, 0.8, 0.4)
# A field is left out where this many people live around it, as growth would pull it down.
CROWDED_FIELD = 8
# The random numbers that pick which cells are built up, the same every time.
LAYOUT_SEED = 1066
# Where the mine, the lumber mill and the hamlets beside them stand, on the drawn square.
MINE = (5, 6)
MILL = (6, 24)

H, T, F = BuildingKind.HOUSE, BuildingKind.TENEMENT, BuildingKind.FARM


def _grid() -> HexGrid:
    return HexGrid(rows=MAP_ROWS, cols=MAP_COLS, km_per_cell=KM_PER_CELL)


def _layout() -> dict[Cell, tuple[BuildingKind, ...]]:
    """Ashford: a farming village of a few houses, a bakery and a church, among fields that make
    more food than it eats. Brookby: a market town of houses and tenements round a free cell,
    thinning out at its edges, with no fields near enough to feed it, so it wants the food
    trains bring. A coal mine in the hills and a lumber mill in the forest, each with a hamlet."""
    grid = _grid()
    rng = np.random.default_rng(LAYOUT_SEED)
    village = {
        on_map((14, 7)): (H, H, BuildingKind.CHURCH),
        on_map((13, 7)): (H, H),
        on_map((14, 8)): (H, BuildingKind.BAKERY),
        on_map((13, 6)): (H,),
    }
    layout: dict[Cell, tuple[BuildingKind, ...]] = dict(village)
    people = {cell: kinds.count(H) for cell, kinds in village.items()}
    for cell, steps in sorted(grid.within(ASHFORD_STATION, FARMS_REACH).items()):
        if not steps or cell in layout or cell in LINE or rng.random() >= FIELD_CHANCE[steps]:
            continue
        around = sum(people.get(near, 0) for near in [cell, *grid.neighbours(*cell)])
        if around < CROWDED_FIELD:
            layout[cell] = (F,)
    town = sorted(grid.within(BROOKBY_STATION, TOWN_REACH).items())
    tenements = 0
    for cell, steps in town:
        if not steps or cell in LINE or rng.random() >= HOUSE_CHANCE[steps]:
            continue
        houses = int(rng.integers(4 - steps, 6 - steps))
        if steps == 1 and tenements < 3:
            tenements += 1
            layout[cell] = (T,) + (H,) * (houses - 1)
        else:
            layout[cell] = (H,) * houses
    church = next(cell for cell, steps in town if steps == 2 and cell in layout)
    layout[church] = (BuildingKind.CHURCH,) + layout[church][1:]
    layout[on_map(MINE)] = (BuildingKind.MINE,)
    layout[on_map((6, 7))] = (H, H)
    layout[on_map(MILL)] = (BuildingKind.LUMBER_MILL,)
    layout[on_map((7, 23))] = (H,)
    return layout


BUILDINGS: Mapping[Cell, tuple[BuildingKind, ...]] = MappingProxyType(_layout())
# The places stations are named after, at map cells, with how many people each has.
PLACES: tuple[tuple[str, Cell, int], ...] = (
    ("Ashford", ASHFORD, 3_000),
    ("Brookby", BROOKBY, 12_000),
    ("Millfield", on_map((26, 22)), 1_000),
    ("Coalbrook", on_map((5, 6)), 2_000),
    ("Elmwood", on_map((6, 25)), 2_000),
)


def on_place(cell: Cell) -> Cell:
    """The place's cell a cell of the game's map is."""
    return cell[0] + MAP_TOP, cell[1] + MAP_LEFT


# The land is drawn from smooth noise of fixed random numbers, so it has the ragged shapes of real
# land and is the same every time: hills where one layer is high, mountains where it is higher
# still and far from the towns, forest and lakes where two others are, and gentle rises in the
# flat land from a fourth. Each layer is the seed's offset and the size of its features, in cells.
NOISE_SEED = 1850
NOISE = ((0, 9.0), (1, 6.0), (2, 8.0), (3, 4.0))
HILLS_ABOVE = 0.68
MOUNTAINS_ABOVE = 0.82
MOUNTAINS_AWAY = 11.0
FOREST_ABOVE = 0.62
LAKES_ABOVE = 0.86
# A lake south of the line between the towns, on the drawn square; and the rivers, each from a
# cell of the drawn square down to the bottom of the place, swaying by so many cells every so
# many rows: one out of the lake, and one past Brookby to the east, from the top of the place.
LAKE = (24, 16)
RIVERS = (((26, 16), 2.0, 3.5), ((-SQUARE_TOP, 29), 1.6, 4.5))


def _noise(shape: tuple[int, int], seed: int, size: float) -> np.ndarray:
    """Smooth noise from 0 to 1 with features about `size` cells across."""
    rng = np.random.default_rng(seed)
    coarse = rng.random((int(shape[0] / size) + 2, int(shape[1] / size) + 2))

    def smooth(n: int) -> tuple[np.ndarray, np.ndarray]:
        at = np.arange(n) / size
        whole = at.astype(int)
        part = at - whole
        return whole, part * part * (3 - 2 * part)

    (row, down), (col, across) = smooth(shape[0]), smooth(shape[1])
    top = coarse[row][:, col] * (1 - across) + coarse[row][:, col + 1] * across
    bottom = coarse[row + 1][:, col] * (1 - across) + coarse[row + 1][:, col + 1] * across
    return top * (1 - down[:, None]) + bottom * down[:, None]


def _drawn(cell: Cell) -> Cell:
    """The cell of the drawn square a cell of the game's map is."""
    return cell[0] - SQUARE_TOP + MAP_TOP, cell[1] - SQUARE_LEFT + MAP_LEFT


def _between(grid: HexGrid, west: Cell, east: Cell) -> np.ndarray:
    """The place's cells along the row between two cells of the drawn square, and a row either
    side."""
    rows, cols = np.indices(grid.shape)
    row = west[0] + SQUARE_TOP
    return (
        (abs(rows - row) <= 1) & (cols >= west[1] + SQUARE_LEFT) & (cols <= east[1] + SQUARE_LEFT)
    )


def _river(course: tuple[Cell, float, float]) -> np.ndarray:
    """The place's cells a river runs through, from its source on the drawn square down to the
    bottom of the place, swaying from side to side."""
    (row, col), sway, every = course
    river = np.zeros((PLACE_SIZE, PLACE_SIZE), dtype=bool)
    last = None
    for place_row in range(row + SQUARE_TOP, PLACE_SIZE):
        at = round(col + SQUARE_LEFT + sway * np.sin((place_row - row - SQUARE_TOP) / every))
        low, high = (at, at) if last is None else (min(last, at), max(last, at))
        river[place_row, max(low, 0) : min(high, PLACE_SIZE - 1) + 1] = True
        last = at
    return river


def place_map() -> TileMap:
    """The whole place: the square the game starts on and the land for sale around it."""
    grid = HexGrid(rows=PLACE_SIZE, cols=PLACE_SIZE, km_per_cell=KM_PER_CELL)
    width_km, height_km = (side * KM_PER_CELL for side in grid.size())
    half_lon = width_km / 2 / (KM_PER_DEGREE_LAT * np.cos(np.radians(CENTER_LAT)))
    half_lat = height_km / 2 / KM_PER_DEGREE_LAT
    bounds = Bounds(
        west=CENTER_LON - half_lon,
        east=CENTER_LON + half_lon,
        south=CENTER_LAT - half_lat,
        north=CENTER_LAT + half_lat,
    )
    shape = grid.shape
    xs, ys = grid.centers()

    def blob(cell: Cell, radius: float, wobble: np.ndarray) -> np.ndarray:
        """The cells within about `radius` of a cell of the drawn square, its edge ragged."""
        x, y = grid.center(cell[0] + SQUARE_TOP, cell[1] + SQUARE_LEFT)
        return np.hypot(xs - x, ys - y) < radius * (0.75 + 0.5 * wobble)

    rugged, wooded, wet, gentle = (_noise(shape, NOISE_SEED + n, size) for n, size in NOISE)
    towns = (
        blob(_drawn(ASHFORD_STATION), FARMS_REACH + 1.5, np.full(shape, 0.5))
        | blob(_drawn(BROOKBY_STATION), TOWN_REACH + 1.5, np.full(shape, 0.5))
        | _between(grid, _drawn(ASHFORD_STATION), _drawn(BROOKBY_STATION))
    )
    far = ~(
        blob(_drawn(ASHFORD_STATION), MOUNTAINS_AWAY, np.full(shape, 0.5))
        | blob(_drawn(BROOKBY_STATION), MOUNTAINS_AWAY, np.full(shape, 0.5))
    )
    hills = ((rugged > HILLS_ABOVE) | blob(MINE, 3.5, rugged)) & ~towns
    mountains = (rugged > MOUNTAINS_ABOVE) & far
    relief = np.where(mountains, Relief.MOUNTAINS, np.where(hills, Relief.HILLS, Relief.FLAT))
    height = (
        60
        + 50 * gentle
        + np.where(hills | mountains, 120 + 900 * np.maximum(rugged - 0.5, 0.05), 0)
    )
    forest = ((wooded > FOREST_ABOVE) | blob(MILL, 3.5, wooded)) & ~towns
    terrain = np.full(shape, Terrain.LAND)
    terrain[(blob(LAKE, 3.0, wet) | ((wet > LAKES_ABOVE) & far)) & ~towns] = Terrain.LAKE
    for course in RIVERS:
        terrain[_river(course)] = Terrain.RIVER
    water = terrain != Terrain.LAND
    relief[water] = Relief.FLAT
    height[water] = 50
    vegetation = np.where(forest & ~water, Vegetation.FOREST, Vegetation.GRASS)
    vegetation[water] = Vegetation.BARREN
    terrain, vegetation, relief = (a.astype(np.uint8) for a in (terrain, vegetation, relief))
    height = height.astype(np.int16)
    return TileMap(
        name="Tutorial",
        grid=grid,
        bounds=bounds,
        terrain=terrain,
        vegetation=vegetation,
        temperature=np.full(shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=height,
        relief=relief,
        deposits=np.zeros(shape, dtype=np.uint8),
    )


def start_map(place: TileMap) -> TileMap:
    """The honeycomb in the middle of the place, which the game starts on, its corners locked."""
    return replace(
        place.window(MAP_TOP, MAP_LEFT, MAP_ROWS, MAP_COLS), locked=start_lots().locked()
    )


def buildings() -> list[Building]:
    """Every building the tutorial starts with, in a fixed order."""
    return [Building(cell, kind) for cell, kinds in BUILDINGS.items() for kind in kinds]


def station_names(tile_map: TileMap) -> StationNames:
    """Stations named after the tutorial's own places."""
    cells = [cell for _, cell, _ in PLACES]
    lonlat = np.array([tile_map.lonlat(cell) for cell in cells])
    places = Places(
        tuple(name for name, _, _ in PLACES),
        lonlat[:, 0],
        lonlat[:, 1],
        np.array([people for _, _, people in PLACES], dtype=np.int64),
    )
    return StationNames(tile_map, places)
