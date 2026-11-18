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
# How far out from its station Ashford's farms and Brookby's houses reach.
FARMS_REACH = 4
TOWN_REACH = 2
# The cells left free between the two stations, for the track.
LINE = frozenset(
    (ASHFORD_STATION[0], col) for col in range(ASHFORD_STATION[1] + 1, BROOKBY_STATION[1])
)

H, T, F = BuildingKind.HOUSE, BuildingKind.TENEMENT, BuildingKind.FARM


def _layout() -> dict[Cell, tuple[BuildingKind, ...]]:
    """Ashford: a farming village, a few houses and a bakery among fields that make more food
    than it eats. Brookby: houses and tenements round a free cell, with no farm near enough to
    feed them, so they want the food trains bring. A coal mine in the hills and a lumber mill
    in the forest."""
    grid = HexGrid(rows=MAP_ROWS, cols=MAP_COLS, km_per_cell=KM_PER_CELL)
    layout: dict[Cell, tuple[BuildingKind, ...]] = {
        on_map((14, 7)): (H, H),
        on_map((14, 8)): (H, BuildingKind.BAKERY),
    }
    for cell, steps in sorted(grid.within(ASHFORD_STATION, FARMS_REACH).items()):
        if steps and cell not in layout and cell not in LINE:
            layout[cell] = (F,)
    town = sorted(grid.within(BROOKBY_STATION, TOWN_REACH).items())
    for cell, steps in town:
        if steps and cell not in LINE:
            layout[cell] = (H, H, H) if steps == 1 else (H, H)
    for cell in [cell for cell, steps in town if steps == 1][:3]:
        layout[cell] = (T, H, H)
    layout[on_map((5, 6))] = (BuildingKind.MINE,)
    layout[on_map((6, 24))] = (BuildingKind.LUMBER_MILL,)
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
    rows, cols = np.indices(shape)
    terrain = np.full(shape, Terrain.LAND, dtype=np.uint8)
    vegetation = np.full(shape, Vegetation.GRASS, dtype=np.uint8)
    relief = np.full(shape, Relief.FLAT, dtype=np.uint8)
    height = np.full(shape, 80, dtype=np.int16)

    def square(top: int, left: int, bottom: int, right: int) -> np.ndarray:
        """Square cells from (top, left) to (bottom, right), both included, on the place."""
        return (
            (rows >= top + SQUARE_TOP)
            & (rows <= bottom + SQUARE_TOP)
            & (cols >= left + SQUARE_LEFT)
            & (cols <= right + SQUARE_LEFT)
        )

    hills = square(1, 1, 8, 11) | square(-14, -14, -4, 6) | square(36, 30, 44, 44)
    relief[hills] = Relief.HILLS
    height[hills] = 260
    forest = (
        square(1, 21, 9, 31)
        | square(-12, 18, -2, 40)
        | square(10, 36, 26, 46)
        | square(34, -14, 46, 4)
    )
    vegetation[forest] = Vegetation.FOREST
    lake = square(22, 12, 25, 17) | square(-10, -10, -6, -4)
    terrain[lake] = Terrain.LAKE
    river = (cols == 14 + SQUARE_LEFT) & (rows >= 26 + SQUARE_TOP)
    terrain[river] = Terrain.RIVER
    vegetation[terrain != Terrain.LAND] = Vegetation.BARREN
    height[terrain != Terrain.LAND] = 70
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
