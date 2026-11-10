import numpy as np
import pytest

from railroads.sim.constructions import (
    Building,
    BuildingKind,
    CannotBuild,
    Constructions,
    Station,
    Track,
    TrackWidth,
)
from railroads.sim.geo import Bounds
from railroads.sim.grids import Grid, HexGrid, SquareGrid
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.sim.track_rules import TrackBed

LEGEND = {
    ".": (Terrain.LAND, Relief.FLAT),
    "h": (Terrain.LAND, Relief.HILLS),
    "m": (Terrain.LAND, Relief.MOUNTAINS),
    "r": (Terrain.RIVER, Relief.FLAT),
    "l": (Terrain.LAKE, Relief.FLAT),
    "s": (Terrain.SEA, Relief.FLAT),
}
WIDE_LAND = ["....."] * 5


def make_map(*rows: str, grid_type: type[Grid] = HexGrid) -> TileMap:
    cells = [[LEGEND[letter] for letter in row] for row in rows]
    terrain = np.array([[terrain for terrain, _ in row] for row in cells], dtype=np.uint8)
    relief = np.array([[relief for _, relief in row] for row in cells], dtype=np.uint8)
    grid = grid_type(rows=terrain.shape[0], cols=terrain.shape[1], km_per_cell=2.0)
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=terrain,
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=relief,
    )


@pytest.mark.parametrize(("width", "houses"), [(TrackWidth.SINGLE, 5), (TrackWidth.DOUBLE, 4)])
def test_track_leaves_the_rest_of_the_cell_for_houses(width: TrackWidth, houses: int) -> None:
    constructions = Constructions(make_map("..."))
    constructions.build(Track((0, 1), (0, 6), width))

    for _ in range(houses):
        constructions.build(Building((0, 1), BuildingKind.HOUSE))

    assert constructions.space_left((0, 1)) == 0
    assert constructions.why_not(Building((0, 1), BuildingKind.HOUSE)) == (
        "Not enough room: needs 1, 0 left"
    )


def test_a_station_fills_its_whole_cell() -> None:
    constructions = Constructions(make_map("..."))

    constructions.build(Station((0, 1)))

    assert constructions.space_left((0, 1)) == 0
    assert constructions.why_not(Track((0, 1), (0, 6))) == "Not enough room: needs 1, 0 left"


def test_a_station_needs_an_empty_cell() -> None:
    constructions = Constructions(make_map("..."))
    constructions.build(Building((0, 1), BuildingKind.HOUSE))

    problem = constructions.why_not(Station((0, 1)))

    assert problem == "Not enough room: needs 6, 5 left"


@pytest.mark.parametrize(
    ("cells", "problem"),
    [
        (".r.", "Track over water needs a bridge"),
        (".l.", "Track over water needs a bridge"),
        (".s.", "Track cannot be laid in the sea"),
        (".m.", "Track through mountains needs a tunnel"),
        (".h.", None),
    ],
)
def test_track_on_the_ground_needs_land_off_the_mountains(cells: str, problem: str | None) -> None:
    constructions = Constructions(make_map(cells))

    found = constructions.why_not(Track((0, 1), (0, 6)))

    assert found == problem


@pytest.mark.parametrize(
    ("cells", "problem"),
    [
        (".r.", None),
        (".l.", None),
        (".s.", "Bridges go over rivers and lakes"),
        ("...", "Bridges go over rivers and lakes"),
    ],
)
def test_bridges_go_over_rivers_and_lakes(cells: str, problem: str | None) -> None:
    constructions = Constructions(make_map(cells))

    found = constructions.why_not(Track((0, 1), (0, 6), TrackWidth.DOUBLE, TrackBed.BRIDGE))

    assert found == problem


@pytest.mark.parametrize(
    ("cells", "problem"),
    [
        (".m.", None),
        (".h.", "Tunnels go through mountains"),
        (".r.", "Tunnels go through mountains"),
    ],
)
def test_tunnels_go_through_mountains(cells: str, problem: str | None) -> None:
    constructions = Constructions(make_map(cells))

    found = constructions.why_not(Track((0, 1), (0, 6), bed=TrackBed.TUNNEL))

    assert found == problem


@pytest.mark.parametrize(
    ("grid_type", "end", "allowed"),
    [
        (HexGrid, 6, True),
        (HexGrid, 5, True),
        (HexGrid, 4, True),
        (HexGrid, 3, False),
        (SquareGrid, 4, True),
        (SquareGrid, 3, True),
        (SquareGrid, 2, False),
    ],
    ids=["hex-straight", "hex-30", "hex-60", "hex-90", "square-straight", "square-45", "square-90"],
)
def test_track_turns_at_most_sixty_degrees_in_a_cell(
    grid_type: type[Grid], end: int, allowed: bool
) -> None:
    constructions = Constructions(make_map(*WIDE_LAND, grid_type=grid_type))

    problem = constructions.why_not(Track((2, 2), (0, end)))

    assert (problem is None) == allowed


@pytest.mark.parametrize("ends", [(3, 3), (0, 12), (-1, 5)])
def test_track_joins_two_different_track_points(ends: tuple[int, int]) -> None:
    constructions = Constructions(make_map(*WIDE_LAND))

    problem = constructions.why_not(Track((2, 2), ends))

    assert problem == "Track joins two different track points, numbered 0 to 11"


def test_track_cannot_run_off_the_map() -> None:
    constructions = Constructions(make_map("..."))

    problem = constructions.why_not(Track((0, 0), (0, 6)))

    assert problem == "Track cannot run off the map"


def test_track_through_a_corner_needs_the_cells_on_both_sides_to_take_it() -> None:
    constructions = Constructions(make_map(".....", ".....", "...l.", ".....", "....."))

    beside_the_lake = constructions.why_not(Track((2, 2), (1, 7)))
    away_from_it = constructions.why_not(Track((2, 2), (3, 9)))

    assert beside_the_lake == "Track through a corner needs the cells on both sides to take it"
    assert away_from_it is None


def test_the_same_track_cannot_be_laid_twice() -> None:
    constructions = Constructions(make_map("..."))
    constructions.build(Track((0, 1), (0, 6)))

    problem = constructions.why_not(Track((0, 1), (6, 0), TrackWidth.DOUBLE))

    assert problem == "There is track between those points already"


def test_a_port_needs_water_next_to_it() -> None:
    constructions = Constructions(make_map("s...."))

    by_the_sea = constructions.why_not(Building((0, 1), BuildingKind.PORT))
    inland = constructions.why_not(Building((0, 3), BuildingKind.PORT))

    assert by_the_sea is None
    assert inland == "A port needs water next to it"


def test_only_mines_and_town_buildings_can_be_built_in_mountains() -> None:
    constructions = Constructions(make_map("m"))

    allowed = {
        kind for kind in BuildingKind if constructions.why_not(Building((0, 0), kind)) is None
    }

    assert allowed == {
        BuildingKind.MINE,
        BuildingKind.IRON_MINE,
        BuildingKind.COPPER_MINE,
        BuildingKind.GOLD_MINE,
        BuildingKind.DIAMOND_MINE,
        BuildingKind.HOUSE,
        BuildingKind.TENEMENT,
        BuildingKind.FLATS,
        BuildingKind.SKYSCRAPER,
        BuildingKind.BAKERY,
        BuildingKind.TAVERN,
        BuildingKind.DEPARTMENT_STORE,
        BuildingKind.HOTEL,
        BuildingKind.LIMESTONE_QUARRY,
    }
    assert constructions.why_not(Station((0, 0))) == "A station cannot be built on mountains"


def test_stations_and_buildings_cannot_be_built_on_water() -> None:
    constructions = Constructions(make_map("l"))

    problems = [
        constructions.why_not(Station((0, 0))),
        constructions.why_not(Building((0, 0), BuildingKind.MINE)),
    ]

    assert problems == ["A station cannot be built on water", "A mine cannot be built on water"]


def test_nothing_can_be_built_off_the_map() -> None:
    constructions = Constructions(make_map("..."))

    problem = constructions.why_not(Building((1, 0), BuildingKind.HOUSE))

    assert problem == "That is off the map"


def test_building_where_it_is_not_allowed_raises() -> None:
    constructions = Constructions(make_map("s"))

    with pytest.raises(CannotBuild, match="on water"):
        constructions.build(Station((0, 0)))


def test_removing_a_construction_frees_its_room() -> None:
    constructions = Constructions(make_map("..."))
    track = Track((0, 1), (0, 6), TrackWidth.DOUBLE)
    constructions.build(track)

    constructions.remove(track)

    assert constructions.space_left((0, 1)) == 6
    assert list(constructions) == []


def test_a_cell_is_electrified_when_all_its_track_is_electric() -> None:
    constructions = Constructions(make_map(*WIDE_LAND))
    constructions.build(Track((2, 2), (0, 6), electric=True))
    wired = constructions.electrified((2, 2))
    constructions.build(Track((2, 2), (3, 9)))

    assert wired
    assert not constructions.electrified((2, 2))
    assert constructions.electrified((2, 2), (6, 0))
    assert not constructions.electrified((2, 2), (3, 9))
    assert not constructions.electrified((2, 3))
