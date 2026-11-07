import numpy as np

from railroads.sim.constructions import Building, BuildingKind, Constructions, Station, Track
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth, GrowthRule
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.sim.track_rules import TrackBed

NO_TRADE = GrowthRule(trades=False)


def make_map(river_col: int | None = None, mountain_col: int | None = None) -> TileMap:
    grid = HexGrid(rows=12, cols=20, km_per_cell=2.0)
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    if river_col is not None:
        terrain[:, river_col] = Terrain.RIVER
    if mountain_col is not None:
        relief[:, mountain_col] = Relief.MOUNTAINS
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


def test_a_route_over_open_land_runs_cell_to_cell_and_can_all_be_laid() -> None:
    constructions = Constructions(make_map())
    growth = Growth(constructions, NO_TRADE)

    route = plan_route(constructions, (6, 2), (6, 12))

    assert route is not None
    assert route.cells[0] == (6, 2) and route.cells[-1] == (6, 12)
    assert len(route.pieces) == len(route.cells)
    assert all(constructions.why_not(piece) is None for piece in route.pieces)
    lay_route(growth, route)
    laid = [built for built in constructions if isinstance(built, Track)]
    assert len(laid) == len(route.pieces)


def test_rivers_are_bridged_and_mountains_tunnelled() -> None:
    constructions = Constructions(make_map(river_col=6, mountain_col=10))

    route = plan_route(constructions, (6, 2), (6, 14))

    assert route is not None
    beds = {piece.bed for piece in route.pieces}
    assert {TrackBed.BRIDGE, TrackBed.TUNNEL} <= beds
    assert all(constructions.why_not(piece) is None for piece in route.pieces)


def test_track_keeps_to_empty_land_when_it_can() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    for col in range(4, 11):
        for _ in range(6):
            constructions.build(Building((6, col), BuildingKind.HOUSE))

    route = plan_route(constructions, (6, 2), (6, 12))

    assert route is not None
    assert not any(constructions.space_left(cell) == 0 for cell in route.cells)


def test_laying_track_moves_buildings_aside_or_pulls_down_the_lowest() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    for _ in range(5):
        constructions.build(Building((6, 5), BuildingKind.HOUSE))
    constructions.build(Building((6, 5), BuildingKind.FLATS))
    growth = Growth(constructions, NO_TRADE)

    changed = growth.clear_room((6, 5), 1)

    houses_here = sum(1 for b in constructions.at((6, 5)) if b.kind == BuildingKind.HOUSE)
    houses = sum(
        1 for b in constructions if isinstance(b, Building) and b.kind == BuildingKind.HOUSE
    )
    assert constructions.space_left((6, 5)) == 1
    assert houses_here == 4 and houses == 5
    assert len(changed) == 2


def test_with_nowhere_to_move_the_lowest_building_comes_down() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    for _ in range(5):
        constructions.build(Building((6, 5), BuildingKind.FLATS))
    constructions.build(Building((6, 5), BuildingKind.HOUSE))
    for near in tile_map.grid.neighbours(6, 5):
        for _ in range(6):
            constructions.build(Building(near, BuildingKind.FARM))
    growth = Growth(constructions, NO_TRADE)

    growth.clear_room((6, 5), 1)

    kinds = [b.kind for b in constructions.at((6, 5))]
    assert kinds.count(BuildingKind.FLATS) == 5
    assert BuildingKind.HOUSE not in kinds
    assert growth.count(BuildingKind.HOUSE).sum() == 0


def test_a_station_clears_its_cell_and_track_joins_it_without_entering() -> None:
    constructions = Constructions(make_map())
    for _ in range(3):
        constructions.build(Building((6, 12), BuildingKind.HOUSE))
    growth = Growth(constructions, NO_TRADE)

    build_station(growth, (6, 12))
    route = plan_route(constructions, (6, 2), (6, 12))

    assert Station((6, 12)) in constructions.at((6, 12))
    assert growth.count(BuildingKind.HOUSE).sum() == 3
    assert route is not None
    assert all(piece.cell != (6, 12) for piece in route.pieces)


def test_no_route_into_the_sea() -> None:
    tile_map = make_map()
    terrain = tile_map.terrain.copy()
    constructions = Constructions(tile_map)

    route = plan_route(constructions, (6, 2), (6, 2))

    assert route is None
    assert (terrain == tile_map.terrain).all()


def test_track_runs_straight_through_corners_where_edges_would_zigzag() -> None:
    constructions = Constructions(make_map())

    route = plan_route(constructions, (2, 8), (8, 8))

    assert route is not None
    assert len(route.pieces) == 4
    assert all(constructions.why_not(piece) is None for piece in route.pieces)


def test_cells_within_steps_form_hexagonal_rings() -> None:
    grid = make_map().grid

    reached = grid.within((6, 10), 2)

    counts = [sum(1 for steps in reached.values() if steps == ring) for ring in range(3)]
    assert counts == [1, 6, 12]
