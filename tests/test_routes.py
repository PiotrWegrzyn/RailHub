import numpy as np
import pytest

from railroads.sim.constructions import (
    Building,
    BuildingKind,
    Constructions,
    Station,
    Track,
    TrackWidth,
)
from railroads.sim.geo import Bounds
from railroads.sim.grids import Cell, Grid, HexGrid, SquareGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import Route, build_station, lay_route, plan_route, route_price
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.sim.track_rules import TrackBed

NO_TRADE = GrowthRule(trades=False)


def make_map(
    river_col: int | None = None, mountain_col: int | None = None, grid: Grid | None = None
) -> TileMap:
    grid = grid or HexGrid(rows=12, cols=20, km_per_cell=2.0)
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


def laid(constructions: Constructions, start: Cell, end: Cell, **options: object) -> Route:
    route = plan_route(constructions, start, end, **options)  # type: ignore[arg-type]
    assert route is not None
    lay_route(Growth(constructions, NO_TRADE), route)
    return route


def track(constructions: Constructions) -> set[Track]:
    return {built for built in constructions if isinstance(built, Track)}


def dead_ends(constructions: Constructions) -> list[tuple[Cell, int]]:
    grid = constructions.tile_map.grid
    ends = []
    for piece in track(constructions):
        for point in piece.ends:
            beyond, entry = grid.track_exit(piece.cell, point)
            joined = any(
                isinstance(built, Station) or (isinstance(built, Track) and entry in built.ends)
                for built in constructions.at(beyond)
            )
            if not joined:
                ends.append((piece.cell, point))
    return ends


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

    changed = growth.room.clear((6, 5), 1)

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
        constructions.build(Building(near, BuildingKind.FARM))
    growth = Growth(constructions, NO_TRADE)

    growth.room.clear((6, 5), 1)

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


def test_on_a_square_grid_track_runs_diagonally_through_corners() -> None:
    constructions = Constructions(make_map(grid=SquareGrid(rows=12, cols=20, km_per_cell=2.0)))

    route = plan_route(constructions, (2, 2), (8, 8))

    assert route is not None
    assert route.cells == [(2 + step, 2 + step) for step in range(7)]
    assert all(constructions.why_not(piece) is None for piece in route.pieces)


def test_a_route_along_the_edge_of_the_map_keeps_to_it() -> None:
    constructions = Constructions(make_map())

    route = plan_route(constructions, (0, 2), (0, 17))

    assert route is not None
    assert route.cells == [(0, col) for col in range(2, 18)]
    assert all(constructions.why_not(piece) is None for piece in route.pieces)


def test_no_route_from_or_to_a_cell_off_the_map() -> None:
    constructions = Constructions(make_map())

    routes = [
        plan_route(constructions, (-1, 4), (6, 8)),
        plan_route(constructions, (6, 8), (6, 20)),
    ]

    assert routes == [None, None]


def test_cells_within_steps_form_hexagonal_rings() -> None:
    grid = make_map().grid

    reached = grid.within((6, 10), 2)

    counts = [sum(1 for steps in reached.values() if steps == ring) for ring in range(3)]
    assert counts == [1, 6, 12]


def test_track_laid_from_a_line_adds_only_the_branch_and_joins_the_line() -> None:
    constructions = Constructions(make_map())
    line = laid(constructions, (6, 2), (6, 10))
    before = track(constructions)

    branch = laid(constructions, (6, 5), (2, 8))

    added = track(constructions) - before
    assert {piece.cell for piece in added} <= set(branch.cells)
    assert sum(1 for piece in added if piece.cell in line.cells) == 1
    assert len(dead_ends(constructions)) == 3


def test_a_junction_cell_holds_the_line_and_a_branch_that_shares_one_end_with_it() -> None:
    constructions = Constructions(make_map())
    line = laid(constructions, (6, 2), (6, 10))

    laid(constructions, (6, 5), (2, 8))

    junctions = [cell for cell in line.cells if len(constructions.track_at(cell)) == 2]
    assert len(junctions) == 1
    through, branch = constructions.track_at(junctions[0])
    shared = set(through.ends) & set(branch.ends)
    assert len(shared) == 1
    assert constructions.tile_map.grid.track_turn(*branch.ends) <= 60 + 1e-6


def test_track_laid_on_from_a_dead_end_turns_it_instead_of_leaving_a_stub() -> None:
    constructions = Constructions(make_map())
    laid(constructions, (6, 2), (6, 6))
    route = plan_route(constructions, (6, 6), (2, 8))
    assert route is not None
    price = route_price(constructions, route)

    lay_route(Growth(constructions, NO_TRADE), route)

    assert constructions.track_at((6, 6)) == (Track((6, 6), (6, 10)),)
    assert len(dead_ends(constructions)) == 2
    assert price == pytest.approx(300 * (len(route.pieces) - 1))


def test_track_ending_on_a_line_joins_it_so_trains_run_onto_it() -> None:
    constructions = Constructions(make_map())
    growth = Growth(constructions, NO_TRADE)
    for station in [(6, 2), (6, 14), (1, 10)]:
        build_station(growth, station)
    line = laid(constructions, (6, 2), (6, 14))

    laid(constructions, (1, 10), line.cells[5])

    assert dead_ends(constructions) == []
    assert Railway(constructions).route((1, 10), (6, 2)) is not None


def test_track_crossing_a_line_crosses_it_in_one_cell() -> None:
    constructions = Constructions(make_map())
    line = laid(constructions, (6, 2), (6, 10))

    laid(constructions, (2, 6), (10, 6))

    crossings = [cell for cell in line.cells if len(constructions.track_at(cell)) == 2]
    assert len(crossings) == 1
    first, second = constructions.track_at(crossings[0])
    assert not set(first.ends) & set(second.ends)
    assert len(dead_ends(constructions)) == 4


@pytest.mark.parametrize(
    ("width", "electric", "per_cell"),
    [
        (TrackWidth.SINGLE, False, 300),
        (TrackWidth.DOUBLE, False, 540),
        (TrackWidth.SINGLE, True, 750),
        (TrackWidth.DOUBLE, True, 1350),
    ],
)
def test_double_and_electric_track_cost_more_a_cell(
    width: TrackWidth, electric: bool, per_cell: float
) -> None:
    constructions = Constructions(make_map())

    route = plan_route(constructions, (6, 3), (6, 8), width, electric)

    assert route is not None
    assert route_price(constructions, route) == pytest.approx(per_cell * len(route.pieces))
    assert all(piece.width == width and piece.electric == electric for piece in route.pieces)


def test_electric_track_is_electrified_once_laid() -> None:
    constructions = Constructions(make_map())

    route = laid(constructions, (6, 3), (6, 8), electric=True)

    assert all(constructions.electrified(cell) for cell in route.cells)
    assert all(piece.electric for piece in track(constructions))


@pytest.mark.parametrize(
    ("width", "electric", "per_cell"),
    [
        (TrackWidth.DOUBLE, False, 240),
        (TrackWidth.SINGLE, True, 450),
        (TrackWidth.DOUBLE, True, 1050),
    ],
)
def test_laying_along_a_line_widens_or_electrifies_it_without_new_track(
    width: TrackWidth, electric: bool, per_cell: float
) -> None:
    constructions = Constructions(make_map())
    line = laid(constructions, (6, 2), (6, 12))
    route = plan_route(constructions, (6, 12), (6, 2), width, electric)
    assert route is not None
    price = route_price(constructions, route)

    lay_route(Growth(constructions, NO_TRADE), route)

    pieces = track(constructions)
    assert len(pieces) == len(line.pieces)
    assert all(piece.width == width and piece.electric == electric for piece in pieces)
    assert price == pytest.approx(per_cell * len(pieces))


def test_a_station_built_on_a_line_takes_the_place_of_its_track_and_keeps_it_joined() -> None:
    constructions = Constructions(make_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, (6, 2))
    build_station(growth, (6, 14))
    route = plan_route(constructions, (6, 2), (6, 14))
    assert route is not None
    lay_route(growth, route)
    middle = route.cells[len(route.cells) // 2]

    changed = build_station(growth, middle)

    assert changed is not None
    assert constructions.track_at(middle) == ()
    assert Station(middle) in constructions.at(middle)
    assert Railway(constructions).route((6, 2), (6, 14)) is not None


def test_no_second_station_goes_up_where_one_stands() -> None:
    constructions = Constructions(make_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, (6, 6))

    again = build_station(growth, (6, 6))

    assert again is None
