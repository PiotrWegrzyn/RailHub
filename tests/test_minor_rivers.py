from pathlib import Path

import numpy as np
import pytest

from builders import MemoryFiles, flat_map, layer
from railroads.mapgen.area import MapArea
from railroads.mapgen.streams import (
    BIG_STREAM_KM2,
    StreamLines,
    StreamsFromLines,
    read_streams,
    write_streams,
)
from railroads.mapgen.world_files import World, WorldIndex, square_of, streams_file
from railroads.sim.constructions import CELL_SPACE, Building, BuildingKind, Constructions, Track
from railroads.sim.finance import track_price
from railroads.sim.grids import HexGrid
from railroads.sim.ladders import FARMS, TOWN_LAND
from railroads.sim.surroundings import Land
from railroads.sim.tilemap import Relief, Terrain, TileMap
from railroads.sim.track_rules import RELIEF_RULES, STREAM_FACTOR, track_rules_for
from railroads.ui.ground import cell_colors
from railroads.ui.pixel.streams import NONE_NEAR, STEPS, stream_distance, stream_links

ROWS, COLS = 8, 12


def stream_map(
    cells: list[tuple[int, int]],
    terrain: np.ndarray | None = None,
    height_m: np.ndarray | None = None,
    relief: np.ndarray | None = None,
) -> TileMap:
    stream = layer(ROWS, COLS, 0)
    for cell in cells:
        stream[cell] = 1
    tile_map = flat_map(ROWS, COLS, terrain=terrain, height_m=height_m, relief=relief)
    object.__setattr__(tile_map, "stream", stream)
    return tile_map


def reach(area: MapArea, cells: list[tuple[float, float]], upland_km2: float) -> StreamLines:
    points = np.asarray(cells, dtype=float)
    lon, lat = area.to_lonlat(points[:, 0], points[:, 1])
    return StreamLines(np.column_stack([lon, lat]), np.array([len(points)]), np.array([upland_km2]))


def test_a_minor_river_counts_as_a_river_on_its_own_cell_only() -> None:
    tile_map = stream_map([(4, 5)])
    land = Land(water={Terrain.RIVER: 4.0, Terrain.LAND: 0.5})

    weights = land.weights(tile_map)

    assert weights[4, 5] == pytest.approx(4.0)
    assert weights[4, 6] == pytest.approx(0.5)
    assert weights[1, 1] == pytest.approx(0.5)


def test_the_sea_helps_no_more_than_dry_land_unless_a_ladder_names_it() -> None:
    terrain = layer(ROWS, COLS, Terrain.LAND)
    terrain[:, 0] = Terrain.SEA
    tile_map = flat_map(ROWS, COLS, terrain=terrain)
    fresh_water_lover = Land(water={Terrain.RIVER: 3.0, Terrain.LAND: 0.3})

    weights = fresh_water_lover.weights(tile_map)

    assert weights[3, 1] == pytest.approx(0.3)
    assert weights[3, 6] == pytest.approx(0.3)


@pytest.mark.parametrize("land", [TOWN_LAND, FARMS.land], ids=["towns", "farms"])
def test_towns_and_farms_get_no_help_from_the_sea(land: Land) -> None:
    terrain = layer(ROWS, COLS, Terrain.LAND)
    terrain[:, 0] = Terrain.SEA
    tile_map = flat_map(ROWS, COLS, terrain=terrain)

    weights = land.weights(tile_map)

    assert weights[3, 1] == pytest.approx(weights[3, 6])


def test_a_minor_river_takes_no_room_so_a_farm_fills_its_cell() -> None:
    constructions = Constructions(stream_map([(4, 5)]))

    constructions.build(Building((4, 5), BuildingKind.FARM))

    assert (
        constructions.space_left((4, 5)) == CELL_SPACE - Building((4, 5), BuildingKind.FARM).space
    )


def test_track_across_a_minor_river_costs_twice_as_much_without_a_bridge() -> None:
    tile_map = stream_map([(4, 5)])
    piece = Track((4, 5), (0, 3))

    across = track_price(piece, tile_map.tile((4, 5)))

    assert across == pytest.approx(2 * track_price(piece, tile_map.tile((1, 1))))
    assert track_rules_for(tile_map.tile((4, 5))).can_build


def test_route_planning_shuns_track_across_a_minor_river_as_much_more() -> None:
    relief = layer(ROWS, COLS, Relief.HILLS)
    tile_map = stream_map([(4, 5)], relief=relief)

    rules = track_rules_for(tile_map.tile((4, 5)))

    assert rules.cost_factor == pytest.approx(
        STREAM_FACTOR * RELIEF_RULES[Relief.HILLS].cost_factor
    )
    assert rules.price_factor == pytest.approx(
        STREAM_FACTOR * RELIEF_RULES[Relief.HILLS].price_factor
    )


def test_a_reach_marks_the_land_cells_it_flows_through_but_not_the_water() -> None:
    grid = HexGrid(rows=ROWS, cols=COLS, km_per_cell=2.0)
    area = MapArea.of(flat_map(ROWS, COLS).bounds, grid)
    terrain = layer(ROWS, COLS, Terrain.LAND)
    terrain[:, 9:] = Terrain.SEA
    lines = reach(area, [(1.5, 4.0), (10.5, 4.0)], 2000.0)

    stream = StreamsFromLines(lines, least_km2=1000.0)(area, {"terrain": terrain})

    assert stream[:, :9].sum() >= 7
    assert stream[:, 9:].sum() == 0


def test_reaches_draining_too_little_land_are_left_out() -> None:
    grid = HexGrid(rows=ROWS, cols=COLS, km_per_cell=2.0)
    area = MapArea.of(flat_map(ROWS, COLS).bounds, grid)
    lines = reach(area, [(1.5, 4.0), (10.5, 4.0)], 500.0)

    stream = StreamsFromLines(lines, least_km2=1000.0)(area, {"terrain": layer(ROWS, COLS, 0)})

    assert stream.sum() == 0


def test_a_big_reach_along_a_drawn_river_is_taken_for_that_river() -> None:
    grid = HexGrid(rows=ROWS, cols=COLS, km_per_cell=2.0)
    area = MapArea.of(flat_map(ROWS, COLS).bounds, grid)
    terrain = layer(ROWS, COLS, Terrain.LAND)
    terrain[4, :] = Terrain.RIVER
    lines = reach(area, [(1.5, 3.2), (10.5, 3.2)], BIG_STREAM_KM2 * 2)

    stream = StreamsFromLines(lines, least_km2=1000.0)(area, {"terrain": terrain})

    assert stream.sum() == 0


def test_stream_squares_keep_their_reaches_when_packed() -> None:
    lines = StreamLines(
        np.array([[20.001, 50.002], [20.011, 50.012], [21.5, 49.0], [21.52, 49.03], [21.6, 49.1]]),
        np.array([2, 5]),
        np.array([600.0, 12000.0]),
    )

    unpacked = read_streams(write_streams(lines, 20.0, 50.0))

    assert np.allclose(unpacked.points, lines.points, atol=2e-4)
    assert unpacked.ends.tolist() == [2, 5]
    assert unpacked.upland_km2.tolist() == [600.0, 12000.0]


def test_the_world_reads_the_reaches_of_the_squares_round_a_map() -> None:
    near = StreamLines(np.array([[14.5, 50.5], [14.6, 50.6]]), np.array([2]), np.array([900.0]))
    far = StreamLines(np.array([[100.5, 10.5], [100.6, 10.6]]), np.array([2]), np.array([900.0]))
    squares = {square_of(14.5, 50.5): near, square_of(100.5, 10.5): far}
    files = {
        streams_file(*square): write_streams(lines, *lines.points[0])
        for square, lines in squares.items()
    }
    index = WorldIndex(frozenset(), frozenset(), frozenset(), frozenset(squares))
    world = World(MemoryFiles(files | {"index.npz": index.write()}))

    found = world.streams(flat_map().bounds)

    assert len(found) == 1
    assert found.points[0].tolist() == pytest.approx([14.5, 50.5], abs=2e-4)


def test_a_reach_links_the_cells_it_runs_through_one_to_the_next_and_into_the_sea() -> None:
    grid = HexGrid(rows=ROWS, cols=COLS, km_per_cell=2.0)
    area = MapArea.of(flat_map(ROWS, COLS).bounds, grid)
    terrain = layer(ROWS, COLS, Terrain.LAND)
    terrain[4, 8:] = Terrain.SEA
    row_y = float(grid.center(4, 0)[1])
    lines = reach(area, [(2.6, row_y), (9.6, row_y)], 2000.0)
    east = 1 << STEPS.index((1.0, 0.0))
    west = 1 << STEPS.index((-1.0, 0.0))

    links = StreamsFromLines(lines, least_km2=1000.0)(area, {"terrain": terrain})

    assert links[4, 2] == east
    assert all(links[4, col] == east | west for col in range(3, 8))
    assert links[4, 8:].sum() == 0
    assert links[3].sum() == links[5].sum() == 0


def test_a_piece_of_stream_wiggling_along_a_river_is_taken_for_the_river() -> None:
    grid = HexGrid(rows=ROWS, cols=COLS, km_per_cell=2.0)
    area = MapArea.of(flat_map(ROWS, COLS).bounds, grid)
    terrain = layer(ROWS, COLS, Terrain.LAND)
    terrain[4, :] = Terrain.RIVER
    above = float(grid.center(3, 0)[1])
    lines = reach(area, [(1.6, above), (10.6, above)], 2000.0)

    links = StreamsFromLines(lines, least_km2=1000.0)(area, {"terrain": terrain})

    assert links.sum() == 0


def test_stream_water_lies_along_the_line_between_linked_cells() -> None:
    tile_map = stream_map([(4, 5), (4, 6)])
    links = stream_links(tile_map)
    grid = tile_map.grid
    x, y = grid.center(np.array([4, 4, 1]), np.array([5, 5, 1]))
    x, y = np.asarray(x, dtype=float) + np.array([0.3, 0.3, 0.0]), np.asarray(y, dtype=float)
    y = y + np.array([0.0, 0.4, 0.0])

    gap = stream_distance(links, grid, x, y, np.array([4, 4, 1]), np.array([5, 5, 1]))

    assert gap[0] < 0.1
    assert gap[1] > 0.3
    assert gap[2] == NONE_NEAR


def test_the_minimap_tints_cells_with_a_minor_river_blue() -> None:
    tile_map = stream_map([(4, 5)])

    colours = cell_colors(tile_map)

    assert colours[4, 5, 2] > colours[1, 1, 2]


def test_the_stream_layer_is_kept_when_a_map_is_saved(tmp_path: Path) -> None:
    tile_map = stream_map([(4, 5), (2, 3)])
    path = tmp_path / "map.npz"

    tile_map.save(path)
    loaded = TileMap.load(path)

    assert np.array_equal(loaded.stream, tile_map.stream)
    assert tile_map.window(2, 3, 4, 4).stream[2, 2] == 1
