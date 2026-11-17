import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import Constructions, Track, TrackWidth
from railroads.sim.grids import HexGrid
from railroads.sim.tilemap import Relief, TileMap
from railroads.sim.track_rules import TrackBed
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH
from railroads.ui.pixel import track_art
from railroads.ui.pixel.art import draw_sheet
from railroads.ui.pixel.projection import PixelProjection
from railroads.ui.pixel.relief import Elevation
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.track_art import PORTAL_HEIGHT
from railroads.ui.pixel.world import PixelWorld

WEST, EAST = 6, 0
ROW = 8


@pytest.fixture(scope="module")
def sheet() -> SpriteSheet:
    image, places = draw_sheet()
    return SpriteSheet.from_image(image, places)


def foothills() -> TileMap:
    """Hills rising to mountains from the eighth column east."""
    grid = HexGrid(rows=16, cols=14, km_per_cell=2.0)
    relief = np.full(grid.shape, Relief.HILLS, dtype=np.uint8)
    relief[:, 7:] = Relief.MOUNTAINS
    return flat_map(grid=grid, relief=relief)


def line_along(tile_map: TileMap, start: int, end: int) -> np.ndarray:
    (x0, y0), (x1, y1) = (tile_map.grid.center(ROW, col) for col in (start, end))
    t = np.linspace(0, 1, 60)
    return np.array([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t])


def bumpiness(lifts: np.ndarray) -> float:
    return float(np.abs(np.diff(lifts, 2)).sum())


def world_with(sheet: SpriteSheet, *pieces: Track) -> PixelWorld:
    tile_map = foothills()
    constructions = Constructions(tile_map)
    for piece in pieces:
        constructions.build(piece)
    return PixelWorld(tile_map, constructions, sheet, 0.0, 1.0)


def drawn_pixels(world: PixelWorld, cell: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    """Where track is drawn round a cell, as rows and columns of art pixels, and where the
    cell's middle lies among them."""
    px, py = world.projection.to_pixels(*world.tile_map.grid.center(*cell))
    left, top = int(px) - 64, int(py) - 128
    track, _ = world.track.lifted(world.track.constructions.track(), left, top, 128, 192)
    rows, cols = np.nonzero(track[..., 3] > 0.5)
    return np.array([rows, cols]), int(px) - left, int(py) - top


def test_track_through_hills_climbs_a_bed_smoother_than_the_ground_round_it() -> None:
    tile_map = foothills()
    elevation = Elevation(tile_map, PixelProjection.of(tile_map.grid, 0.0, 1.0))
    line = line_along(tile_map, 1, 6)
    ground = elevation.lift(*line)

    elevation.grade([line])

    assert bumpiness(elevation.lift(*line)) < bumpiness(ground) / 2


def test_grading_a_bed_leaves_the_ground_away_from_the_track_as_it_was() -> None:
    tile_map = foothills()
    elevation = Elevation(tile_map, PixelProjection.of(tile_map.grid, 0.0, 1.0))
    away = tile_map.grid.center(ROW - 4, 4)
    before = elevation.lift(*away)

    elevation.grade([line_along(tile_map, 1, 6)])

    assert elevation.lift(*away) == pytest.approx(before)


def test_laying_track_through_hills_grades_the_ground_under_it(sheet: SpriteSheet) -> None:
    world = world_with(sheet)
    pieces = [Track((ROW, col), (WEST, EAST)) for col in range(1, 7)]
    line = line_along(world.tile_map, 1, 6)
    ground = world.elevation.lift(*line)

    for piece in pieces:
        world.constructions.build(piece)
    world.buildings_changed([piece.cell for piece in pieces])

    assert bumpiness(world.elevation.lift(*line)) < bumpiness(ground) / 2


def test_a_tunnel_shows_only_a_portal_where_it_comes_out_of_the_mountains(
    sheet: SpriteSheet,
) -> None:
    cell = (ROW, 7)
    world = world_with(sheet, Track(cell, (WEST, EAST), bed=TrackBed.TUNNEL))
    portal = PORTAL_HEIGHT * PIXEL_CELL_WIDTH

    (rows, cols), x, y = drawn_pixels(world, cell)

    assert len(cols) > 0
    assert cols.max() < x - PIXEL_CELL_WIDTH / 4
    assert (rows > y - world.elevation.lift(*world.tile_map.grid.center(*cell)) - 2 * portal).all()


def test_a_tunnel_deep_in_the_mountains_shows_nothing(sheet: SpriteSheet) -> None:
    cell = (ROW, 10)
    world = world_with(sheet, Track(cell, (WEST, EAST), bed=TrackBed.TUNNEL))

    (rows, _), _, _ = drawn_pixels(world, cell)

    assert len(rows) == 0


def test_trains_go_out_of_sight_only_inside_a_tunnel(sheet: SpriteSheet) -> None:
    world = world_with(
        sheet,
        Track((ROW, 6), (WEST, EAST), TrackWidth.DOUBLE),
        Track((ROW, 7), (WEST, EAST), TrackWidth.DOUBLE, TrackBed.TUNNEL),
    )
    grid = world.tile_map.grid

    in_the_open = world.underground(*grid.center(ROW, 6))
    in_the_tunnel = world.underground(*grid.center(ROW, 7))
    in_untouched_mountains = world.underground(*grid.center(ROW + 3, 7))

    assert not in_the_open
    assert in_the_tunnel
    assert not in_untouched_mountains


def test_many_points_are_found_in_or_out_of_tunnels_at_once_as_one_by_one(
    sheet: SpriteSheet,
) -> None:
    world = world_with(
        sheet,
        Track((ROW, 6), (WEST, EAST), TrackWidth.DOUBLE),
        Track((ROW, 7), (WEST, EAST), TrackWidth.DOUBLE, TrackBed.TUNNEL),
    )
    grid = world.tile_map.grid
    points = [grid.center(row, col) for row in (ROW, ROW + 3) for col in range(4, 10)]
    xs, ys = [float(x) for x, _ in points], [float(y) for _, y in points]

    together = world.underground_all(xs, ys)

    assert together == [world.underground(x, y) for x, y in zip(xs, ys, strict=True)]
    assert any(together)


def test_track_lifted_up_the_picture_keeps_each_side_where_it_lay_flat() -> None:
    down_the_picture = [(10.0, 0.0), (10.0, 10.0), (10.0, 20.0), (10.0, 30.0)]
    climbing_faster = [0.0, 25.0, 50.0, 75.0]

    left = track_art._raised(track_art._offset(down_the_picture, 4.0), climbing_faster)
    right = track_art._raised(track_art._offset(down_the_picture, -4.0), climbing_faster)

    assert all(x == pytest.approx(6.0) for x, _ in left)
    assert all(x == pytest.approx(14.0) for x, _ in right)
    assert [y for _, y in left] == pytest.approx([0.0, -15.0, -30.0, -45.0])
