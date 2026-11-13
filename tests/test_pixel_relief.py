import numpy as np
import numpy.typing as npt
import pytest

from builders import flat_map
from railroads.sim.grids import Grid, HexGrid, SquareGrid
from railroads.sim.tilemap import Relief, TileMap
from railroads.ui.pixel.lattice import Lattice
from railroads.ui.pixel.projection import PixelProjection
from railroads.ui.pixel.relief import Blend, Elevation, raise_picture, raycast, sample

LOOKS = {HexGrid: (0.0, 1.0), SquareGrid: (-45.0, float(np.sqrt(2)))}


def sample_point_by_point(raster: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    fx, fy = np.clip(x - 0.5, 0, raster.shape[1] - 1), np.clip(y - 0.5, 0, raster.shape[0] - 1)
    x0 = np.minimum(fx.astype(np.int64), raster.shape[1] - 2)
    y0 = np.minimum(fy.astype(np.int64), raster.shape[0] - 2)
    tx, ty = fx - x0, fy - y0
    top = raster[y0, x0] * (1 - tx) + raster[y0, x0 + 1] * tx
    bottom = raster[y0 + 1, x0] * (1 - tx) + raster[y0 + 1, x0 + 1] * tx
    return top * (1 - ty) + bottom * ty


def raycast_from_the_top(lift: np.ndarray, rows: int, cols: int) -> npt.NDArray[np.int64]:
    offset = np.zeros((rows, cols), dtype=np.int64)
    found = np.zeros((rows, cols), dtype=bool)
    for d in range(int(lift.max()), -1, -1):
        reach = lift[d : d + rows, :cols] >= d
        offset[reach & ~found] = d
        found |= reach
    return offset


def mountain_map(grid_type: type[Grid] = HexGrid) -> TileMap:
    grid = grid_type(rows=30, cols=24, km_per_cell=2.0)
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    relief[:, 14:18] = Relief.HILLS
    relief[:, 18:] = Relief.MOUNTAINS
    heights = np.broadcast_to(np.arange(grid.cols) * 90.0, grid.shape).astype(np.int16)
    return flat_map(grid=grid, height_m=heights, relief=relief)


def elevation_of(tile_map: TileMap) -> Elevation:
    grid = tile_map.grid
    return Elevation(tile_map, PixelProjection.of(grid, *LOOKS[type(grid)]))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_blend_reads_rasters_as_sampling_each_does(dtype: type) -> None:
    rng = np.random.default_rng(1)
    rasters = [rng.random((40, 30)).astype(dtype) for _ in range(3)]
    x, y = rng.uniform(-3, 33, (50, 60)), rng.uniform(-3, 43, (50, 60))

    blend = Blend.at((40, 30), x, y)

    for raster in rasters:
        assert np.array_equal(blend.read(raster), sample_point_by_point(raster, x, y))
        assert np.array_equal(sample(raster, x, y), sample_point_by_point(raster, x, y))


def test_a_blend_of_chosen_positions_reads_what_the_whole_blend_does_there() -> None:
    rng = np.random.default_rng(2)
    raster = rng.random((20, 20)).astype(np.float32)
    x, y = rng.uniform(0, 20, 500), rng.uniform(0, 20, 500)
    chosen = rng.random(500) < 0.3

    part = Blend.at(raster.shape, x, y).where(chosen)

    assert np.array_equal(part.read(raster), sample_point_by_point(raster, x, y)[chosen])


def test_a_blend_refuses_a_raster_of_another_shape() -> None:
    blend = Blend.at((10, 10), np.zeros(3), np.zeros(3))

    with pytest.raises(ValueError):
        blend.read(np.zeros((10, 11)))


@pytest.mark.parametrize("highest", [0.0, 3.5, 64.0, 128.0])
def test_raycast_finds_the_furthest_point_reaching_each_pixel(highest: float) -> None:
    rng = np.random.default_rng(3)
    lift = (rng.random((60 + int(highest) + 1, 40)) * highest).astype(np.float32)
    lift[5:9, 3:7] = highest

    offset = raycast(lift, 60, 40)

    assert offset.dtype == np.int64
    assert np.array_equal(offset, raycast_from_the_top(lift, 60, 40))


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("below", [6, 2], ids=["rows enough", "rows short"])
def test_a_raised_picture_takes_each_pixel_from_its_offset_below(k: int, below: int) -> None:
    rng = np.random.default_rng(4)
    offset = rng.integers(0, 6, (10, 12))
    picture = rng.random(((10 + below) * k, 12 * k, 4)).astype(np.float32)

    raised = raise_picture(picture, offset, k)

    fine = np.repeat(np.repeat(offset, k, axis=0), k, axis=1) * k
    shown = np.minimum(np.arange(10 * k)[:, None] + fine, picture.shape[0] - 1)
    assert np.array_equal(raised, picture[shown, np.arange(12 * k)[None, :]])


@pytest.mark.parametrize("grid_type", [HexGrid, SquareGrid], ids=["hex", "square"])
def test_the_bound_on_rising_ground_holds_at_every_pixel(grid_type: type[Grid]) -> None:
    elevation = elevation_of(mountain_map(grid_type))
    width, rows = 96, 80

    for left, top in ((0, 0), (300, 40), (500, 100), (700, 150)):
        py, px = np.mgrid[0:rows, 0:width]
        x, y = elevation.projection.to_plane(left + px + 0.5, top + py + 0.5)
        mountains, hills = elevation.shares(x, y)
        most_mountains, most_hills = elevation.most(left, top, width, rows)
        assert mountains.max() <= max(most_mountains, 0.0)
        assert hills.max() <= max(most_hills, 0.0)
        assert most_mountains > 0 or not mountains.any()


def test_ground_far_from_hills_and_mountains_may_not_rise() -> None:
    elevation = elevation_of(mountain_map())
    flat = elevation.projection.to_pixels(3.0, 15.0)
    raised = elevation.projection.to_pixels(21.0, 15.0)

    left, top = int(flat[0]) - 20, int(flat[1]) - 20
    high_left, high_top = int(raised[0]) - 20, int(raised[1]) - 20

    assert not elevation.may_rise(left, top, 40, 40)
    assert elevation.area(left, top, 40, 40) is None
    assert elevation.may_rise(high_left, high_top, 40, 40)
    assert elevation.area(high_left, high_top, 40, 40) is not None


@pytest.mark.parametrize("k", [1, 2])
def test_raised_ground_from_points_of_a_larger_block_rises_as_its_own_points_do(k: int) -> None:
    elevation = elevation_of(mountain_map())
    corner = elevation.projection.to_pixels(19.0, 12.0)
    left, top = int(corner[0]), int(corner[1])
    points = Lattice(elevation.projection, left - 4, top - 4, 50, 60, k)

    raised = elevation.area(left, top, 40, 48, k, points)

    py, px = np.mgrid[0 : 48 * k, 0 : 40 * k]
    x, y = elevation.projection.to_plane(left + (px + 0.5) / k, top + (py + 0.5) / k)
    rise, mountains, hills = elevation.rise(x, y)
    assert raised is not None
    assert np.array_equal(raised.rise, rise)
    assert np.array_equal(raised.mountains, mountains)
    assert np.array_equal(raised.hills, hills)
    assert np.array_equal(raised.lift, elevation.to_lift(rise) * k)


def test_the_lift_of_a_single_point_is_its_lift_among_many() -> None:
    elevation = elevation_of(mountain_map())
    rng = np.random.default_rng(6)
    x, y = rng.uniform(15, 24, 200), rng.uniform(0, 20, 200)

    lifts = elevation.lift(x, y)

    singly = [elevation.lift_at(float(a), float(b)) for a, b in zip(x, y, strict=True)]
    assert singly == lifts.tolist()
