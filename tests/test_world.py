import numpy as np

from railroads.sim.units import KM_PER_TILE
from railroads.sim.world import Terrain, World


def test_generate_creates_terrain_grid_of_requested_size() -> None:
    world = World.generate(width=64, height=32, seed=7)

    assert world.terrain.shape == (32, 64)
    assert world.terrain.dtype == np.uint8


def test_generate_is_deterministic_for_a_seed() -> None:
    first = World.generate(width=64, height=64, seed=7)

    second = World.generate(width=64, height=64, seed=7)

    assert np.array_equal(first.terrain, second.terrain)


def test_generate_differs_between_seeds() -> None:
    first = World.generate(width=64, height=64, seed=7)

    second = World.generate(width=64, height=64, seed=8)

    assert not np.array_equal(first.terrain, second.terrain)


def test_generate_contains_every_terrain_type() -> None:
    world = World.generate(width=64, height=64, seed=7)

    terrain_types = set(np.unique(world.terrain).tolist())

    assert terrain_types == {terrain.value for terrain in Terrain}


def test_terrain_at_reads_column_x_of_row_y() -> None:
    terrain = np.array([[0, 1, 2], [3, 3, 3]], dtype=np.uint8)
    world = World(width=3, height=2, terrain=terrain)

    result = world.terrain_at(x=2, y=0)

    assert result is Terrain.HILLS


def test_world_size_in_km_uses_km_per_tile() -> None:
    world = World.generate(width=100, height=50, seed=7)

    size = (world.width_km, world.height_km)

    assert size == (100 * KM_PER_TILE, 50 * KM_PER_TILE)


def test_tile_at_km_maps_position_to_tile() -> None:
    world = World.generate(width=10, height=10, seed=7)

    tile = world.tile_at_km(3.5 * KM_PER_TILE, 0.2 * KM_PER_TILE)

    assert tile == (3, 0)


def test_tile_at_km_returns_none_outside_the_map() -> None:
    world = World.generate(width=10, height=10, seed=7)

    tile = world.tile_at_km(-0.1, 10 * KM_PER_TILE)

    assert tile is None
