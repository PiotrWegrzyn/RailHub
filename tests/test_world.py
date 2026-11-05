import numpy as np

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
