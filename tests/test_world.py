from pathlib import Path

import numpy as np

from railroads import maps
from railroads.mapgen.sources import Shape
from railroads.mapgen.world import WorldLand


def test_baking_marks_water_shapes_as_sea() -> None:
    atlantic = Shape(
        "polygon", (np.array([[-40.0, 10.0], [-20.0, 10.0], [-20.0, 30.0], [-40.0, 30.0]]),)
    )

    world = WorldLand.bake([atlantic], degrees_per_cell=5.0)

    assert world.land.shape == (36, 72)
    assert world.land_at(-30.0, 20.0) == 0
    assert world.land_at(20.0, 50.0) == 255


def test_saved_world_loads_the_same(tmp_path: Path) -> None:
    world = WorldLand(np.arange(8, dtype=np.uint8).reshape(2, 4))

    world.save(tmp_path / "world.npz")
    loaded = WorldLand.load(tmp_path / "world.npz")

    assert loaded.land.tolist() == world.land.tolist()
    assert loaded.degrees_per_cell == 90.0


def test_shipped_world_has_land_and_sea_in_the_right_places() -> None:
    world = maps.world_land()

    on_land = world.land_at(
        np.array([19.0, 2.3, -100.0, 135.0]), np.array([52.0, 48.9, 40.0, -25.0])
    )
    at_sea = world.land_at(np.array([-30.0, -150.0, 80.0]), np.array([30.0, 0.0, -20.0]))

    assert (on_land > 200).all()
    assert (at_sea < 50).all()
