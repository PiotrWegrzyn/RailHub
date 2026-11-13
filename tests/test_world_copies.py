import pickle
import subprocess
import sys

import numpy as np
from test_elevation import world_over_mountains

from railroads.sim.constructions import Building, BuildingKind, Track
from railroads.sim.economy import Satisfaction
from railroads.sim.tilemap import Vegetation
from railroads.ui.pixel.copies import ChunkJob, Seen, WorldCopy, update_message, world_message
from railroads.ui.pixel.projection import CHUNK
from railroads.ui.pixel.world import PixelWorld


def loads_imgui(module: str) -> bool:
    script = f"import sys, {module}; print(any(m.startswith('imgui_bundle') for m in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return result.stdout.strip() == "True"


def test_the_pixel_world_paints_without_imgui() -> None:
    loaded = loads_imgui("railroads.ui.pixel.world")

    assert not loaded


def test_a_copy_of_the_world_loads_without_imgui() -> None:
    loaded = loads_imgui("railroads.ui.pixel.copies")

    assert not loaded


def copied(world: PixelWorld, number: int = 1) -> WorldCopy:
    copy = WorldCopy()
    copy.start(number, world_message(world))
    return copy


def chunk_of(world: PixelWorld, cell: tuple[int, int]) -> tuple[int, int]:
    px, py = world.projection.to_pixels(*world.tile_map.grid.center(*cell))
    return int(px // CHUNK), int(py // CHUNK)


def test_a_copy_paints_the_very_pixels_its_world_paints() -> None:
    world, mountains, plains = world_over_mountains()
    jobs = [
        ChunkJob(world, *chunk, resolution, fine, over)
        for chunk in (mountains, plains)
        for resolution, fine, over in ((1, False, False), (2, False, True), (2, True, False))
    ]

    copy = copied(world)

    for job in jobs:
        np.testing.assert_array_equal(copy.paint(1, job.drawing), job())


def test_a_copy_keeps_up_with_buildings_track_and_forests() -> None:
    world, _, plains = world_over_mountains()
    copy = copied(world)
    seen = Seen.now(world)
    world.constructions.build(Building((10, 4), BuildingKind.HOUSE))
    world.constructions.build(Track((10, 3), (0, 6)))
    forest = np.ravel_multi_index(([9, 9, 11], [3, 4, 5]), world.tile_map.grid.shape)
    world.tile_map.set_vegetation(forest, Vegetation.FOREST)
    before = world.render(*plains, 2)
    changed = world.buildings_changed([(10, 4), (10, 3), (9, 3), (9, 4), (11, 5)])

    copy.update(1, update_message(world, seen))

    assert chunk_of(world, (10, 4)) in changed
    for chunk in changed:
        np.testing.assert_array_equal(
            copy.paint(1, (*chunk, 2, False, False)), world.render(*chunk, 2)
        )
    assert not np.array_equal(world.render(*plains, 2), before)


def test_a_copy_keeps_up_with_how_satisfied_cells_are() -> None:
    world, _, plains = world_over_mountains()
    copy = copied(world)
    seen = Seen.now(world)
    levels = np.full(world.tile_map.grid.shape, -1, dtype=np.int8)
    levels[8:13, 2:7] = Satisfaction.THRIVING

    changed = world.satisfaction_changed(levels)
    copy.update(1, update_message(world, seen))

    assert plains in changed
    np.testing.assert_array_equal(
        copy.paint(1, (*plains, 2, False, False)), world.render(*plains, 2)
    )


def test_nothing_is_sent_while_the_world_stays_as_it_was() -> None:
    world, _, _ = world_over_mountains()
    seen = Seen.now(world)

    message = update_message(world, seen)

    assert message is None


def test_a_copy_leaves_out_what_belongs_to_a_world_it_no_longer_has() -> None:
    world, _, plains = world_over_mountains()
    copy = copied(world, number=1)
    seen = Seen.now(world)
    old = copy.world
    world.constructions.build(Building((10, 4), BuildingKind.HOUSE))
    world.buildings_changed([(10, 4)])

    copy.start(2, world_message(world))
    copy.update(1, pickle.dumps({"satisfaction": np.zeros(world.tile_map.grid.shape, np.int8)}))
    painted = copy.paint(1, (*plains, 2, False, False))

    assert copy.world is not old
    assert painted is None
    assert update_message(world, seen) is not None
    np.testing.assert_array_equal(
        copy.paint(2, (*plains, 2, False, False)), world.render(*plains, 2)
    )
