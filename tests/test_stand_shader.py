import numpy as np
import pytest
from test_pixel_world import make_world

from railroads.ui.pixel import world as world_module
from railroads.ui.pixel.art import draw_sheet
from railroads.ui.pixel.projection import CHUNK
from railroads.ui.pixel.sheet import DENSITIES, SpriteSheet
from railroads.ui.pixel.stand_shader import (
    PAGE,
    StandShader,
    lift_source,
    sheet_for,
    standing,
    tiles,
    track_source,
)


def test_each_scale_stands_sprites_from_the_sheet_drawn_at_it_or_the_finest_that_blows_up_evenly() -> (
    None
):
    scales = [1, 2, 3, 4, 6, 8, 12, 9, 5]

    sheets = [sheet_for(scale) for scale in scales]

    assert sheets == [1, 2, 3, 4, 6, 8, 12, 3, 1]


def test_every_sheet_holds_the_same_sprites_as_big_as_its_resolution_makes_them() -> None:
    plain = SpriteSheet.load()

    sheets = {density: SpriteSheet.load_bytes(density) for density in DENSITIES}

    for density, drawn in sheets.items():
        assert drawn.names == plain.names
        assert np.abs(drawn.sizes - density * plain.sizes).max() <= 3 * density


def test_a_zipped_sprite_unpacks_to_the_pixels_drawn_on_its_sheet() -> None:
    sheet, places = draw_sheet(resolution=3)
    x, y, w, h, _, _ = places["pine_0"]

    zipped = SpriteSheet.load_bytes(3)

    sprite = zipped.pixels[zipped.names.index("pine_0")]
    np.testing.assert_array_equal(sprite, np.asarray(sheet.crop((x, y, x + w, y + h))))


def test_the_sprites_standing_in_a_chunk_are_those_the_cpu_stands_there_in_its_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sheet = SpriteSheet.load()
    world = make_world(sheet)
    chunk = (int(world.scenery.x[0]) // CHUNK, int(world.scenery.y[0]) // CHUNK)
    stood: list[tuple[int, int]] = []
    monkeypatch.setattr(world_module, "_blend", lambda canvas, pixels, x, y: stood.append((x, y)))
    canvas = np.zeros((CHUNK, CHUNK, 4))
    world._stand_scenery(canvas, chunk[0] * CHUNK, chunk[1] * CHUNK, 1, under_track=True)
    world._stand_scenery(canvas, chunk[0] * CHUNK, chunk[1] * CHUNK, 1)

    found = standing(world.scenery, sheet.sizes, sheet.anchors, chunk, 1, 1)

    assert [tuple(corner) for corner in found.corner.tolist()] == stood
    assert len(stood) > 3


def test_sprites_blown_up_cover_as_many_pixels_more_from_the_same_corner_scaled() -> None:
    world = make_world(SpriteSheet.load())
    drawn = world.sheet
    chunk = (int(world.scenery.x[0]) // CHUNK, int(world.scenery.y[0]) // CHUNK)
    plain = standing(world.scenery, drawn.sizes, drawn.anchors, chunk, 1, 1)

    tripled = standing(world.scenery, drawn.sizes, drawn.anchors, chunk, 3, 3)

    np.testing.assert_array_equal(tripled.corner, plain.corner * 3)
    np.testing.assert_array_equal(tripled.size, plain.size * 3)


def test_track_is_drawn_finely_only_in_the_parts_of_a_chunk_it_crosses() -> None:
    span, quality = CHUNK * 12, 2
    step = PAGE // quality

    crossed = tiles(span, quality, (10.0, 10.0, step + 10.0, 20.0))
    every = tiles(span, quality, (0.0, 0.0, float(span), float(span)))

    assert crossed == [(0, 0, step, step), (step, 0, step, step)]
    assert len(every) == 9
    assert sum(width * height for _, _, width, height in every) == span * span


def test_the_track_shaders_have_every_number_of_the_cpu_relief_written_in() -> None:
    sources = [track_source(), lift_source()]

    assert all("@" not in source for source in sources)


def test_without_a_ground_shader_nothing_stands_on_the_graphics_card() -> None:
    stand = StandShader.make(None, SpriteSheet.load_bytes)

    assert stand is None
