import pytest

from builders import flat_map
from railroads.ui.map_view import LOOKS
from railroads.ui.pixel import ground_shader
from railroads.ui.pixel.ground_shader import GroundShader, view_of
from railroads.ui.pixel.projection import PixelProjection


def test_the_view_measures_the_map_in_device_pixels_from_its_corner() -> None:
    origin, top_left, size = (130, 70), (10.0, 40.0), (800.0, 600.0)

    view = view_of(origin, 4, top_left, size, 2, (0.6, 0.8))

    assert (view.width, view.height) == (1600, 1200)
    assert view.origin == (240.0, 60.0)
    assert view.scale == 8


def test_without_opengl_there_is_no_shader_and_the_map_keeps_its_ground_picture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tile_map = flat_map()
    look = LOOKS[tile_map.grid.kind]
    projection = PixelProjection.of(tile_map.grid, look.angle_deg, look.cell_units)
    monkeypatch.setattr(ground_shader, "GL", None)

    shader = GroundShader.make(projection, tile_map)

    assert shader is None
