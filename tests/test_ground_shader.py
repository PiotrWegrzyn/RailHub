import numpy as np
import pytest

from builders import flat_map
from railroads.ui.map_view import LOOKS
from railroads.ui.pixel import ground_shader
from railroads.ui.pixel.ground_shader import ChunkTexture, GroundShader, screen_of, view_of
from railroads.ui.pixel.projection import PixelProjection


def test_the_view_holds_every_art_pixel_showing_in_the_map_area() -> None:
    origin, top_left, size = (130, 70), (10.0, 40.0), (800.0, 600.0)

    view = view_of(origin, 4, top_left, size)

    assert (view.left, view.top, view.width, view.height) == (-30, -8, 200, 151)


def test_the_screen_measures_the_map_in_device_pixels_from_its_corner() -> None:
    origin, top_left, size = (130, 70), (10.0, 40.0), (800.0, 600.0)

    screen = screen_of(origin, 4, top_left, size, 2)

    assert (screen.width, screen.height) == (1600, 1200)
    assert screen.origin == (240.0, 60.0)
    assert screen.scale == 8


def test_without_opengl_there_is_no_shader_and_the_map_keeps_its_ground_picture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tile_map = flat_map()
    look = LOOKS[tile_map.grid.kind]
    projection = PixelProjection.of(tile_map.grid, look.angle_deg, look.cell_units)
    monkeypatch.setattr(ground_shader, "GL", None)

    shader = GroundShader.make(projection, tile_map, np.zeros((4, 4)))

    assert shader is None


def test_the_shader_has_every_table_and_number_of_the_cpu_ground_written_in() -> None:
    source = ground_shader.fragment_source()

    assert "@" not in source
    assert source.count("vec3(") > 52


def test_each_cell_hands_the_shader_its_biome_deposit_hills_and_glow() -> None:
    biomes = np.array([[7, 0]])
    deposits = np.array([[8, 0]])
    hills = np.array([[True, False]])

    info = ground_shader.cell_info(biomes, deposits, hills, np.array([[0.24, 0.0]]))

    assert info.tolist() == [[[7, 8, 1, 60], [0, 0, 0, 0]]]


class DeletingGL:
    def __init__(self) -> None:
        self.deleted: list[int] = []

    def glDeleteTextures(self, count: int, textures: list[int]) -> None:
        self.deleted += textures[:count]


def test_a_blown_up_chunk_texture_is_deleted_on_a_frame_after_nothing_holds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gl = DeletingGL()
    monkeypatch.setattr(ground_shader, "GL", gl)
    monkeypatch.setattr(ground_shader, "_UNWANTED", [])
    kept, dropped = ChunkTexture(7, 1024, 1024), ChunkTexture(8, 1024, 1024)
    del dropped

    ground_shader.free_textures()
    ground_shader.free_textures()

    assert gl.deleted == [8]
    assert kept.texture_id() == 7
