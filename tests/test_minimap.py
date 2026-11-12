from collections.abc import Iterator

import numpy as np
import pytest
from imgui_bundle import imgui

from builders import flat_map
from railroads.mapgen.world import WorldLand
from railroads.sim.geo import Bounds
from railroads.sim.tilemap import TileMap
from railroads.ui.iso_camera import IsoCamera
from railroads.ui.minimap import (
    MIN_BOX,
    OVERVIEW_SIDE,
    WORLD_NORTH,
    WORLD_SOUTH,
    Minimap,
    MinimapLayout,
    view_corners,
    world_rgba,
)

SCREEN = (1280.0, 800.0)


def make_tile_map() -> TileMap:
    return flat_map(20, 30)


@pytest.fixture
def draw_list() -> Iterator[imgui.ImDrawList]:
    context = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(*SCREEN)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    imgui.new_frame()
    yield imgui.get_background_draw_list()
    imgui.render()
    imgui.destroy_context(context)


def test_both_maps_sit_side_by_side_in_the_corner() -> None:
    layout = MinimapLayout.of((30.0, 20.0), SCREEN)

    world, overview = layout.world, layout.overview

    assert overview.right < SCREEN[0] and overview.bottom < SCREEN[1]
    assert world.right < overview.left
    assert world.bottom == pytest.approx(overview.bottom)
    assert overview.width / overview.height == pytest.approx(30.0 / 20.0)


def test_overview_turns_map_points_into_spots_and_back() -> None:
    layout = MinimapLayout.of((30.0, 20.0), SCREEN)

    spot = layout.overview_point(12.0, 5.0)
    back = layout.overview_to_map(*spot)

    assert layout.overview.contains(*spot)
    assert back == pytest.approx((12.0, 5.0))


def test_clicks_beyond_the_overview_land_on_the_map_edge() -> None:
    layout = MinimapLayout.of((30.0, 20.0), SCREEN)

    point = layout.overview_to_map(layout.overview.right + 50, layout.overview.top - 50)

    assert point == pytest.approx((30.0, 0.0))


def test_world_map_spans_the_shown_latitudes() -> None:
    layout = MinimapLayout.of((30.0, 20.0), SCREEN)

    top_left = layout.world_point(-180.0, WORLD_NORTH)
    bottom_right = layout.world_point(180.0, WORLD_SOUTH)
    poland = layout.world_point(19.0, 52.0)

    assert top_left == pytest.approx((layout.world.left, layout.world.top))
    assert bottom_right == pytest.approx((layout.world.right, layout.world.bottom))
    assert layout.world.contains(*poland)
    assert poland[0] > layout.world.left + layout.world.width / 2


def test_view_corners_go_round_the_screen() -> None:
    camera = IsoCamera(center_x=15.0, center_y=10.0, angle_deg=0.0, cell_units=1.0)

    corners = view_corners(camera, (640.0, 400.0), (0.0, 0.0), SCREEN)

    xs, ys = zip(*corners, strict=True)
    assert np.mean(xs) == pytest.approx(15.0)
    assert np.mean(ys) == pytest.approx(10.0)
    assert xs[0] < xs[1] and ys[1] < ys[2]


def test_world_picture_is_sea_blue_and_land_green() -> None:
    world = WorldLand(np.array([[0, 255]], dtype=np.uint8))

    rgba = world_rgba(world)

    sea, land = rgba[0, 0], rgba[0, 1]
    assert rgba.shape == (1, 2, 4)
    assert sea[2] > sea[1] and land[1] > land[2]
    assert (rgba[..., 3] == 255).all()


def test_minimap_draws_both_maps_with_the_view_and_towns(draw_list: imgui.ImDrawList) -> None:
    tile_map = make_tile_map()
    minimap = Minimap(tile_map, WorldLand(np.zeros((4, 8), dtype=np.uint8)), [(3, 4), (10, 12)])
    without_towns = Minimap(tile_map, minimap.world)
    layout = MinimapLayout.of(tile_map.grid.size(), SCREEN)
    textures = (imgui.ImTextureRef(1), imgui.ImTextureRef(2))
    corners = [(5.0, 5.0), (20.0, 5.0), (20.0, 15.0), (5.0, 15.0)]

    without_towns.draw(draw_list, layout, textures, corners)
    plain = draw_list.vtx_buffer.size()
    minimap.draw(draw_list, layout, textures, corners)

    assert plain > 0
    assert draw_list.vtx_buffer.size() - plain > plain


def test_a_small_map_gets_a_box_big_enough_to_see() -> None:
    tile_map = make_tile_map()
    layout = MinimapLayout.of(tile_map.grid.size(), SCREEN)

    left, top, right, bottom = layout.map_box(tile_map.bounds)

    centre = layout.world_point(*tile_map.bounds.center)
    assert right - left == pytest.approx(MIN_BOX)
    assert bottom - top == pytest.approx(MIN_BOX)
    assert ((left + right) / 2, (top + bottom) / 2) == pytest.approx(centre)


def test_a_big_map_keeps_its_true_size_on_the_world() -> None:
    bounds = Bounds(west=0.0, east=60.0, south=0.0, north=30.0)
    layout = MinimapLayout.of((30.0, 20.0), SCREEN)

    left, top, right, bottom = layout.map_box(bounds)

    assert right - left == pytest.approx(layout.world.width / 6)
    assert bottom - top == pytest.approx(layout.world.height * 30 / (WORLD_NORTH - WORLD_SOUTH))


def test_the_world_map_alone_takes_the_corner() -> None:
    layout = MinimapLayout.of((30.0, 20.0), SCREEN, overview=False, world=True)

    world = layout.world

    assert layout.overview is None
    assert world is not None
    assert world.right < SCREEN[0] and world.bottom < SCREEN[1]
    assert layout.contains(world.left + 1, world.top + 1)


def test_a_zoomed_overview_shows_the_part_of_the_map_around_the_view() -> None:
    layout = MinimapLayout.of((40.0, 20.0), SCREEN, zoom=4, focus=(20.0, 10.0))

    middle = layout.overview_to_map(
        layout.overview.left + layout.overview.width / 2,
        layout.overview.top + layout.overview.height / 2,
    )

    assert layout.window == pytest.approx((15.0, 7.5, 25.0, 12.5))
    assert middle == pytest.approx((20.0, 10.0))
    assert layout.overview_to_map(*layout.overview_point(17.0, 9.0)) == pytest.approx((17.0, 9.0))


def test_a_zoomed_overview_stays_on_the_map_near_its_edge() -> None:
    layout = MinimapLayout.of((40.0, 20.0), SCREEN, zoom=2, focus=(1.0, 19.0))

    window = layout.window

    assert window == pytest.approx((0.0, 10.0, 20.0, 20.0))


def test_the_overview_zooms_in_and_out_within_its_steps() -> None:
    minimap = Minimap(make_tile_map(), WorldLand(np.zeros((4, 8), dtype=np.uint8)))

    minimap.zoom_by(2)
    closest = minimap.zoom
    minimap.zoom_by(10)
    minimap.zoom_by(-1)

    assert closest == 4
    assert minimap.zoom == 4


def test_the_overview_of_a_long_thin_map_keeps_its_shape_within_the_same_side() -> None:
    layout = MinimapLayout.of((80.0, 250.0), SCREEN)

    overview = layout.overview

    assert overview is not None
    assert overview.height == pytest.approx(OVERVIEW_SIDE)
    assert overview.width / overview.height == pytest.approx(80.0 / 250.0)
