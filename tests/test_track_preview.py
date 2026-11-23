import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image
from screens import world_over_mountains

from builders import flat_map
from railroads.sim.constructions import Constructions, Track, TrackWidth
from railroads.sim.routes import Route, plan_route
from railroads.ui.iso_camera import IsoCamera
from railroads.ui.map_view import LOOKS
from railroads.ui.pixel import track_preview
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.track_preview import TrackPreview
from railroads.ui.pixel.world import PixelWorld


def test_a_route_too_long_to_draw_shows_no_preview_instead_of_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, _, _ = world_over_mountains()
    preview = TrackPreview(world)

    def too_big(*args: object) -> None:
        raise Image.DecompressionBombError("too many pixels")

    monkeypatch.setattr(world, "planned_track", too_big)

    preview._make([Track((5, 5), (0, 6))], 4)

    assert preview._texture is None


def test_a_route_just_planned_is_drawn_on_the_frame_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, _, _ = world_over_mountains()
    preview = TrackPreview(world)
    made: list[int] = []
    monkeypatch.setattr(preview, "_make", lambda pieces, scale: made.append(len(pieces)))
    monkeypatch.setattr(track_preview.hello_imgui, "request_refresh", lambda: None)
    route = Route([(5, 5)], [Track((5, 5), (0, 6))], 0.0)
    camera = IsoCamera(center_x=5.0, center_y=5.0, angle_deg=0.0, cell_units=1.0)
    camera.zoom_level = 4
    context = imgui.create_context()
    imgui.get_io().display_size = imgui.ImVec2(1280, 800)
    imgui.get_io().backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    imgui.new_frame()
    try:
        draw_list = imgui.get_background_draw_list()
        preview.draw(draw_list, route, camera, (640.0, 400.0))
        first = list(made)
        preview.draw(draw_list, route, camera, (640.0, 400.0))
        second = list(made)
    finally:
        imgui.render()
        imgui.destroy_context(context)

    assert first == []
    assert second == [1]


def _world_with(constructions: Constructions) -> PixelWorld:
    look = LOOKS[constructions.tile_map.grid.kind]
    return PixelWorld(
        constructions.tile_map,
        constructions,
        SpriteSheet.load(),
        look.angle_deg,
        look.cell_units,
    )


def test_double_track_planned_over_single_track_looks_as_it_will_once_laid() -> None:
    tile_map = flat_map(14, 20)
    with_single, with_double = Constructions(tile_map), Constructions(tile_map)
    for piece in plan_route(with_single, (2, 3), (10, 12)).pieces:
        with_single.build(piece)
    route = plan_route(with_single, (2, 3), (10, 12), TrackWidth.DOUBLE)
    for piece in route.pieces:
        with_double.build(piece)
    area = (0, 0, 400, 200, 1)

    planned = _world_with(with_single).planned_track(route.pieces, *area)
    laid = _world_with(with_double).planned_track(route.pieces, *area)

    assert planned[..., 3].any()
    assert np.allclose(planned, laid, rtol=1e-7, atol=1e-6)
