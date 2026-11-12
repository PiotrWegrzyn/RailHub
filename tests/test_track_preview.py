import numpy as np
import pytest
from PIL import Image
from test_elevation import world_over_mountains

from builders import flat_map
from railroads.sim.constructions import Constructions, Track, TrackWidth
from railroads.sim.routes import plan_route
from railroads.ui.map_view import LOOKS
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
    np.testing.assert_allclose(planned, laid, atol=1e-6)
