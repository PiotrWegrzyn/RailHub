import pytest
from PIL import Image
from test_elevation import world_over_mountains

from railroads.sim.constructions import Track
from railroads.ui.pixel.track_preview import TrackPreview


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
