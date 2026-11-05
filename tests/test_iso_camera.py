import pytest

from railroads.ui.iso_camera import ZOOM_TILE_WIDTHS, IsoCamera

SCREEN_CENTER = (640.0, 400.0)


def test_camera_center_tile_is_drawn_at_screen_center() -> None:
    camera = IsoCamera(center_x=10, center_y=20, zoom_level=2)

    position = camera.to_screen(10, 20, SCREEN_CENTER)

    assert position == SCREEN_CENTER


def test_next_tile_along_x_is_half_a_diamond_right_and_down() -> None:
    camera = IsoCamera(center_x=0, center_y=0, zoom_level=2)

    position = camera.to_screen(1, 0, SCREEN_CENTER)

    width = ZOOM_TILE_WIDTHS[2]
    assert position == (SCREEN_CENTER[0] + width / 2, SCREEN_CENTER[1] + width / 4)


def test_to_tile_inverts_to_screen() -> None:
    camera = IsoCamera(center_x=50, center_y=30, zoom_level=3)
    screen = camera.to_screen(57.25, 21.5, SCREEN_CENTER)

    tile = camera.to_tile(*screen, SCREEN_CENTER)

    assert tile == pytest.approx((57.25, 21.5))


def test_pan_moves_map_with_the_mouse() -> None:
    camera = IsoCamera(center_x=50, center_y=30, zoom_level=1)
    before = camera.to_screen(60, 40, SCREEN_CENTER)

    camera.pan(25, -10)

    after = camera.to_screen(60, 40, SCREEN_CENTER)
    assert after == pytest.approx((before[0] + 25, before[1] - 10))


def test_zoom_keeps_tile_under_anchor_in_place() -> None:
    camera = IsoCamera(center_x=50, center_y=30, zoom_level=1)
    anchor = (900.0, 250.0)
    tile_before = camera.to_tile(*anchor, SCREEN_CENTER)

    camera.zoom(2, anchor, SCREEN_CENTER)

    assert camera.zoom_level == 3
    assert camera.to_tile(*anchor, SCREEN_CENTER) == pytest.approx(tile_before)


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_to_tile_inverts_to_screen_at_every_rotation(rotation: int) -> None:
    camera = IsoCamera(center_x=50, center_y=30, zoom_level=3, rotation=rotation)
    screen = camera.to_screen(57.25, 21.5, SCREEN_CENTER)

    tile = camera.to_tile(*screen, SCREEN_CENTER)

    assert tile == pytest.approx((57.25, 21.5))


def test_turning_right_moves_x_axis_to_where_y_axis_was() -> None:
    camera = IsoCamera(center_x=0, center_y=0, zoom_level=2)
    y_axis_before = camera.to_screen(0, 1, SCREEN_CENTER)

    camera.turn(1)

    assert camera.to_screen(1, 0, SCREEN_CENTER) == pytest.approx(y_axis_before)


def test_four_turns_return_to_the_start() -> None:
    camera = IsoCamera(center_x=0, center_y=0, rotation=3)

    camera.turn(1)

    assert camera.rotation == 0


def test_pan_moves_map_with_the_mouse_when_turned() -> None:
    camera = IsoCamera(center_x=50, center_y=30, zoom_level=1, rotation=1)
    before = camera.to_screen(60, 40, SCREEN_CENTER)

    camera.pan(25, -10)

    after = camera.to_screen(60, 40, SCREEN_CENTER)
    assert after == pytest.approx((before[0] + 25, before[1] - 10))


def test_rotation_three_points_north_to_the_top_left() -> None:
    camera = IsoCamera(center_x=10, center_y=10, zoom_level=2, rotation=3)

    north = camera.to_screen(10, 9, SCREEN_CENTER)

    assert north[0] < SCREEN_CENTER[0]
    assert north[1] < SCREEN_CENTER[1]


@pytest.mark.parametrize(("start", "steps", "expected"), [(0, -1, 0), (4, 3, 4), (2, 10, 4)])
def test_zoom_is_limited_to_five_levels(start: int, steps: int, expected: int) -> None:
    camera = IsoCamera(center_x=0, center_y=0, zoom_level=start)

    camera.zoom(steps, SCREEN_CENTER, SCREEN_CENTER)

    assert camera.zoom_level == expected
