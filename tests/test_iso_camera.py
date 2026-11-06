import math

import pytest

from railroads.ui.iso_camera import ZOOM_CELL_WIDTHS, IsoCamera

SCREEN_CENTER = (640.0, 400.0)
SQUARES = {"angle_deg": -45.0, "cell_units": math.sqrt(2)}
HEXAGONS = {"angle_deg": 0.0, "cell_units": 1.0}


@pytest.mark.parametrize("look", [SQUARES, HEXAGONS], ids=["square", "hex"])
def test_camera_centre_is_drawn_at_the_screen_centre(look: dict) -> None:
    camera = IsoCamera(center_x=10, center_y=20, zoom_level=2, **look)

    position = camera.to_screen(10, 20, SCREEN_CENTER)

    assert position == pytest.approx(SCREEN_CENTER)


def test_next_square_east_is_half_a_diamond_right_and_up() -> None:
    camera = IsoCamera(center_x=0, center_y=0, zoom_level=2, **SQUARES)

    position = camera.to_screen(1, 0, SCREEN_CENTER)

    width = ZOOM_CELL_WIDTHS[2]
    assert position == pytest.approx((SCREEN_CENTER[0] + width / 2, SCREEN_CENTER[1] - width / 4))


def test_next_hexagon_along_x_is_one_cell_width_to_the_right() -> None:
    camera = IsoCamera(center_x=0, center_y=0, zoom_level=2, **HEXAGONS)

    position = camera.to_screen(1, 0, SCREEN_CENTER)

    assert position == pytest.approx((SCREEN_CENTER[0] + ZOOM_CELL_WIDTHS[2], SCREEN_CENTER[1]))


@pytest.mark.parametrize("look", [SQUARES, HEXAGONS], ids=["square", "hex"])
def test_to_plane_inverts_to_screen(look: dict) -> None:
    camera = IsoCamera(center_x=50, center_y=30, zoom_level=3, **look)
    screen = camera.to_screen(57.25, 21.5, SCREEN_CENTER)

    point = camera.to_plane(*screen, SCREEN_CENTER)

    assert point == pytest.approx((57.25, 21.5))


@pytest.mark.parametrize("look", [SQUARES, HEXAGONS], ids=["square", "hex"])
def test_pan_moves_the_map_with_the_mouse(look: dict) -> None:
    camera = IsoCamera(center_x=50, center_y=30, zoom_level=1, **look)
    before = camera.to_screen(60, 40, SCREEN_CENTER)

    camera.pan(25, -10)

    after = camera.to_screen(60, 40, SCREEN_CENTER)
    assert after == pytest.approx((before[0] + 25, before[1] - 10))


def test_zoom_keeps_the_point_under_the_anchor_in_place() -> None:
    camera = IsoCamera(center_x=50, center_y=30, zoom_level=1, **SQUARES)
    anchor = (900.0, 250.0)
    point_before = camera.to_plane(*anchor, SCREEN_CENTER)

    camera.zoom(2, anchor, SCREEN_CENTER)

    assert camera.zoom_level == 3
    assert camera.to_plane(*anchor, SCREEN_CENTER) == pytest.approx(point_before)


def test_squares_have_north_at_the_top_left() -> None:
    camera = IsoCamera(center_x=10, center_y=10, zoom_level=2, **SQUARES)

    north = camera.to_screen(10, 9, SCREEN_CENTER)

    assert north[0] < SCREEN_CENTER[0]
    assert north[1] < SCREEN_CENTER[1]


def test_hexagons_have_north_up() -> None:
    camera = IsoCamera(center_x=10, center_y=10, zoom_level=2, **HEXAGONS)

    north = camera.to_screen(10, 9, SCREEN_CENTER)

    assert north[0] == pytest.approx(SCREEN_CENTER[0])
    assert north[1] < SCREEN_CENTER[1]


@pytest.mark.parametrize(("start", "steps", "expected"), [(0, -1, 0), (5, 3, 5), (2, 10, 5)])
def test_zoom_is_limited_to_six_levels(start: int, steps: int, expected: int) -> None:
    camera = IsoCamera(center_x=0, center_y=0, zoom_level=start)

    camera.zoom(steps, SCREEN_CENTER, SCREEN_CENTER)

    assert camera.zoom_level == expected


def test_screen_top_left_is_north_for_squares() -> None:
    camera = IsoCamera(center_x=0, center_y=0, **SQUARES)

    x, y = camera.map_direction(-1.0, -1.0)

    assert y < 0
    assert abs(y) > abs(x)


def test_screen_top_left_is_north_west_for_hexagons() -> None:
    camera = IsoCamera(center_x=0, center_y=0, **HEXAGONS)

    x, y = camera.map_direction(-1.0, -1.0)

    assert x < 0
    assert y < 0
    assert x * x + y * y == pytest.approx(1.0)


@pytest.mark.parametrize(("level", "scale"), [(0, 0), (2, 0), (3, 1), (4, 2), (5, 3)])
def test_close_zoom_levels_show_pixel_art_at_whole_scales(level: int, scale: int) -> None:
    camera = IsoCamera(center_x=0, center_y=0, zoom_level=level)

    assert camera.pixel_scale == scale
