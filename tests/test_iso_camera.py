import math

import pytest

from railroads.ui.iso_camera import ZOOM_CELL_WIDTHS, IsoCamera

SCREEN_CENTER = (640.0, 400.0)
SQUARES = {"turn_steps": 4, "base_angle": 45.0, "cell_units": math.sqrt(2)}
HEXAGONS = {"turn_steps": 6, "base_angle": 0.0, "cell_units": 1.0}
EVERY_ROTATION = [(SQUARES, r) for r in range(4)] + [(HEXAGONS, r) for r in range(6)]


@pytest.mark.parametrize("look", [SQUARES, HEXAGONS], ids=["square", "hex"])
def test_camera_centre_is_drawn_at_the_screen_centre(look: dict) -> None:
    camera = IsoCamera(center_x=10, center_y=20, zoom_level=2, **look)

    position = camera.to_screen(10, 20, SCREEN_CENTER)

    assert position == pytest.approx(SCREEN_CENTER)


def test_next_square_along_x_is_half_a_diamond_right_and_down() -> None:
    camera = IsoCamera(center_x=0, center_y=0, zoom_level=2, **SQUARES)

    position = camera.to_screen(1, 0, SCREEN_CENTER)

    width = ZOOM_CELL_WIDTHS[2]
    assert position == pytest.approx((SCREEN_CENTER[0] + width / 2, SCREEN_CENTER[1] + width / 4))


def test_next_hexagon_along_x_is_one_cell_width_to_the_right() -> None:
    camera = IsoCamera(center_x=0, center_y=0, zoom_level=2, **HEXAGONS)

    position = camera.to_screen(1, 0, SCREEN_CENTER)

    assert position == pytest.approx((SCREEN_CENTER[0] + ZOOM_CELL_WIDTHS[2], SCREEN_CENTER[1]))


@pytest.mark.parametrize(("look", "rotation"), EVERY_ROTATION)
def test_to_plane_inverts_to_screen_at_every_rotation(look: dict, rotation: int) -> None:
    camera = IsoCamera(center_x=50, center_y=30, zoom_level=3, rotation=rotation, **look)
    screen = camera.to_screen(57.25, 21.5, SCREEN_CENTER)

    point = camera.to_plane(*screen, SCREEN_CENTER)

    assert point == pytest.approx((57.25, 21.5))


@pytest.mark.parametrize(("look", "rotation"), [(SQUARES, 0), (SQUARES, 1), (HEXAGONS, 2)])
def test_pan_moves_the_map_with_the_mouse(look: dict, rotation: int) -> None:
    camera = IsoCamera(center_x=50, center_y=30, zoom_level=1, rotation=rotation, **look)
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


def test_turning_squares_right_moves_the_x_axis_to_where_the_y_axis_was() -> None:
    camera = IsoCamera(center_x=0, center_y=0, zoom_level=2, **SQUARES)
    y_axis_before = camera.to_screen(0, 1, SCREEN_CENTER)

    camera.turn(1)

    assert camera.to_screen(1, 0, SCREEN_CENTER) == pytest.approx(y_axis_before)


@pytest.mark.parametrize(("look", "turns"), [(SQUARES, 4), (HEXAGONS, 6)], ids=["square", "hex"])
def test_a_full_circle_of_turns_returns_to_the_start(look: dict, turns: int) -> None:
    camera = IsoCamera(center_x=0, center_y=0, rotation=1, **look)

    for _ in range(turns):
        camera.turn(1)

    assert camera.rotation == 1


def test_squares_at_rotation_three_have_north_at_the_top_left() -> None:
    camera = IsoCamera(center_x=10, center_y=10, zoom_level=2, rotation=3, **SQUARES)

    north = camera.to_screen(10, 9, SCREEN_CENTER)

    assert north[0] < SCREEN_CENTER[0]
    assert north[1] < SCREEN_CENTER[1]


def test_hexagons_at_rotation_zero_have_north_up() -> None:
    camera = IsoCamera(center_x=10, center_y=10, zoom_level=2, **HEXAGONS)

    north = camera.to_screen(10, 9, SCREEN_CENTER)

    assert north[0] == pytest.approx(SCREEN_CENTER[0])
    assert north[1] < SCREEN_CENTER[1]


@pytest.mark.parametrize(("start", "steps", "expected"), [(0, -1, 0), (5, 3, 5), (2, 10, 5)])
def test_zoom_is_limited_to_six_levels(start: int, steps: int, expected: int) -> None:
    camera = IsoCamera(center_x=0, center_y=0, zoom_level=start)

    camera.zoom(steps, SCREEN_CENTER, SCREEN_CENTER)

    assert camera.zoom_level == expected


def test_screen_top_left_is_north_for_squares_at_rotation_three() -> None:
    camera = IsoCamera(center_x=0, center_y=0, rotation=3, **SQUARES)

    x, y = camera.map_direction(-1.0, -1.0)

    assert y < 0
    assert abs(y) > abs(x)


def test_screen_top_left_is_north_west_for_unturned_hexagons() -> None:
    camera = IsoCamera(center_x=0, center_y=0, **HEXAGONS)

    x, y = camera.map_direction(-1.0, -1.0)

    assert x < 0
    assert y < 0
    assert x * x + y * y == pytest.approx(1.0)


def test_zoom_level_four_sits_between_three_and_five() -> None:
    widths = [IsoCamera(center_x=0, center_y=0, zoom_level=level).cell_width for level in (2, 3, 4)]

    assert widths == [16, 24, 32]
