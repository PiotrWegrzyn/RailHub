import pytest

from railroads.ui.camera_flight import PAN_SECONDS, ZOOM_STEP_SECONDS, Flight
from railroads.ui.iso_camera import IsoCamera


def flown(camera: IsoCamera, flight: Flight, seconds: float) -> bool:
    flying = True
    for _ in range(round(seconds * 60)):
        flying = flight.fly(camera, 1 / 60)
    return flying


def test_the_view_glides_across_at_its_zoom_before_closing_in() -> None:
    camera = IsoCamera(center_x=10.0, center_y=10.0, zoom_level=3)
    flight = Flight(camera, (50.0, 30.0), 5)

    flown(camera, flight, PAN_SECONDS / 2)

    assert camera.zoom_level == 3
    assert camera.center_x == pytest.approx(30.0, abs=1.0)
    assert camera.center_y == pytest.approx(20.0, abs=1.0)


def test_the_flight_ends_on_the_town_at_its_zoom() -> None:
    camera = IsoCamera(center_x=10.0, center_y=10.0, zoom_level=3)
    flight = Flight(camera, (50.0, 30.0), 5)

    flying = flown(camera, flight, PAN_SECONDS + 2 * ZOOM_STEP_SECONDS + 0.1)

    assert not flying
    assert (camera.center_x, camera.center_y, camera.zoom_level) == (50.0, 30.0, 5)


def test_moving_the_view_meanwhile_ends_the_flight_where_the_player_left_it() -> None:
    camera = IsoCamera(center_x=10.0, center_y=10.0, zoom_level=3)
    flight = Flight(camera, (50.0, 30.0), 5)
    flown(camera, flight, 0.5)
    camera.center_x = 5.0

    flying = flight.fly(camera, 1 / 60)

    assert not flying
    assert camera.center_x == 5.0
