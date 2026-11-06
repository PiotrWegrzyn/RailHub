import pytest

from railroads.sim.geo import Bounds, describe_lonlat


@pytest.mark.parametrize(
    ("lon", "lat", "expected"),
    [(21.01, 52.23, "52.23°N 21.01°E"), (-70.65, -33.45, "33.45°S 70.65°W")],
)
def test_places_are_described_with_compass_letters(lon: float, lat: float, expected: str) -> None:
    text = describe_lonlat(lon, lat)

    assert text == expected


def test_bounds_centre_is_halfway_between_the_edges() -> None:
    bounds = Bounds(west=10.0, east=20.0, south=40.0, north=50.0)

    center = bounds.center

    assert center == (15.0, 45.0)


def test_km_round_trip_through_lonlat() -> None:
    bounds = Bounds(west=10.0, east=20.0, south=40.0, north=50.0)

    lon, lat = bounds.to_lonlat(*bounds.to_km(17.5, 44.0))

    assert (lon, lat) == pytest.approx((17.5, 44.0))
