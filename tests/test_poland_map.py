import pytest

from railroads.sim.maps import poland
from railroads.sim.world import Terrain, World


@pytest.fixture(scope="module")
def world() -> World:
    return poland.generate()


def terrain_at_lonlat(world: World, lon: float, lat: float) -> Terrain:
    x, y = poland.PROJECTION.to_tile(lon, lat)
    return world.terrain_at(int(x), int(y))


def test_map_covers_poland_at_two_km_per_tile(world: World) -> None:
    size = (world.width_km, world.height_km)

    assert 700 < size[0] < 800
    assert 650 < size[1] < 720


@pytest.mark.parametrize(
    ("place", "lon", "lat", "expected"),
    [
        ("Baltic Sea north of Leba", 17.5, 54.95, Terrain.WATER),
        ("Lodz", 19.46, 51.76, Terrain.PLAINS),
        ("Tatras", 20.0, 49.25, Terrain.MOUNTAINS),
        ("Vistula in Warsaw", 21.02, 52.23, Terrain.WATER),
    ],
)
def test_known_places_have_expected_terrain(
    world: World, place: str, lon: float, lat: float, expected: Terrain
) -> None:
    terrain = terrain_at_lonlat(world, lon, lat)

    assert terrain is expected, place
