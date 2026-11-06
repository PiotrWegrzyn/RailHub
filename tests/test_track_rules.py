import pytest

from railroads.sim.tilemap import Relief, Temperature, Terrain, Tile, Vegetation
from railroads.sim.track_rules import RELIEF_RULES, TrackBed, track_rules_for


def make_tile(terrain: Terrain, relief: Relief = Relief.FLAT) -> Tile:
    return Tile(
        terrain=terrain,
        vegetation=Vegetation.GRASS,
        temperature=Temperature.NORMAL,
        height_m=0,
        relief=relief,
    )


def test_track_cannot_be_built_on_mountains() -> None:
    rules = RELIEF_RULES[Relief.MOUNTAINS]

    assert not rules.can_build


def test_hills_make_track_dearer_and_trains_slower_than_flat_land() -> None:
    flat, hills = RELIEF_RULES[Relief.FLAT], RELIEF_RULES[Relief.HILLS]

    assert hills.can_build
    assert hills.cost_factor > flat.cost_factor == 1.0
    assert hills.speed_factor < flat.speed_factor == 1.0


@pytest.mark.parametrize("relief", list(Relief))
def test_ground_track_on_land_follows_the_relief(relief: Relief) -> None:
    rules = track_rules_for(make_tile(Terrain.LAND, relief))

    assert rules == RELIEF_RULES[relief]


@pytest.mark.parametrize("terrain", [Terrain.RIVER, Terrain.LAKE, Terrain.SEA])
def test_ground_track_cannot_be_laid_on_water(terrain: Terrain) -> None:
    rules = track_rules_for(make_tile(terrain))

    assert not rules.can_build


@pytest.mark.parametrize(
    ("terrain", "can_build"),
    [(Terrain.RIVER, True), (Terrain.LAKE, True), (Terrain.SEA, False), (Terrain.LAND, False)],
)
def test_bridges_cross_rivers_and_lakes_but_not_the_sea(terrain: Terrain, can_build: bool) -> None:
    rules = track_rules_for(make_tile(terrain), TrackBed.BRIDGE)

    assert rules.can_build == can_build


@pytest.mark.parametrize(
    ("relief", "can_build"),
    [(Relief.FLAT, False), (Relief.HILLS, False), (Relief.MOUNTAINS, True)],
)
def test_tunnels_go_only_through_mountains(relief: Relief, can_build: bool) -> None:
    rules = track_rules_for(make_tile(Terrain.LAND, relief), TrackBed.TUNNEL)

    assert rules.can_build == can_build
