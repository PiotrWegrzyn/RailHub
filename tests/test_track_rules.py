from railroads.sim.tilemap import Relief
from railroads.sim.track_rules import RELIEF_RULES


def test_track_cannot_be_built_on_mountains() -> None:
    rules = RELIEF_RULES[Relief.MOUNTAINS]

    assert not rules.can_build


def test_hills_make_track_dearer_and_trains_slower_than_flat_land() -> None:
    flat, hills = RELIEF_RULES[Relief.FLAT], RELIEF_RULES[Relief.HILLS]

    assert hills.can_build
    assert hills.cost_factor > flat.cost_factor == 1.0
    assert hills.speed_factor < flat.speed_factor == 1.0
