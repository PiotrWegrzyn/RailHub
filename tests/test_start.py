import numpy as np
import pytest

from railroads.mapgen.sources import Shape
from railroads.mapgen.start import StartRegion, land_share, pick_start_bounds
from railroads.sim.geo import Bounds

REGION = StartRegion(west=0.0, east=20.0, south=0.0, north=10.0, min_land_share=0.9)
WEST_HALF_SEA = Shape(
    "polygon", (np.array([(-1.0, -1.0), (10.0, -1.0), (10.0, 11.0), (-1.0, 11.0)]),)
)
ALL_SEA = Shape("polygon", (np.array([(-1.0, -1.0), (21.0, -1.0), (21.0, 11.0), (-1.0, 11.0)]),))


def pick(seed: int, water: list[Shape]) -> Bounds:
    return pick_start_bounds(
        np.random.default_rng(seed),
        REGION,
        200,
        200,
        lambda bounds: land_share(bounds, water),
        attempts=200,
    )


def test_start_area_fits_inside_the_region() -> None:
    picks = [pick(seed, []) for seed in range(20)]

    inside = [REGION.contains(bounds) for bounds in picks]

    assert all(inside)


def test_start_area_keeps_out_of_the_sea() -> None:
    picks = [pick(seed, [WEST_HALF_SEA]) for seed in range(10)]

    shares = [land_share(bounds, [WEST_HALF_SEA]) for bounds in picks]

    assert min(shares) >= 0.9
    assert all(bounds.west > 9.0 for bounds in picks)


def test_start_area_has_the_requested_size() -> None:
    bounds = pick(3, [])

    size = (bounds.width_km, bounds.height_km)

    assert size == pytest.approx((200, 200))


def test_same_seed_gives_the_same_start_and_others_differ() -> None:
    first, again, other = pick(5, []), pick(5, []), pick(6, [])

    assert first == again
    assert first != other


def test_region_without_enough_land_fails() -> None:
    with pytest.raises(ValueError, match="land"):
        pick(1, [ALL_SEA])


def test_land_share_counts_what_the_water_leaves() -> None:
    bounds = Bounds(west=9.0, east=11.0, south=4.0, north=6.0)

    share = land_share(bounds, [WEST_HALF_SEA])

    assert share == pytest.approx(0.5, abs=0.05)
