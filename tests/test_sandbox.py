import numpy as np

from builders import flat_map
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.growth import Growth
from railroads.sim.sandbox import FOOD_ONLY, small_map
from railroads.sim.tilemap import Terrain


def test_the_sandbox_map_is_the_piece_of_a_map_around_its_river() -> None:
    terrain = np.full((120, 120), Terrain.SEA, dtype=np.uint8)
    terrain[61:111, 31:81] = Terrain.LAND
    terrain[61:111, 50] = Terrain.RIVER
    big = flat_map(120, 120, terrain=terrain)

    small = small_map(big, 50)

    assert small.grid.shape == (50, 50)
    assert (small.terrain == Terrain.RIVER).sum() >= 49
    assert (small.terrain == Terrain.LAND).sum() >= 49 * 49 - 50


def test_the_sandbox_grows_only_houses_and_food() -> None:
    growth = Growth(Constructions(flat_map(30, 30)), FOOD_ONLY, start_year=850)

    with growth.log_changes() as changes:
        growth.run(12 * 600, np.random.default_rng(1))

    grown = {kind for _, kind in changes}
    assert BuildingKind.HOUSE in grown
    assert grown <= {
        BuildingKind.HOUSE,
        BuildingKind.FARM,
        BuildingKind.ORCHARD,
        BuildingKind.FISHERY,
    }
