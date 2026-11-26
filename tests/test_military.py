from dataclasses import replace

import numpy as np
import pytest

from builders import square_map
from railroads.mapgen.deposits import GARRISON_SIZE, GARRISONS, place_deposits
from railroads.mapgen.seeding import muster_garrisons
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.economy import Economy
from railroads.sim.goods import NEEDS, Good, Need
from railroads.sim.growth import Growth
from railroads.sim.ladders import AIRPORTS, MILITARY, ORANGE_FARMS, GrowthRule
from railroads.sim.tilemap import Deposit, Temperature, Vegetation
from railroads.sim.yearly import needs_of


def garrison_at(*cells: tuple[int, int]) -> np.ndarray:
    deposits = np.full((40, 40), Deposit.NONE, dtype=np.uint8)
    for cell in cells:
        deposits[cell] = Deposit.GARRISON
    return deposits


def count(constructions: Constructions, kind: BuildingKind) -> int:
    return sum(1 for b in constructions if isinstance(b, Building) and b.kind == kind)


@pytest.mark.parametrize("seed", range(6))
def test_a_map_gets_only_a_few_small_patches_of_military_ground(seed: int) -> None:
    layer = place_deposits(square_map(), np.random.default_rng(seed))

    ground = int((layer == Deposit.GARRISON).sum())

    assert GARRISON_SIZE[0] <= ground <= GARRISONS[1] * GARRISON_SIZE[1]


def test_outposts_open_only_on_military_ground() -> None:
    ground = [(10, 10), (10, 11), (11, 10), (11, 11), (12, 10), (12, 11)]
    constructions = Constructions(square_map(deposits=garrison_at(*ground)))
    busy = replace(MILITARY, base=10 * MILITARY.base)
    growth = Growth(constructions, GrowthRule(ladders=(busy,), trades=False), start_year=1850)
    military = (BuildingKind.OUTPOST, BuildingKind.BASE, BuildingKind.HEADQUARTERS)

    with growth.log_changes() as changes:
        growth.run(240, np.random.default_rng(0))

    built = {cell for (cell, kind), _ in changes.items() if kind in military}
    assert built
    assert built <= set(ground)


def test_outposts_do_not_open_before_1800() -> None:
    constructions = Constructions(square_map(deposits=garrison_at((10, 10))))
    growth = Growth(constructions, start_year=1700)

    growth.run(600, np.random.default_rng(1))

    assert count(constructions, BuildingKind.OUTPOST) == 0


def military_on(constructions: Constructions, cells: set[tuple[int, int]]) -> list[BuildingKind]:
    return sorted(
        (b.kind for b in constructions if isinstance(b, Building) and b.cell in cells),
        key=lambda kind: kind.name,
    )


def test_each_patch_of_military_ground_starts_play_with_one_base_and_two_outposts() -> None:
    first = {(5, 5), (5, 6), (6, 5), (6, 6)}
    second = {(30, 30), (30, 31), (31, 30), (31, 31)}
    constructions = Constructions(square_map(deposits=garrison_at(*first, *second)))
    for _ in range(3):
        constructions.build(Building((5, 5), BuildingKind.BASE))

    muster_garrisons(constructions, 1850, np.random.default_rng(0))

    wanted = [BuildingKind.BASE, BuildingKind.OUTPOST, BuildingKind.OUTPOST]
    assert military_on(constructions, first) == wanted
    assert military_on(constructions, second) == wanted


def test_no_military_is_mustered_when_play_begins_before_1800() -> None:
    constructions = Constructions(square_map(deposits=garrison_at((5, 5), (5, 6))))

    muster_garrisons(constructions, 1750, np.random.default_rng(0))

    assert military_on(constructions, {(5, 5), (5, 6)}) == []


@pytest.mark.parametrize(
    ("small", "big"),
    [
        (BuildingKind.OUTPOST, BuildingKind.BASE),
        (BuildingKind.BASE, BuildingKind.HEADQUARTERS),
    ],
)
def test_three_of_a_rank_become_one_of_the_next(small: BuildingKind, big: BuildingKind) -> None:
    constructions = Constructions(square_map(deposits=garrison_at((10, 10))))
    for _ in range(3):
        constructions.build(Building((10, 10), small))
    growth = Growth(constructions)

    growth._turn((10, 10), MILITARY, np.random.default_rng(0))

    assert count(constructions, small) == 0
    assert count(constructions, big) == 1


@pytest.mark.parametrize(
    ("kind", "year", "need", "wanted"),
    [
        (BuildingKind.OUTPOST, 1872, Need.GUNS, False),
        (BuildingKind.OUTPOST, 1873, Need.GUNS, True),
        (BuildingKind.OUTPOST, 1900, Need.CARS, False),
        (BuildingKind.OUTPOST, 1901, Need.CARS, True),
        (BuildingKind.OUTPOST, 2000, Need.PLANES, False),
        (BuildingKind.BASE, 1932, Need.PLANES, False),
        (BuildingKind.BASE, 1933, Need.PLANES, True),
        (BuildingKind.BASE, 2000, Need.TANKS, False),
        (BuildingKind.HEADQUARTERS, 1932, Need.TANKS, False),
        (BuildingKind.HEADQUARTERS, 1933, Need.TANKS, True),
    ],
)
def test_what_the_military_wants_goes_by_rank_and_era(
    kind: BuildingKind, year: int, need: Need, wanted: bool
) -> None:
    needs = needs_of(kind, year)

    assert (need in needs) is wanted


def test_a_gun_factory_arms_an_outpost_next_to_it_on_its_own() -> None:
    tile_map = square_map(12)
    economy = Economy(tile_map)
    economy.year = 1880
    flat = int(np.ravel_multi_index((5, 5), tile_map.grid.shape))
    economy.stock[(BuildingKind.GUN_FACTORY, Need.STEEL)][flat] = 60.0
    economy.stock[(BuildingKind.GUN_FACTORY, Need.PLANKS)][flat] = 30.0
    factories = np.zeros(tile_map.grid.shape, dtype=np.int64)
    factories[5, 5] = 1
    outposts = np.zeros(tile_map.grid.shape, dtype=np.int64)
    outposts[tile_map.grid.neighbours(5, 5)[0]] = 1
    placed = {BuildingKind.GUN_FACTORY: factories, BuildingKind.OUTPOST: outposts}

    trade = economy.trade(placed)

    got, _ = trade.goods_got[(BuildingKind.OUTPOST, Need.GUNS)]
    assert got.sum() == NEEDS[BuildingKind.OUTPOST][Need.GUNS]


def test_soldiers_travel_more_the_higher_their_rank() -> None:
    constructions = Constructions(square_map())
    constructions.build(Building((5, 5), BuildingKind.OUTPOST))
    constructions.build(Building((9, 9), BuildingKind.HEADQUARTERS))
    growth = Growth(constructions)
    grid = constructions.tile_map.grid

    outpost = growth.station_trade.travellers_in(
        np.array([np.ravel_multi_index((5, 5), grid.shape)])
    )
    headquarters = growth.station_trade.travellers_in(
        np.array([np.ravel_multi_index((9, 9), grid.shape)])
    )

    assert headquarters[Good.PASSENGERS] > outpost[Good.PASSENGERS] > 0


@pytest.mark.parametrize(("flats", "houses", "opens"), [(4, 5, False), (5, 0, True)])
def test_airports_need_eighty_population_nearby(flats: int, houses: int, opens: bool) -> None:
    constructions = Constructions(square_map())
    for _ in range(flats):
        constructions.build(Building((10, 10), BuildingKind.FLATS))
    for _ in range(houses):
        constructions.build(Building((11, 11), BuildingKind.HOUSE))
    growth = Growth(constructions)

    chance = growth.chances(AIRPORTS)[10, 10]

    assert bool(chance > 0) is opens


@pytest.mark.parametrize(
    ("vegetation", "temperature", "grows"),
    [
        (Vegetation.GRASS, Temperature.HOT, True),
        (Vegetation.BARREN, Temperature.HOT, False),
        (Vegetation.FOREST, Temperature.HOT, False),
        (Vegetation.GRASS, Temperature.NORMAL, False),
    ],
)
def test_orange_farms_want_hot_land_but_not_desert_or_forest(
    vegetation: Vegetation, temperature: Temperature, grows: bool
) -> None:
    tile_map = square_map(12)
    tile_map.vegetation.flags.writeable = True
    tile_map.temperature.flags.writeable = True
    tile_map.vegetation[:] = vegetation
    tile_map.temperature[:] = temperature

    weights = ORANGE_FARMS.land.weights(tile_map)

    assert bool(weights[5, 5] > 0) is grows
