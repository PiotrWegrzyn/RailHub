from dataclasses import replace

import numpy as np
import pytest

from builders import flat_map
from railroads.mapgen.deposits import place_deposits
from railroads.mapgen.seeding import (
    FORESTS_EVERY_MONTHS,
    TRADE_EVERY_MONTHS,
    play_begins,
    seed_buildings,
)
from railroads.sim.clock import HISTORY_START
from railroads.sim.constructions import Building, BuildingKind, Constructions, Station, Track
from railroads.sim.economy import Neighbourhoods
from railroads.sim.forests import MILLS_CLOSE_BELOW, forests_nearby
from railroads.sim.grids import HexGrid
from railroads.sim.growth import MONTHS_BEFORE_PLAY, Growth
from railroads.sim.ladders import (
    FACTORIES,
    FARMS,
    GROWTH,
    LUMBER_MILLS,
    MINES,
    PORTS,
    TOWNS,
    Ladder,
)
from railroads.sim.tilemap import Relief, Terrain, TileMap, Vegetation


def busier(ladder: Ladder, times: float = 10) -> Ladder:
    return replace(
        ladder,
        base_chance=times * ladder.base_chance,
        chance_per_doubling={p: times * v for p, v in ladder.chance_per_doubling.items()},
    )


# A small country grows the industry under test ten times as fast, and farms a hundred times, so
# that each turns up in it often enough to judge where; towns grow as they always do.
INDUSTRY = (busier(FARMS), MINES, LUMBER_MILLS, PORTS, FACTORIES)
BUSY = replace(GROWTH, ladders=(TOWNS, *(busier(ladder) for ladder in INDUSTRY)))


def make_map(
    terrain: np.ndarray,
    relief: np.ndarray | None = None,
    vegetation: np.ndarray | None = None,
) -> TileMap:
    grid = HexGrid(rows=terrain.shape[0], cols=terrain.shape[1], km_per_cell=2.0)
    return flat_map(
        grid=grid,
        terrain=terrain.astype(np.uint8),
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8)
        if vegetation is None
        else vegetation.astype(np.uint8),
        relief=np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
        if relief is None
        else relief.astype(np.uint8),
    )


def make_country(size: int = 80) -> TileMap:
    terrain = np.full((size, size), Terrain.LAND)
    terrain[:, :5] = Terrain.SEA
    terrain[:, size // 2] = Terrain.RIVER
    relief = np.full((size, size), Relief.FLAT)
    relief[size // 2 :, -25:] = Relief.HILLS
    relief[-15:, -25:] = Relief.MOUNTAINS
    vegetation = np.full((size, size), Vegetation.GRASS)
    vegetation[: size // 3, -30:] = Vegetation.FOREST
    return make_map(terrain, relief, vegetation)


def buildings_of(constructions: Constructions, kind: BuildingKind) -> list[Building]:
    return [b for b in constructions if isinstance(b, Building) and b.kind == kind]


Grown = tuple[Constructions, set[tuple[tuple[int, int], BuildingKind]]]


@pytest.fixture(scope="module")
def history() -> Grown:
    """The country grown as seed_buildings grows it, with every cell and kind that went up on the
    way, standing at the end or not."""
    tile_map = make_country()
    rng = np.random.default_rng(1)
    tile_map = replace(
        tile_map, vegetation=tile_map.vegetation.copy(), deposits=place_deposits(tile_map, rng)
    )
    constructions = Constructions(tile_map)
    rule = replace(
        BUSY, trade_every_months=TRADE_EVERY_MONTHS, forests_every_months=FORESTS_EVERY_MONTHS
    )
    growth = Growth(constructions, rule, start_year=HISTORY_START)
    with growth.log_changes() as changes:
        growth.run(MONTHS_BEFORE_PLAY, rng)
    return constructions, set(changes)


@pytest.fixture(scope="module")
def country(history: Grown) -> Constructions:
    return history[0]


def test_seeding_grows_towns_and_the_industry_that_suits_the_land(history: Grown) -> None:
    kinds = {kind for _, kind in history[1]}

    assert {
        BuildingKind.HOUSE,
        BuildingKind.FARM,
        BuildingKind.MINE,
        BuildingKind.LUMBER_MILL,
        BuildingKind.FACTORY,
    } <= kinds


@pytest.mark.parametrize(
    ("kind", "allowed"),
    [
        (BuildingKind.FARM, lambda t: t.vegetation != Vegetation.FOREST),
        (BuildingKind.MINE, lambda t: t.relief != Relief.FLAT),
    ],
)
def test_seeded_industry_mostly_goes_up_where_its_land_suits_it(
    history: Grown, kind: BuildingKind, allowed: object
) -> None:
    country, grown = history
    tile_map = country.tile_map

    built = [cell for cell, grown_kind in grown if grown_kind == kind]

    suited = [allowed(tile_map.tile(cell)) for cell in built]  # type: ignore[operator]
    assert built
    assert np.mean(suited) > 0.9


def test_seeded_lumber_mills_all_keep_enough_forest_around_them(history: Grown) -> None:
    country, grown = history
    tile_map = country.tile_map
    forests = forests_nearby(tile_map, Neighbourhoods(tile_map.grid).index)

    mills = buildings_of(country, BuildingKind.LUMBER_MILL)

    assert any(kind == BuildingKind.LUMBER_MILL for _, kind in grown)
    assert all(
        forests[np.ravel_multi_index(mill.cell, tile_map.grid.shape)] >= MILLS_CLOSE_BELOW
        for mill in mills
    )


def test_ports_stand_by_the_water(country: Constructions) -> None:
    tile_map = country.tile_map

    ports = buildings_of(country, BuildingKind.PORT)

    terrain = tile_map.terrain
    assert all(
        any(terrain[near] != Terrain.LAND for near in tile_map.grid.neighbours(*port.cell))
        for port in ports
    )


def test_seeded_counts_match_a_fresh_count(country: Constructions) -> None:
    growth = Growth(country)

    total = sum(int(growth.count(kind).sum()) for kind in growth.rule.points)

    assert total == len(buildings_of_all(country))


def buildings_of_all(constructions: Constructions) -> list[Building]:
    return [built for built in constructions if isinstance(built, Building)]


def test_seeding_with_the_same_seed_gives_the_same_buildings() -> None:
    tile_map = make_country(40)

    first = list(seed_buildings(tile_map, np.random.default_rng(3), months=600))
    second = list(seed_buildings(tile_map, np.random.default_rng(3), months=600))

    assert first == second


def test_seeding_leaves_track_to_the_player(country: Constructions) -> None:
    built = list(country)

    assert not any(isinstance(construction, Track) for construction in built)


def test_seeding_leaves_stations_to_the_player() -> None:
    constructions = seed_buildings(make_country(), np.random.default_rng(2), months=1500)

    stations = [built for built in constructions if isinstance(built, Station)]

    assert not stations


@pytest.mark.parametrize(("years", "begins"), [(1000, 1850), (1100, 1950), (1200, 2050)])
def test_history_begins_in_850_and_play_where_it_left_off(years: int, begins: int) -> None:
    year = play_begins(years * 12)

    assert year == begins
    assert HISTORY_START == 850
