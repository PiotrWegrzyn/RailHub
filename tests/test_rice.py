from dataclasses import replace

import numpy as np
import pytest
from test_economy import STATION, counts, holding, position
from test_economy import make_map as trade_map
from test_growth import NO_TRADE, build

from builders import flat_map, layer
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.economy import Economy
from railroads.sim.geo import Bounds
from railroads.sim.goods import FITS, Good, Need
from railroads.sim.growth import VENUES, Growth
from railroads.sim.ladders import (
    LADDERS,
    RICE_PADDIES,
    SAKE_BREWERIES,
    SUSHI_APART,
    SUSHI_HUTS,
    SUSHI_LAND_CELLS,
    SUSHI_TOWN,
    TAVERNS,
)
from railroads.sim.tech import GOOD_TREES, Tree
from railroads.sim.tilemap import Relief, Terrain, TileMap, Vegetation
from railroads.sim.trains import BULK_CAR_GOODS


def at_latitude(south: float, north: float) -> TileMap:
    return replace(flat_map(), bounds=Bounds(west=135.0, east=136.0, south=south, north=north))


@pytest.mark.parametrize(
    ("south", "north", "grows"),
    [(34.5, 35.5, True), (-36.0, -35.0, True), (49.5, 50.5, False), (-46.0, -45.0, False)],
)
def test_rice_grows_only_within_41_degrees_of_the_equator(
    south: float, north: float, grows: bool
) -> None:
    tile_map = at_latitude(south, north)

    weights = RICE_PADDIES.land.weights(tile_map)

    assert bool((weights > 0).all()) == grows


def test_a_map_across_the_41st_parallel_grows_rice_only_south_of_it() -> None:
    tile_map = at_latitude(40.85, 41.15)

    weights = RICE_PADDIES.land.weights(tile_map)

    assert weights[0].max() == 0
    assert weights[-1].min() > 0


def by_water(water: Terrain) -> float:
    terrain = layer(12, 30, Terrain.LAND)
    terrain[:, 0] = water
    tile_map = replace(flat_map(terrain=terrain), bounds=Bounds(135.0, 136.0, 34.5, 35.5))
    weights = RICE_PADDIES.land.weights(tile_map)
    return float(weights[5, 1] / weights[5, 10])


def test_salty_sea_water_does_rice_no_good() -> None:
    beside_sea = by_water(Terrain.SEA)

    assert beside_sea == pytest.approx(1.0)


@pytest.mark.parametrize("water", [Terrain.RIVER, Terrain.LAKE])
def test_rice_grows_four_times_as_readily_by_fresh_water(water: Terrain) -> None:
    beside_fresh_water = by_water(water)

    assert beside_fresh_water == pytest.approx(4.0)


def test_rice_grows_on_flat_land_and_hills_but_never_in_mountains_or_forest() -> None:
    relief = layer(12, 30, Relief.FLAT)
    relief[:, 10:20] = Relief.HILLS
    relief[:, 20:] = Relief.MOUNTAINS
    vegetation = layer(12, 30, Vegetation.GRASS)
    vegetation[:2, :10] = Vegetation.FOREST
    tile_map = replace(
        flat_map(relief=relief, vegetation=vegetation), bounds=Bounds(135.0, 136.0, 34.5, 35.5)
    )

    weights = RICE_PADDIES.land.weights(tile_map)

    assert weights[5, 5] > 0
    assert weights[5, 15] > 0
    assert weights[5, 25] == 0
    assert weights[0, 5] == 0


def test_rice_feeds_homes_and_rides_in_bulk_cars_with_the_food_tree() -> None:
    food = FITS[Need.FOOD]

    assert Good.RICE in food
    assert Good.RICE in BULK_CAR_GOODS
    assert GOOD_TREES[Good.RICE] == Tree.FOOD
    assert RICE_PADDIES in LADDERS


def test_a_house_eats_the_rice_a_station_brings() -> None:
    tile_map = trade_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(RICE=20.0))

    eaten, _ = trade.goods_got[(BuildingKind.HOUSE, Need.FOOD)]
    assert eaten[position(trade, economy, (8, 9))] == pytest.approx(20.0)


@pytest.mark.parametrize(("people", "opens"), [(SUSHI_TOWN - 1, False), (SUSHI_TOWN, True)])
def test_a_sushi_hut_opens_only_where_twenty_people_live_around(people: int, opens: bool) -> None:
    constructions = Constructions(at_latitude(34.5, 35.5))
    for cell, houses in zip(((5, 5), (5, 4), (5, 6), (4, 5)), (6, 6, 6, people - 18), strict=True):
        build(constructions, cell, BuildingKind.HOUSE, houses)

    chance = Growth(constructions, NO_TRADE).chances(SUSHI_HUTS)[5, 5]

    assert (chance > 0) == opens


def test_a_sushi_hut_takes_a_lot_of_fish_and_rice_and_livens_the_homes_around() -> None:
    tile_map = trade_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.SUSHI_HUT, 1))

    trade = economy.trade(placed, holding(FISH=100.0, RICE=100.0))

    here = position(trade, economy, (8, 8))
    fish, _ = trade.goods_got[(BuildingKind.SUSHI_HUT, Need.FISH)]
    rice, _ = trade.goods_got[(BuildingKind.SUSHI_HUT, Need.RICE)]
    assert fish[here] == pytest.approx(60.0)
    assert rice[here] == pytest.approx(60.0)
    assert BuildingKind.SUSHI_HUT in VENUES


def test_a_sushi_hut_with_fish_but_no_rice_is_not_fully_served() -> None:
    tile_map = trade_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.SUSHI_HUT, 1))
    fed = economy.trade(placed, holding(FISH=100.0, RICE=100.0))

    fish_only = economy.trade(placed, holding(FISH=100.0))

    share = fish_only.share((BuildingKind.SUSHI_HUT,), empty=0.0)
    full = fed.share((BuildingKind.SUSHI_HUT,), empty=0.0)
    assert np.max(share) < np.max(full)


def test_no_sushi_hut_opens_where_rice_cannot_grow() -> None:
    constructions = Constructions(at_latitude(49.5, 50.5))
    for cell in ((5, 5), (5, 4), (5, 6), (4, 5)):
        build(constructions, cell, BuildingKind.HOUSE, 6)

    chance = Growth(constructions, NO_TRADE).chances(SUSHI_HUTS)[5, 5]

    assert chance == 0


def test_sushi_huts_are_a_rare_treat_next_to_taverns() -> None:
    rarity = SUSHI_HUTS.base_chance / TAVERNS.base_chance

    assert rarity == pytest.approx(0.1)
    assert SUSHI_HUTS.apart == SUSHI_APART
    assert SUSHI_HUTS.land_cells_each == SUSHI_LAND_CELLS


def test_a_sake_brewery_brews_rice_into_sake_that_taverns_pour_as_alcohol() -> None:
    tile_map = trade_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.SAKE_BREWERY, 1))

    trade = economy.trade(placed, holding(100.0, RICE=10.0))

    assert trade.gathered[STATION][Good.SAKE] == pytest.approx(20.0)
    assert Good.SAKE in FITS[Need.ALCOHOL]


def test_a_tavern_takes_sake_for_its_alcohol() -> None:
    tile_map = trade_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.TAVERN, 1))

    trade = economy.trade(placed, holding(SAKE=200.0))

    poured, _ = trade.goods_got[(BuildingKind.TAVERN, Need.ALCOHOL)]
    assert poured[position(trade, economy, (8, 8))] == pytest.approx(90.0)


@pytest.mark.parametrize(("south", "north", "opens"), [(34.5, 35.5, True), (49.5, 50.5, False)])
def test_sake_breweries_open_only_where_rice_grows(south: float, north: float, opens: bool) -> None:
    tile_map = at_latitude(south, north)

    weights = SAKE_BREWERIES.land.weights(tile_map)

    assert bool((weights > 0).all()) == opens
