from dataclasses import replace

import pytest
from trading import STATION, counts, holding, position
from trading import make_map as trade_map

from builders import flat_map, layer
from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import Economy
from railroads.sim.geo import Bounds
from railroads.sim.goods import FITS, Good, Need
from railroads.sim.growth import inputs_of
from railroads.sim.ladders import SOY_FIELDS, TEA_PLANTATIONS
from railroads.sim.tilemap import Relief, TileMap
from railroads.sim.yearly import bonus_of, needs_of


def placed_at(west: float, south: float, relief: Relief = Relief.FLAT) -> TileMap:
    tile_map = flat_map(relief=layer(12, 30, relief))
    return replace(tile_map, bounds=Bounds(west, west + 1.0, south, south + 1.0))


@pytest.mark.parametrize(
    ("west", "south", "grows"),
    [(135.0, 34.5, True), (120.0, 30.5, True), (126.0, 37.0, True), (20.0, 51.0, False)],
)
def test_soy_grows_only_in_east_asia(west: float, south: float, grows: bool) -> None:
    tile_map = placed_at(west, south)

    weights = SOY_FIELDS.land.weights(tile_map)

    assert bool((weights > 0).all()) == grows


def test_soy_stays_out_of_east_asia_s_cold_north() -> None:
    tile_map = placed_at(125.0, 51.0)

    weights = SOY_FIELDS.land.weights(tile_map)

    assert weights.max() == 0


@pytest.mark.parametrize(
    ("west", "south", "grows"),
    [(135.0, 34.5, True), (91.0, 26.0, True), (80.0, 7.0, True), (135.0, 40.0, False)],
)
def test_tea_grows_only_in_the_warm_south_and_east_of_asia(
    west: float, south: float, grows: bool
) -> None:
    tile_map = placed_at(west, south, Relief.HILLS)

    weights = TEA_PLANTATIONS.land.weights(tile_map)

    assert bool((weights > 0).all()) == grows


def test_tea_likes_hillsides_five_times_as_much_as_plains() -> None:
    hills = TEA_PLANTATIONS.land.weights(placed_at(135.0, 34.5, Relief.HILLS))[5, 15]

    plains = TEA_PLANTATIONS.land.weights(placed_at(135.0, 34.5))[5, 15]

    assert hills / plains == pytest.approx(5.0)


def test_tea_and_coffee_both_meet_the_need_for_hot_drinks() -> None:
    drinks = FITS[Need.HOT_DRINKS]

    tenement = needs_of(BuildingKind.TENEMENT, 1850)

    assert set(drinks) == {Good.COFFEE, Good.TEA}
    assert Need.HOT_DRINKS in tenement and Need.COFFEE not in tenement


@pytest.mark.parametrize(
    ("kind", "inputs", "made", "amount"),
    [
        (BuildingKind.TOFU_MAKER, {"SOYBEANS": 10.0}, Good.TOFU, 25.0),
        (BuildingKind.MISO_MAKER, {"SOYBEANS": 5.0, "RICE": 3.0, "SALT": 2.0}, Good.MISO, 20.0),
        (BuildingKind.FISH_DRYER, {"FISH": 8.0, "SALT": 2.0}, Good.DRIED_FISH, 20.0),
        (
            BuildingKind.SOY_SAUCE_BREWERY,
            {"SOYBEANS": 5.0, "WHEAT": 5.0, "SALT": 2.0},
            Good.SOY_SAUCE,
            10.0,
        ),
    ],
)
def test_the_soy_and_fish_works_make_their_goods_from_their_inputs(
    kind: BuildingKind, inputs: dict[str, float], made: Good, amount: float
) -> None:
    tile_map = trade_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), kind, 1))

    trade = economy.trade(placed, holding(100.0, **inputs))

    assert trade.gathered[STATION][made] == pytest.approx(amount)


def test_a_house_eats_the_tofu_a_station_brings() -> None:
    tile_map = trade_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(TOFU=20.0))

    eaten, _ = trade.goods_got[(BuildingKind.HOUSE, Need.FOOD)]
    assert eaten[position(trade, economy, (8, 9))] == pytest.approx(20.0)


@pytest.mark.parametrize(
    "kind", [BuildingKind.HOUSE, BuildingKind.TENEMENT, BuildingKind.FLATS, BuildingKind.SUSHI_HUT]
)
def test_soy_sauce_is_a_bonus_for_homes_and_sushi_huts(kind: BuildingKind) -> None:
    bonus = bonus_of(kind, 1850)

    assert Need.SOY_SAUCE in bonus


def test_miso_needs_soy_rice_and_salt_on_the_map() -> None:
    makers = inputs_of(BuildingKind.MISO_MAKER, 1850)

    assert [BuildingKind.SOY_FIELD in m for m in makers] == [True, False, False]
    assert any(BuildingKind.RICE_PADDY in m for m in makers)
    assert any(BuildingKind.SALT_WORKS in m for m in makers)
