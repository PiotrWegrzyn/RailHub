import pytest
from trading import STATION, counts, holding, make_map

from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import Economy
from railroads.sim.goods import Good, Need
from railroads.sim.growth import ladder_open
from railroads.sim.ladders import LADDERS
from railroads.sim.tiles import Tile, Tiles, activate, of_year
from railroads.sim.yearly import bonus_of


def first_year_at(tile: Tile, level: int) -> int:
    return next(year for year in range(1850, 2100) if of_year(year).demand.get(tile, 0) >= level)


def test_every_home_takes_cola_as_a_bonus_only_once_food_and_drink_demand_reaches_its_level() -> (
    None
):
    activate(Tiles.until(2050))
    opened = first_year_at(Tile.FOOD, 13)

    before = Need.COLA in bonus_of(BuildingKind.HOUSE, opened - 1)
    after = [
        Need.COLA in bonus_of(kind, opened)
        for kind in (BuildingKind.HOUSE, BuildingKind.TENEMENT, BuildingKind.FLATS)
    ]

    assert not before
    assert after == [True, True, True]


@pytest.mark.parametrize(
    ("tile", "need", "level"), [(Tile.VEHICLES, Need.CARS, 13), (Tile.MODERN, Need.APPLIANCES, 14)]
)
def test_tenements_take_cars_and_appliances_a_demand_level_before_houses(
    tile: Tile, need: Need, level: int
) -> None:
    activate(Tiles.until(2050))
    tenements_from = first_year_at(tile, level)

    taken = [
        need in bonus_of(kind, tenements_from)
        for kind in (BuildingKind.TENEMENT, BuildingKind.HOUSE)
    ]

    assert taken == [True, False]


def test_cola_bottling_plants_come_up_only_once_homes_want_cola() -> None:
    activate(Tiles.until(2050))
    opened = first_year_at(Tile.FOOD, 13)
    ladder = next(ladder for ladder in LADDERS if ladder.kinds[0] == BuildingKind.COLA_BOTTLER)

    states = [ladder_open(ladder, year) for year in (opened - 1, opened)]

    assert states == [False, True]


def test_limestone_lets_a_chemical_works_make_up_to_half_as_much_again() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.CHEMICAL_WORKS, 1))

    trade = economy.trade(placed, holding(1000.0, SALT=30.0, COAL=30.0, LIMESTONE=100.0))

    assert trade.gathered[STATION][Good.CHEMICALS] == pytest.approx(90.0)


def test_potatoes_let_a_pig_farm_raise_up_to_half_as_many_pigs_again() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.PIG_FARM, 1))
    without = economy.trade(placed, holding(1000.0))

    fed = economy.trade(placed, holding(1000.0, POTATOES=100.0))

    pigs = without.gathered[STATION][Good.PIGS]
    assert fed.gathered[STATION][Good.PIGS] == pytest.approx(pigs * 1.5)
