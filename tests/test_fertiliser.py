import pytest
from trading import STATION, counts, holding, make_map

from builders import boards_with
from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import Economy
from railroads.sim.goods import Good
from railroads.sim.tiles import Tile, activate


def test_a_chemical_works_leaves_half_a_unit_of_fertiliser_over_every_unit_of_chemicals() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.CHEMICAL_WORKS, 1))

    trade = economy.trade(placed, holding(1000.0, SALT=30.0, COAL=30.0))

    made = trade.gathered[STATION]
    assert made[Good.CHEMICALS] == pytest.approx(60.0)
    assert made[Good.FERTILISER] == pytest.approx(30.0)


@pytest.mark.parametrize(("fertiliser", "wheat"), [(0.0, 6.0), (1.5, 7.5), (3.0, 9.0)])
def test_fertiliser_lets_a_farm_grow_up_to_half_as_much_again(
    fertiliser: float, wheat: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1950
    placed = counts(tile_map, ((8, 9), BuildingKind.FARM, 1))

    trade = economy.trade(placed, holding(1000.0, FERTILISER=fertiliser))

    assert trade.gathered[STATION].get(Good.WHEAT, 0.0) == pytest.approx(wheat)


@pytest.mark.parametrize(
    ("kind", "good", "made"),
    [(BuildingKind.MINE, Good.COAL, 45.0), (BuildingKind.IRON_MINE, Good.IRON, 45.0)],
)
def test_machinery_lets_a_mine_dig_up_to_half_as_much_again(
    kind: BuildingKind, good: Good, made: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1950
    placed = counts(tile_map, ((8, 9), kind, 1))

    trade = economy.trade(placed, holding(1000.0, MACHINERY=5.0))

    assert trade.gathered[STATION].get(good, 0.0) == pytest.approx(made)


@pytest.mark.parametrize(
    ("made", "wanted", "gas"), [(1.0, 1.0, 100.0), (1.5, 1.0, 150.0), (1.0, 1.25, 80.0)]
)
def test_supply_makes_a_works_make_more_from_its_inputs_and_demand_take_more_for_as_much(
    made: float, wanted: float, gas: float
) -> None:
    activate(boards_with({Tile.OIL: made}, {Tile.COAL: wanted}))
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.GASWORKS, 1))

    trade = economy.trade(placed, holding(1000.0, COAL=50.0))

    assert trade.gathered[STATION][Good.GAS] == pytest.approx(gas)
