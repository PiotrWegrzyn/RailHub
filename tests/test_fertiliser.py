import pytest
from test_economy import STATION, counts, holding, make_map

from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import Economy
from railroads.sim.goods import Good


def test_a_chemical_works_leaves_half_a_unit_of_fertiliser_over_every_unit_of_chemicals() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.CHEMICAL_WORKS, 1))

    trade = economy.trade(placed, holding(1000.0, SALT=50.0, COAL=50.0))

    made = trade.gathered[STATION]
    assert made[Good.CHEMICALS] == pytest.approx(100.0)
    assert made[Good.FERTILISER] == pytest.approx(50.0)


@pytest.mark.parametrize(("fertiliser", "wheat"), [(0.0, 10.0), (2.5, 12.5), (5.0, 15.0)])
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
    [(BuildingKind.MINE, Good.COAL, 150.0), (BuildingKind.IRON_MINE, Good.IRON, 150.0)],
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
