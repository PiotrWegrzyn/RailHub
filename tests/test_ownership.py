import numpy as np
import pytest

from builders import QUIET, flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.goods import Good, price_of
from railroads.sim.growth import Growth
from railroads.sim.ladders import FACTORIES, FARMS_CROWDED_OUT, GrowthRule
from railroads.sim.ownership import OWNER_SHARE, PAYBACK_YEARS
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route


def grown(*placed: tuple[tuple[int, int], BuildingKind]) -> Growth:
    constructions = Constructions(flat_map())
    for cell, kind in placed:
        constructions.build(Building(cell, kind))
    growth = Growth(constructions, QUIET)
    growth.railway = Railway(constructions)
    return growth


def test_a_building_costs_payback_years_of_the_players_share_of_what_it_can_make() -> None:
    growth = grown(((5, 5), BuildingKind.FARM))
    cash = growth.railway.finances.cash

    problem = growth.owners.buy((5, 5), BuildingKind.FARM)

    price = 12 * PAYBACK_YEARS * OWNER_SHARE * 10 * price_of(Good.WHEAT)
    assert problem is None
    assert growth.owners.owned(BuildingKind.FARM)[5, 5] == 1
    assert growth.railway.finances.cash == pytest.approx(cash - price)


@pytest.mark.parametrize(
    ("placed", "kind", "problem"),
    [
        ([((5, 5), BuildingKind.HOUSE)], BuildingKind.HOUSE, "A house cannot be bought"),
        ([((5, 5), BuildingKind.MINE)], BuildingKind.MINE, "You already own every mine here"),
    ],
)
def test_homes_cannot_be_bought_nor_more_than_stand_there(
    placed: list, kind: BuildingKind, problem: str
) -> None:
    growth = grown(*placed)
    growth.railway.finances.earn(1_000_000.0, "test")
    if kind == BuildingKind.MINE:
        growth.owners.buy((5, 5), kind)

    said = growth.owners.buy((5, 5), kind)

    assert said == problem


def joined_with_owned_farm(*houses: tuple[int, int]) -> Growth:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, (6, 3))
    build_station(growth, (6, 20))
    route = plan_route(constructions, (6, 3), (6, 20))
    assert route is not None
    lay_route(growth, route)
    growth.railway = Railway(constructions)
    growth.build(Building((6, 22), BuildingKind.FARM))
    for cell in houses:
        growth.build(Building(cell, BuildingKind.HOUSE))
    growth.owners.own((6, 22), BuildingKind.FARM)
    return growth


def test_a_station_hands_out_the_players_share_of_their_farm_and_it_pays_as_a_delivery() -> None:
    growth = joined_with_owned_farm((6, 21), (6, 21), (6, 21))

    growth.trade_goods()

    assert growth.trade.handed[(6, 20)][Good.WHEAT] == pytest.approx(10 * OWNER_SHARE)
    assert growth.railway.finances.month["cargo"] > 0
    assert growth.railway.finances.month.get("owned buildings", 0.0) == 0.0


def test_an_owned_farm_far_from_any_station_pays_the_player_nothing() -> None:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    growth.railway = Railway(constructions)
    growth.build(Building((6, 10), BuildingKind.FARM))
    growth.build(Building((6, 11), BuildingKind.HOUSE))
    growth.owners.own((6, 10), BuildingKind.FARM)

    growth.trade_goods()

    assert not any(growth.railway.finances.month.values())


def test_a_crowded_out_farm_of_the_players_goes_up_again_nearby_and_stays_theirs() -> None:
    growth = grown(((5, 5), BuildingKind.FARM))
    growth.owners.own((5, 5), BuildingKind.FARM)
    neighbours = growth.constructions.tile_map.grid.neighbours(5, 5)
    for i in range(FARMS_CROWDED_OUT):
        growth.build(Building(neighbours[i % len(neighbours)], BuildingKind.HOUSE))

    growth._crowd_out()

    owned = growth.owners.owned(BuildingKind.FARM)
    assert growth.count(BuildingKind.FARM)[5, 5] == 0
    assert owned.sum() == 1
    assert growth.count(BuildingKind.FARM)[tuple(np.argwhere(owned)[0])] == 1


def test_factories_merging_into_a_big_one_hand_it_to_whoever_owned_one_of_them() -> None:
    growth = grown(
        ((5, 5), BuildingKind.FACTORY),
        ((5, 6), BuildingKind.FACTORY),
        ((5, 6), BuildingKind.FACTORY),
    )
    growth.owners.own((5, 6), BuildingKind.FACTORY)

    growth._turn((5, 5), FACTORIES, np.random.default_rng(0))

    assert growth.count(BuildingKind.BIG_FACTORY)[5, 5] == 1
    assert growth.owners.owned(BuildingKind.BIG_FACTORY)[5, 5] == 1
    assert growth.owners.owned(BuildingKind.FACTORY).sum() == 0


def test_the_company_flies_its_flag_only_over_the_cells_where_it_owns_a_building() -> None:
    growth = grown(((5, 5), BuildingKind.FARM), ((5, 7), BuildingKind.FARM))

    growth.owners.buy((5, 7), BuildingKind.FARM)

    assert growth.owners.company == "The Trains Corp"
    assert growth.owners.cells() == [(5, 7)]
