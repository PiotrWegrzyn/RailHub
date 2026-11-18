import numpy as np
import pytest

from builders import QUIET, flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.goods import Good, price_of
from railroads.sim.growth import Growth
from railroads.sim.ladders import FACTORIES, FARMS_CROWDED_OUT, GrowthRule
from railroads.sim.ownership import HOLD_MONTHS, LOT, PAYBACK_YEARS, SHARES, satisfaction_price
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route


def grown(*placed: tuple[tuple[int, int], BuildingKind]) -> Growth:
    constructions = Constructions(flat_map())
    for cell, kind in placed:
        constructions.build(Building(cell, kind))
    growth = Growth(constructions, QUIET)
    growth.railway = Railway(constructions)
    return growth


def satisfied(growth: Growth, satisfaction: float = 1.0) -> Growth:
    growth.economy.months = [np.full(growth.economy.cells, satisfaction) for _ in range(3)]
    return growth


def test_a_lot_of_shares_costs_its_part_of_payback_years_of_all_a_building_makes() -> None:
    growth = satisfied(grown(((5, 5), BuildingKind.FARM)))
    cash = growth.railway.finances.cash

    problem = growth.owners.buy((5, 5), BuildingKind.FARM)

    price = 12 * PAYBACK_YEARS * 10 * price_of(Good.WHEAT) * LOT / SHARES
    assert problem is None
    assert growth.owners.shares(BuildingKind.FARM)[5, 5] == LOT
    assert growth.railway.finances.cash == pytest.approx(cash - price)


@pytest.mark.parametrize(
    ("satisfaction", "factor"),
    [(0.0, 2 / 3), (0.5, 5 / 6), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0), (3.0, 2.0)],
)
def test_shares_cost_more_the_more_satisfied_the_building_was_over_the_quarter(
    satisfaction: float, factor: float
) -> None:
    price = satisfaction_price(satisfaction)

    assert price == pytest.approx(factor)


def test_shares_sell_at_their_price_now_so_a_building_satisfied_since_earns_more() -> None:
    growth = satisfied(grown(((5, 5), BuildingKind.FARM)), 0.0)
    growth.owners.buy((5, 5), BuildingKind.FARM)
    paid = -growth.railway.finances.month["shares bought"]
    satisfied(growth, 1.0)
    growth._months += HOLD_MONTHS

    growth.owners.sell((5, 5), BuildingKind.FARM)

    assert growth.railway.finances.month["shares sold"] == pytest.approx(1.5 * paid)
    assert growth.owners.shares(BuildingKind.FARM)[5, 5] == 0


@pytest.mark.parametrize(
    ("placed", "kind", "problem"),
    [
        ([((5, 5), BuildingKind.HOUSE)], BuildingKind.HOUSE, "A house cannot be bought"),
        (
            [((5, 5), BuildingKind.MINE)],
            BuildingKind.MINE,
            "You already own every share of the mines here",
        ),
    ],
)
def test_homes_cannot_be_bought_nor_more_than_stand_there(
    placed: list, kind: BuildingKind, problem: str
) -> None:
    growth = grown(*placed)
    growth.railway.finances.earn(1_000_000.0, "test")
    if kind == BuildingKind.MINE:
        growth.owners.own((5, 5), kind)

    said = growth.owners.buy((5, 5), kind)

    assert said == problem


def joined_with_owned_farm(shares: int, *houses: tuple[int, int]) -> Growth:
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
    growth.owners.own((6, 22), BuildingKind.FARM, shares)
    return growth


@pytest.mark.parametrize("shares", [LOT, 3 * LOT, SHARES])
def test_a_station_hands_out_the_players_part_of_their_farm_and_files_its_pay_as_owned(
    shares: int,
) -> None:
    growth = joined_with_owned_farm(shares, (6, 21), (6, 21), (6, 21))

    growth.trade_goods()

    assert growth.trade.handed[(6, 20)][Good.WHEAT] == pytest.approx(10 * shares / SHARES)
    assert growth.railway.finances.month["owned buildings"] > 0
    assert growth.railway.finances.month.get("cargo", 0.0) == 0.0


def test_an_owned_farm_far_from_any_station_pays_the_player_nothing() -> None:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    growth.railway = Railway(constructions)
    growth.build(Building((6, 10), BuildingKind.FARM))
    growth.build(Building((6, 11), BuildingKind.HOUSE))
    growth.owners.own((6, 10), BuildingKind.FARM)

    growth.trade_goods()

    assert not any(growth.railway.finances.month.values())


def test_a_crowded_out_farm_of_the_players_comes_down_with_their_shares() -> None:
    growth = grown(((5, 5), BuildingKind.FARM))
    growth.owners.own((5, 5), BuildingKind.FARM)
    neighbours = growth.constructions.tile_map.grid.neighbours(5, 5)
    for i in range(FARMS_CROWDED_OUT):
        growth.build(Building(neighbours[i % len(neighbours)], BuildingKind.HOUSE))

    growth._crowd_out()

    assert growth.count(BuildingKind.FARM).sum() == 0
    assert growth.owners.shares(BuildingKind.FARM).sum() == 0


def test_factories_merging_into_a_big_one_hand_their_owner_their_part_of_it() -> None:
    growth = grown(
        ((5, 5), BuildingKind.FACTORY),
        ((5, 6), BuildingKind.FACTORY),
        ((5, 6), BuildingKind.FACTORY),
    )
    growth.owners.own((5, 6), BuildingKind.FACTORY)

    growth._turn((5, 5), FACTORIES, np.random.default_rng(0))

    assert growth.count(BuildingKind.BIG_FACTORY)[5, 5] == 1
    assert growth.owners.shares(BuildingKind.BIG_FACTORY)[5, 5] == round(SHARES / 3)
    assert growth.owners.owned(BuildingKind.FACTORY).sum() == 0


def test_the_company_flies_its_flag_only_over_the_cells_where_it_owns_a_building() -> None:
    growth = grown(((5, 5), BuildingKind.FARM), ((5, 7), BuildingKind.FARM))

    growth.owners.buy((5, 7), BuildingKind.FARM)

    assert growth.owners.company == "The Trains Corp"
    assert growth.owners.cells() == [(5, 7)]


def test_the_players_farm_sells_its_goods_at_the_market_price_as_they_enter_the_station() -> None:
    growth = joined_with_owned_farm(SHARES)

    growth.trade_goods()

    sold = growth.railway.finances.month["owned buildings"]
    assert sold == pytest.approx(10 * price_of(Good.WHEAT))
    assert growth.railway.finances.month.get("cargo", 0.0) == 0.0


@pytest.mark.parametrize(("months", "said"), [(HOLD_MONTHS - 1, "from 1855"), (HOLD_MONTHS, None)])
def test_shares_can_be_sold_only_five_years_after_they_were_bought(
    months: int, said: str | None
) -> None:
    growth = satisfied(grown(((5, 5), BuildingKind.FARM)))
    growth.owners.buy((5, 5), BuildingKind.FARM)
    growth._months += months

    problem = growth.owners.sell((5, 5), BuildingKind.FARM)

    assert problem is None if said is None else said in str(problem)
