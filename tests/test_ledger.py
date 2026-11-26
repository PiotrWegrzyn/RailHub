import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.goods import MAKES, Good, freight_of, price_of
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.ownership import HOLD_MONTHS, LOT, SHARES
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route

FARM = (6, 22)
WHEAT_A_MONTH = MAKES[BuildingKind.FARM][Good.WHEAT]


def joined_with_owned_farm(shares: int) -> Growth:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, (6, 3))
    build_station(growth, (6, 20))
    route = plan_route(constructions, (6, 3), (6, 20))
    assert route is not None
    lay_route(growth, route)
    growth.railway = Railway(constructions)
    growth.build(Building(FARM, BuildingKind.FARM))
    growth.owners.own(FARM, BuildingKind.FARM, shares)
    return growth


def trade_for(growth: Growth, months: int) -> None:
    for _ in range(months):
        growth._months += 1
        growth.trade_goods()


def flat(growth: Growth, cell: tuple[int, int]) -> int:
    return int(np.ravel_multi_index(cell, growth.grid.shape))


def test_last_year_holds_a_whole_year_of_what_the_players_farm_sold_and_its_freight() -> None:
    growth = joined_with_owned_farm(SHARES)

    trade_for(growth, 13)

    taken = growth.ledger.last_year(BuildingKind.FARM, flat(growth, FARM))
    assert taken is not None
    assert taken.sold == pytest.approx(12 * WHEAT_A_MONTH * price_of(Good.WHEAT))
    assert taken.freight == pytest.approx(12 * WHEAT_A_MONTH * freight_of(Good.WHEAT))
    assert taken.output == pytest.approx(12 * WHEAT_A_MONTH * price_of(Good.WHEAT))
    assert taken.yours == pytest.approx(taken.sold + taken.freight)


def test_the_players_part_of_a_farm_they_own_a_tenth_of_is_a_tenth_of_its_output() -> None:
    growth = joined_with_owned_farm(LOT)

    trade_for(growth, 13)

    taken = growth.ledger.last_year(BuildingKind.FARM, flat(growth, FARM))
    assert taken is not None
    assert taken.sold == pytest.approx(12 * WHEAT_A_MONTH * LOT / SHARES * price_of(Good.WHEAT))
    assert taken.output == pytest.approx(12 * WHEAT_A_MONTH * price_of(Good.WHEAT))


def test_there_is_no_last_year_before_a_whole_year_of_trade() -> None:
    growth = joined_with_owned_farm(SHARES)

    trade_for(growth, 11)

    assert growth.ledger.last_year(BuildingKind.FARM, flat(growth, FARM)) is None


def test_the_forecast_is_the_last_quarter_for_a_whole_year() -> None:
    growth = joined_with_owned_farm(SHARES)

    trade_for(growth, 2)

    forecast = growth.ledger.forecast(BuildingKind.FARM, flat(growth, FARM))
    assert forecast is not None
    assert forecast.sold == pytest.approx(12 * WHEAT_A_MONTH * price_of(Good.WHEAT))
    assert forecast.freight == pytest.approx(12 * WHEAT_A_MONTH * freight_of(Good.WHEAT))


def test_a_farm_no_station_reaches_forecasts_its_output_but_nothing_for_the_player() -> None:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    growth.railway = Railway(constructions)
    growth.build(Building((6, 10), BuildingKind.FARM))
    growth.build(Building((6, 11), BuildingKind.HOUSE))
    growth.owners.own((6, 10), BuildingKind.FARM)

    trade_for(growth, 3)

    forecast = growth.ledger.forecast(BuildingKind.FARM, flat(growth, (6, 10)))
    assert forecast is not None
    assert forecast.output > 0
    assert forecast.yours == 0


def test_a_cell_of_three_bakeries_is_three_thousand_shares_in_all() -> None:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    for _ in range(3):
        growth.build(Building((5, 5), BuildingKind.BAKERY))

    total = growth.owners.total((5, 5), BuildingKind.BAKERY)

    assert total == 3 * SHARES


def test_only_shares_bought_five_years_ago_can_be_sold_now() -> None:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    growth.railway = Railway(constructions)
    growth.railway.finances.earn(1_000_000.0, "test")
    growth.build(Building((5, 5), BuildingKind.FARM))
    growth.owners.buy((5, 5), BuildingKind.FARM)
    growth._months += 12
    growth.owners.buy((5, 5), BuildingKind.FARM)
    growth.owners.buy((5, 5), BuildingKind.FARM)
    growth._months += HOLD_MONTHS - 12

    sellable = growth.owners.sellable((5, 5), BuildingKind.FARM)

    assert sellable == LOT
    assert growth.owners.sellable_from((5, 5), BuildingKind.FARM) == growth.start_year + 6


def test_a_buildings_satisfaction_is_its_own_over_the_months_of_the_last_quarter() -> None:
    growth = joined_with_owned_farm(SHARES)
    moods = []
    for _ in range(4):
        trade_for(growth, 1)
        position = int(np.searchsorted(growth.last_trade.cells, flat(growth, FARM)))
        moods.append(float(growth.last_trade.share((BuildingKind.FARM,))[position]))

    satisfaction = growth.ledger.satisfaction(BuildingKind.FARM, flat(growth, FARM))

    assert satisfaction == pytest.approx(sum(moods[-3:]) / 3)
