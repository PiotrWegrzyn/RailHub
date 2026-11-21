import pytest

from railroads.sim.clock import HISTORY_START
from railroads.sim.company import BOOK_SHARE
from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import SCARCE_WORTH
from railroads.sim.finance import dollars
from railroads.sim.goods import FREIGHT_SHARE, Good, Need
from railroads.sim.lots import Lots
from railroads.sim.trains import ENGINES
from railroads.ui.help_window import (
    ENGINES_NOTE,
    GOAL_HELP,
    GOODS_NOTE,
    MAP_HELP,
    building_help,
    engine_help,
    good_help,
)


def row_of(kind: BuildingKind):
    return next(row for row in building_help() if row.kind == kind)


def test_the_help_lists_every_building_once() -> None:
    kinds = [row.kind for row in building_help()]

    assert sorted(kinds, key=lambda k: k.value) == sorted(BuildingKind, key=lambda k: k.value)


def test_the_help_gives_what_a_building_makes_and_wants_in_a_month() -> None:
    row = row_of(BuildingKind.WINERY)

    assert row.makes == ((Good.WINE, 20),)
    assert {need: amount for need, amount, _ in row.wants} == {
        Need.GRAPES: 80.0,
        Need.BARRELS: 40.0,
    }


def test_the_help_says_when_a_need_begins() -> None:
    wants = {need: since for need, _, since in row_of(BuildingKind.HEADQUARTERS).wants}

    assert wants[Need.GUNS] == "era 3"
    assert wants[Need.TANKS] == "era 5"


@pytest.mark.parametrize(
    ("kind", "travellers", "space", "since"),
    [
        (BuildingKind.HOUSE, 1, 1, ""),
        (BuildingKind.ST_PETERS, 2137, 1, ""),
        (BuildingKind.FACTORY, 10, 3, "1780"),
        (BuildingKind.AIRPORT, 200, 1, "1920"),
        (BuildingKind.STEEL_MILL, 30, 3, "era 2"),
    ],
)
def test_the_help_gives_travellers_space_and_when_it_opens(
    kind: BuildingKind, travellers: int, space: int, since: str
) -> None:
    row = row_of(kind)

    assert (row.travellers, row.space, row.since) == (travellers, space, since)


def test_the_help_gives_the_market_price_and_freight_of_every_good() -> None:
    prices = {row.good: (row.price, row.freight) for row in good_help()}

    assert set(prices) == set(Good)
    assert prices[Good.INCENSE] == (pytest.approx(1500), pytest.approx(300))


def test_the_help_says_who_makes_and_wants_a_good() -> None:
    honey = next(row for row in good_help() if row.good == Good.HONEY)

    assert honey.made_by == ("beekeeper",)
    assert "house" in honey.wanted_by


def test_the_help_tables_are_worked_out_once() -> None:
    buildings, goods = building_help(), good_help()

    again = building_help(), good_help()

    assert again[0] is buildings
    assert again[1] is goods


def engine_row(name: str):
    return next(row for row in engine_help() if row.name == name)


def test_the_help_lists_every_engine_by_year() -> None:
    rows = engine_help()

    assert sorted(row.name for row in rows) == sorted(ENGINES)
    assert [row.engine.year for row in rows] == sorted(row.engine.year for row in rows)


@pytest.mark.parametrize(
    ("name", "sold_until"),
    [
        ("lion", (1840, "Planet steam engine")),
        ("tank", (1862, "Light tank engine")),
        ("light_tank", (1935, "Railbus")),
        ("big_boy", (1960, "Heavy diesel")),
        ("modern_electric", None),
        ("super_diesel", None),
    ],
)
def test_the_help_says_until_when_an_engine_is_sold(name: str, sold_until) -> None:
    row = engine_row(name)

    assert row.sold_until == sold_until


def test_the_help_gives_the_supply_tile_level_that_opens_an_engine() -> None:
    row = engine_row("pacific")

    opened = row.opened_by

    assert opened == "coal supply level 20"


def test_the_help_links_an_engine_to_the_real_ones_it_is_drawn_after() -> None:
    row = engine_row("big_boy")

    assert row.origins == (
        ("Union Pacific Big Boy, 1941", "https://en.wikipedia.org/wiki/Union_Pacific_Big_Boy"),
    )


def test_only_the_super_heavy_diesel_is_drawn_after_no_real_engine() -> None:
    own = [row.name for row in engine_help() if not row.origins]

    assert own == ["super_diesel"]


def test_the_help_says_which_demand_tile_level_opens_what_a_farm_takes() -> None:
    wants = {need: since for need, _, since in row_of(BuildingKind.FARM).wants}

    assert wants[Need.FUEL] == "oil and fuel demand 11"
    assert wants[Need.LIMESTONE] == "ore and stone demand 3"


def test_the_map_help_names_the_prices_the_first_three_lots_cost() -> None:
    prices = [dollars(Lots(10.0, (0, 0), bought=bought).price()) for bought in range(3)]

    text = " ".join(MAP_HELP)

    assert all(price in text for price in prices)
    assert f"from the year {HISTORY_START}" in text


def test_the_engines_help_says_engines_come_from_supply_tile_levels() -> None:
    text = ENGINES_NOTE

    found = "supply tile reaches its level" in text

    assert found


def test_the_goods_help_gives_freight_as_a_share_of_the_market_price_up_to_the_scarce_worth() -> (
    None
):
    share, most = f"{FREIGHT_SHARE:.0%}", f"{SCARCE_WORTH:g}"

    text = GOODS_NOTE

    assert f"{share} of its market price" in text
    assert f"up to {most} times that" in text


def test_the_goal_help_counts_stations_track_and_trains_at_the_book_share() -> None:
    share = f"{BOOK_SHARE:.0%}"

    text = GOAL_HELP[0]

    assert f"plus {share} of what your stations" in text
