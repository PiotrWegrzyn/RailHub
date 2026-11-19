import pytest

from railroads.sim.clock import HISTORY_START
from railroads.sim.company import BOOK_SHARE
from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import SCARCE_WORTH
from railroads.sim.finance import dollars
from railroads.sim.goods import FREIGHT_SHARE, KIND_NEED_YEARS, Good, Need, needs_of
from railroads.sim.lots import Lots
from railroads.sim.ownership import EXPERIENCE_PER_POINT
from railroads.sim.tech import Tree
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

    assert row.makes == ((Good.WINE, 80),)
    assert {need: amount for need, amount, _ in row.wants} == {
        Need.GRAPES: 80.0,
        Need.BARRELS: 40.0,
    }


def test_the_help_says_when_a_need_begins() -> None:
    wants = {need: year for need, _, year in row_of(BuildingKind.HEADQUARTERS).wants}

    assert wants[Need.GUNS] == 0
    assert wants[Need.TANKS] == 1940


@pytest.mark.parametrize(
    ("kind", "travellers", "space", "year"),
    [
        (BuildingKind.HOUSE, 1, 1, 0),
        (BuildingKind.ST_PAULS, 2137, 1, 0),
        (BuildingKind.FACTORY, 10, 3, 1780),
        (BuildingKind.AIRPORT, 200, 1, 1920),
    ],
)
def test_the_help_gives_travellers_space_and_opening_year(
    kind: BuildingKind, travellers: int, space: int, year: int
) -> None:
    row = row_of(kind)

    assert (row.travellers, row.space, row.from_year) == (travellers, space, year)


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


def test_the_help_gives_every_tech_node_on_the_way_to_an_engine() -> None:
    row = engine_row("pacific")

    assert row.tech == ((Tree.METALS, ("Heavy boilers", "Roller bearings", "Superheaters")),)


def test_the_help_gives_the_tech_of_each_tree_an_engine_needs() -> None:
    row = engine_row("bullet")

    assert dict(row.tech) == {
        Tree.PASSENGERS: ("Timetables", "Dining cars", "Express trains"),
        Tree.MODERN: (
            "Telegraph signals",
            "Electric traction",
            "Relay interlocking",
            "Thyristor control",
        ),
    }


def test_the_help_links_an_engine_to_the_real_ones_it_is_drawn_after() -> None:
    row = engine_row("big_boy")

    assert row.origins == (
        ("Union Pacific Big Boy, 1941", "https://en.wikipedia.org/wiki/Union_Pacific_Big_Boy"),
    )


def test_only_the_super_heavy_diesel_is_drawn_after_no_real_engine() -> None:
    own = [row.name for row in engine_help() if not row.origins]

    assert own == ["super_diesel"]


def test_the_help_says_when_a_farm_starts_needing_what_only_farms_need_later() -> None:
    wants = {need: year for need, _, year in row_of(BuildingKind.FARM).wants}

    assert all(
        needs_of(BuildingKind.FARM, year - 1).get(need) is None
        and needs_of(BuildingKind.FARM, year).get(need) is not None
        for need, year in wants.items()
        if year
    )
    assert wants[Need.FUEL] == KIND_NEED_YEARS[(BuildingKind.FARM, Need.FUEL)]


def test_the_map_help_names_the_prices_the_first_three_lots_cost() -> None:
    prices = [dollars(Lots(10.0, (0, 0), bought=bought).price()) for bought in range(3)]

    text = " ".join(MAP_HELP)

    assert all(price in text for price in prices)
    assert f"from the year {HISTORY_START}" in text


def test_the_engines_help_says_experience_comes_only_from_deliveries_and_owned_buildings_add_to_it() -> (
    None
):
    share = f"{EXPERIENCE_PER_POINT:.0%}"

    text = ENGINES_NOTE

    assert "only from deliveries" in text
    assert f"adds {share} to it" in text


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
