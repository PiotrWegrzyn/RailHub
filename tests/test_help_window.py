import pytest

from railroads.sim.constructions import BuildingKind
from railroads.sim.goods import Good, Need
from railroads.ui.help_window import building_help, good_help


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


def test_the_help_gives_the_price_of_every_good() -> None:
    prices = {row.good: row.price for row in good_help()}

    assert set(prices) == set(Good)
    assert prices[Good.INCENSE] == 300


def test_the_help_says_who_makes_and_wants_a_good() -> None:
    honey = next(row for row in good_help() if row.good == Good.HONEY)

    assert honey.made_by == ("beekeeper",)
    assert "house" in honey.wanted_by
