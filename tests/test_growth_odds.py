from dataclasses import replace

import numpy as np
import pytest
from growing import NO_TRADE, build, make_map, only

from railroads.sim.clock import MONTHS_PER_YEAR
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.growth import Growth
from railroads.sim.growth_odds import (
    ladder_odds,
    monthly_chances,
    other_ways,
    turn_instead,
    why_not_here,
    yearly,
)
from railroads.sim.ladder_types import Ladder
from railroads.sim.ladders import FISHERIES, FISHERIES_PER_PORT, PORTS
from railroads.sim.points import Points
from railroads.sim.surroundings import Land
from railroads.sim.tilemap import Relief, Terrain
from railroads.ui.growth_window import (
    GrowthOdds,
    cell_odds,
    per_year_text,
    percent_text,
    scale_place,
)

COAST = (5, 1)
NEXT_ON_COAST = (4, 1)
INLAND = (5, 5)
HARBOURS = Ladder(
    kinds=(BuildingKind.PORT,),
    land=Land(
        relief={Relief.MOUNTAINS: 0.0},
        water={Terrain.LAND: 0.0, Terrain.SEA: 1.0, Terrain.LAKE: 0.3},
    ),
    base=8,
    needs={Points.POPULATION: 2},
    most_nearby=1,
)


def coastal_town(port: tuple[int, int] | None = None) -> Growth:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    for cell in [COAST, *tile_map.grid.neighbours(*COAST)]:
        if tile_map.terrain[cell] == Terrain.LAND:
            build(constructions, cell, BuildingKind.HOUSE, 1)
    if port is not None:
        build(constructions, port, BuildingKind.PORT, 1)
    return Growth(constructions, only(HARBOURS))


def test_a_coast_with_nobody_around_has_no_chance_of_a_port_for_want_of_people() -> None:
    growth = Growth(Constructions(make_map()), only(HARBOURS))

    reasons = why_not_here(growth, HARBOURS, COAST)

    assert not monthly_chances(growth, HARBOURS).any()
    assert reasons == ["Needs 2 population points nearby; there are 0."]


def test_the_map_says_why_no_port_can_come_up_anywhere() -> None:
    growth = Growth(Constructions(make_map()), only(HARBOURS))

    odds = ladder_odds(growth, HARBOURS, monthly_chances(growth, HARBOURS))

    assert odds.per_year == 0
    assert odds.best is None
    assert odds.why_none == (
        "Needs 2 population points nearby, and no land that suits it has them."
    )


def test_a_coastal_town_has_its_turn_chance_of_a_port() -> None:
    growth = coastal_town()

    monthly = monthly_chances(growth, HARBOURS)

    flat = COAST[0] * growth.grid.cols + COAST[1]
    assert monthly[flat] == pytest.approx(growth.chances(HARBOURS)[COAST])
    assert monthly[flat] > 0
    assert why_not_here(growth, HARBOURS, COAST) == []


def test_the_map_counts_a_year_of_every_cells_chances() -> None:
    growth = coastal_town()
    monthly = monthly_chances(growth, HARBOURS)

    odds = ladder_odds(growth, HARBOURS, monthly)

    assert odds.per_year == pytest.approx(monthly.sum() * MONTHS_PER_YEAR)
    assert odds.best is not None
    assert growth.grid.cols * odds.best[0] + odds.best[1] == int(np.argmax(monthly))


def test_a_port_leaves_no_chance_of_another_beside_it() -> None:
    growth = coastal_town(port=COAST)

    monthly = monthly_chances(growth, HARBOURS)

    flat = NEXT_ON_COAST[0] * growth.grid.cols + NEXT_ON_COAST[1]
    assert monthly[flat] == 0
    assert "1 stand nearby already, the most a neighbourhood holds." in why_not_here(
        growth, HARBOURS, NEXT_ON_COAST
    )


@pytest.mark.parametrize(
    ("cell", "reason"),
    [
        (INLAND, "Only beside the sea or a lake."),
        ((5, 0), "Nothing grows on the water."),
        ((5, 11), "Not in mountains."),
    ],
)
def test_the_land_says_why_it_suits_no_port(cell: tuple[int, int], reason: str) -> None:
    growth = coastal_town()

    reasons = why_not_here(growth, HARBOURS, cell)

    assert reason in reasons


def test_a_ladder_not_yet_open_says_when_it_grows() -> None:
    later = replace(HARBOURS, from_year=3000)
    growth = Growth(Constructions(make_map()), only(later))

    odds = ladder_odds(growth, later, monthly_chances(growth, later))

    assert odds.why_none == "Grows from 3000."


def test_ports_also_come_from_fisheries_joining() -> None:
    growth = Growth(Constructions(make_map()), NO_TRADE)

    ways = other_ways(growth, PORTS)

    assert any(f"fishery x{FISHERIES_PER_PORT} in a neighbourhood join" in way for way in ways)


def test_a_turn_among_enough_fisheries_joins_them_instead() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    for cell in [(4, 1), (5, 1), (6, 1)]:
        build(constructions, cell, BuildingKind.FISHERY, 1)
    growth = Growth(constructions, NO_TRADE)

    instead = turn_instead(growth, FISHERIES, COAST)

    assert instead == f"A turn here joins fishery x{FISHERIES_PER_PORT} into a port."


def test_a_year_is_twelve_months_of_chances() -> None:
    chances = yearly(np.array([0.0, 0.5, 1.0]))

    assert chances.tolist() == pytest.approx([0.0, 1 - 0.5**12, 1.0])


@pytest.mark.parametrize(
    ("per_year", "text"),
    [
        (0.0, "none"),
        (3.25, "up to 3.2 a year"),
        (0.01, "up to once in 100 years"),
        (0.0001, "under once in 1,000 years"),
    ],
)
def test_how_many_a_year_reads(per_year: float, text: str) -> None:
    assert per_year_text(per_year) == text


@pytest.mark.parametrize(
    ("chance", "text"),
    [
        (0.0, "0%"),
        (0.25, "25%"),
        (0.029, "2.9%"),
        (0.000123, "0.012%"),
        (0.0000000055, "under 0.0001%"),
        (1.0, "100%"),
    ],
)
def test_a_chance_reads_as_a_percentage(chance: float, text: str) -> None:
    assert percent_text(chance) == text


def test_the_scale_runs_from_a_thousandth_of_the_top_to_the_top() -> None:
    places = scale_place(np.array([0.1, 0.01, 0.0001, 0.00001]), 0.1)

    assert places.tolist() == pytest.approx([1.0, 2 / 3, 0.0, 0.0])


def test_a_cells_details_say_why_the_lens_kind_cant_come_up_there() -> None:
    growth = coastal_town()

    shown = cell_odds(growth, GrowthOdds(), INLAND, HARBOURS)

    assert shown.lens is not None
    name, chance, reasons, _ = shown.lens
    assert (name, chance) == ("port", 0.0)
    assert "Only beside the sea or a lake." in reasons


def test_a_cells_details_list_the_likeliest_kinds_first() -> None:
    growth = coastal_town()

    shown = cell_odds(growth, GrowthOdds(), COAST, None)

    chances = [chance for _, chance in shown.likeliest]
    assert shown.lens is None
    assert chances == sorted(chances, reverse=True)
    assert chances[0] > 0
